"""Space safety, rollback and disposable-outgoing-store tests for V86."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from patchlib.disk_health_v86 import (
    DiskPressureError, cleanup_working_orphans, require_space, snapshot_cache_report)
from patchlib.elastic import EventJournal
from patchlib.shards import export_state_sharded, read_manifest, shard_map, verify_shards
from patchlib.snapshot_offer_gc_v86 import collect
from patchlib.live_telemetry import LocalTelemetry, recent_samples


def test_signed_archive_fails_before_full_disk_no_loss(tmp_path, monkeypatch):
    peer='p' + 'a'*32
    journal=EventJournal(tmp_path/'live.sqlite3',hashlib.sha256(b'g').hexdigest(),max_bytes=0)
    import patchlib.disk_health_v86 as dh
    monkeypatch.setattr(dh,'free_bytes',lambda _: 0)
    try:
        before=journal.total_bytes()
        with pytest.raises(DiskPressureError,match='DISK PRESSURE'):
            journal.append(peer,1,b'accepted-signed-payload')
        assert journal.count()==0
        assert journal.total_bytes()==before
        with pytest.raises(DiskPressureError):
            journal.append_many([(peer,1,b'one'),(peer,2,b'two')])
        assert journal.count()==0
    finally:
        journal.close()


def test_disposable_offers_gc_preserves_last_full_model(tmp_path):
    from patchlib.core import UniversalAdapter
    torch.manual_seed(5)
    model=nn.Sequential(nn.Linear(3,3))
    adapter=UniversalAdapter(model,'m','v',rank=2)
    root=tmp_path/'offers'
    manifests=root/'manifests'
    store=root/'store'
    manifests.mkdir(parents=True)
    peer='p'+'a'*32
    older=manifests / f"offer-{peer}-1-{'a'*16}.fmanifest"
    newer=manifests / f"offer-{peer}-2-{'b'*16}.fmanifest"
    export_state_sharded(model,store,older,model_id='m',revision='v',peer_id=peer,seq=1,
                         adapter=adapter,shard_bytes=65536)
    old_objects=set(shard_map(read_manifest(older)))
    with torch.no_grad():
        model[0].parametrizations.weight.original.add_(1.0)
    export_state_sharded(model,store,newer,model_id='m',revision='v',peer_id=peer,seq=2,
                         adapter=adapter,shard_bytes=65536)
    new_objects=set(shard_map(read_manifest(newer)))
    assert old_objects != new_objects
    os.utime(older,ns=(1_000_000_000,)*2)
    assert collect(root,busy=True)==(0,0)
    assert older.exists()
    removed=collect(root)
    assert removed[0]==1
    assert not older.exists()
    assert newer.exists()
    verify_shards(read_manifest(newer),store)
    assert collect(root)==(0,0)


def test_working_moment_gc_only_removes_unreferenced_folders(tmp_path):
    j=tmp_path/'test.sqlite3'
    target=Path(str(j)+'.working.safetensors')
    active=tmp_path/(target.name+'.moments-'+('a'*32))
    stale=tmp_path/(target.name+'.moments-'+('b'*32))
    active.mkdir();stale.mkdir()
    (active/'m.f32').write_bytes(b'keep')
    (stale/'m.f32').write_bytes(b'old')
    save_file({'f.0':torch.zeros(1)},str(target),metadata={
        'kcia':json.dumps({'disk_optimizer':{'folder':active.name}})})
    assert cleanup_working_orphans(j)==1
    assert active.exists() and not stale.exists()
    assert cleanup_working_orphans(j)==0


def test_bound_telemetry_keeps_complete_recent_rows(tmp_path):
    path=tmp_path/'train.jsonl'
    t=LocalTelemetry(path,append=True)
    t.MAX_LOG_BYTES=1800
    t.KEEP_LOG_BYTES=900
    for i in range(50):
        t.record(steps=i,loss=float(i),emitted=i,inbound_applied=0,
                 inbox=0,channels=0,force=True)
    t.close()
    assert path.stat().st_size <= 1800
    rows=recent_samples(path,max_samples=20)
    assert rows and rows[-1]['steps']==49
    assert rows[0]['steps'] < rows[-1]['steps']


def test_journal_checkpoint_truncate_is_sqlite_managed(tmp_path):
    j=EventJournal(tmp_path/'db.sqlite3','a'*64)
    try:
        peer='p'+'a'*32
        j.append(peer,1,b'one')
        result=j.truncate_wal_if_idle()
        assert len(result)==3
        assert j.count()==1 and j.get(peer,1)
    finally:
        j.close()


def test_report_does_not_touch_files(tmp_path):
    j=tmp_path/'journal.sqlite3'
    log=Path(str(j)+'.fplog')
    log.write_bytes(b'x'*10)
    assert snapshot_cache_report(j)['signed journal']==10
    assert log.read_bytes()==b'x'*10


def test_outgoing_store_separated_from_received_peer_objects(tmp_path):
    import asyncio
    from patchlib.core import UniversalAdapter
    from patchlib.snapshot_live import LiveSnapshotLink
    from test_live_snapshot import FakeChannel
    from patchlib.roomid import auto_identity

    async def run():
        peer, signer, _, verifier = auto_identity(tmp_path/'identity.pem')
        model=nn.Sequential(nn.Linear(2,2))
        adapter=UniversalAdapter(model,'sep','v',rank=2)
        outgoing=tmp_path/'owned-offers'/'store'
        manifest=tmp_path/'snapshot.fmanifest'
        export_state_sharded(model,outgoing,manifest,model_id='sep',revision='v',
                             peer_id=peer,adapter=adapter,shard_bytes=65536)
        # No outgoing shard may ever be stored in the incoming/epoch folder.
        incoming_store=tmp_path/'incoming-epoch-store'
        alice=LiveSnapshotLink(incoming_store,tmp_path/'a',32*1024**2,
                               lambda:verifier,outgoing_store=outgoing)
        bob=LiveSnapshotLink(tmp_path/'bobstore',tmp_path/'b',32*1024**2,
                             lambda:verifier)
        a,b=FakeChannel(),FakeChannel()
        a.other=b;b.other=a
        alice.attach(a);bob.attach(b)
        alice.on_authenticated(signer);bob.on_authenticated(signer)
        try:
            alice.offer(manifest,signer)
            ready=await asyncio.wait_for(bob.completed.get(),timeout=10)
            verify_shards(read_manifest(ready),tmp_path/'bobstore')
            assert not (incoming_store/'objects').exists()
        finally:
            a.close();b.close()
            await asyncio.sleep(0)
    asyncio.run(run())


def test_offline_reclaim_keeps_active_journal(tmp_path):
    from patchlib.disk_health_v86 import reclaim
    jpath=tmp_path/'keep.sqlite3'
    j=EventJournal(jpath,'a'*64)
    peer='p'+'a'*32
    j.append(peer,1,b'keep-this-signed-event')
    # Running trainer holds an exclusive lock; even manual command cannot GC.
    with pytest.raises(RuntimeError, match='running'):
        reclaim(jpath)
    j.close()
    stats=reclaim(jpath)
    assert stats['orphan optimizer backups']==0
    reopened=EventJournal(jpath,'a'*64)
    try:
        assert reopened.get(peer,1)==b'keep-this-signed-event'
    finally:
        reopened.close()


def test_gui_disk_report_includes_separate_model_objects(tmp_path):
    from patchlib.disk_health_v86 import snapshot_cache_report
    room=tmp_path/'.kcia-gui'/'rooms'/'r1'
    profile='peer-'+'a'*16
    journal=room/'peers'/profile/'events.sqlite3'
    journal.parent.mkdir(parents=True)
    model_file=room/'models'/profile/'objects'/'abc.shard'
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b'x'*5000)
    checkpoint=room/'checkpoints'/profile/'one.fmanifest'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'y'*1000)
    result=snapshot_cache_report(journal)
    assert result['GUI model object store']==5000
    assert result['GUI checkpoint manifests']==1000
