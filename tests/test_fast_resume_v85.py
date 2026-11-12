from pathlib import Path
"""v85: materialized committed bases avoid full signed-history replay."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from patchlib.core import UniversalAdapter, PatchError
from patchlib.continual import ContinualElasticEngine
from patchlib.elastic import EventJournal
from patchlib.fast_resume_v85 import save, restore, root_for
from test_livefed import Tiny

G = hashlib.sha256(b'v85-committed-base-tests').hexdigest()


def make(tmp, peer, initial):
    adapter = UniversalAdapter(deepcopy(initial), 'test', 'init', kind='loha', rank=3)
    j = EventJournal(tmp / (peer + '.db'), G, max_bytes=0)
    return ContinualElasticEngine(adapter, peer, G, j, unsafe=True, reinit_every=2)


def train_and_seal(e):
    opt = torch.optim.AdamW(e.adapter.trainable_parameters(), lr=.001)
    with e.adapter.lock:
        e.adapter.model().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return e.seal_local(1)


def output(e):
    with torch.no_grad():
        return e.adapter.model.proj(e.adapter.model.embed(torch.tensor([[0,1,2]]))).detach().clone()


def capture(e):
    cursor=e.journal.fast_resume_cursor()
    save(e.journal,e,max_state_bytes=16*1024**2,shard_bytes=1024**2,
         min_free_bytes=0,cursor=cursor)
    assert cursor==e.journal.fast_resume_cursor()


def test_skip_all_replay_and_exact_weights_and_anchors(tmp_path):
    torch.manual_seed(100)
    base=Tiny()
    e=make(tmp_path,'alice',base)
    train_and_seal(e)
    train_and_seal(e) # ReLoRA reset
    capture(e)
    expected = output(e)
    anchors = [[getattr(g.overlay.anchor,f'part{j}').clone() for j in range(len(g.overlay.local.parts))] for g in e.adapter.groups]
    e.journal.close()
    resumed=make(tmp_path,'alice',base)
    cursor=restore(resumed.journal,resumed,max_state_bytes=16*1024**2)
    assert cursor==2
    assert resumed.replay_suffix(cursor)==0
    torch.testing.assert_close(output(resumed), expected,rtol=0,atol=0)
    for g,old in zip(resumed.adapter.groups,anchors):
        for j,a in enumerate(old):
            torch.testing.assert_close(getattr(g.overlay.anchor,f'part{j}'),a,rtol=0,atol=0)
    resumed.journal.close()


def test_only_crash_suffix_is_replayed(tmp_path):
    torch.manual_seed(99)
    base=Tiny()
    e=make(tmp_path,'alice',base)
    train_and_seal(e)
    capture(e)
    train_and_seal(e)
    train_and_seal(e)
    expected=output(e)
    e.journal.close()
    resumed=make(tmp_path,'alice',base)
    cursor=restore(resumed.journal,resumed,max_state_bytes=16*1024**2)
    assert cursor==1
    assert resumed.replay_suffix(cursor)==2
    torch.testing.assert_close(output(resumed),expected,atol=0,rtol=0)
    resumed.journal.close()


def test_stale_index_cache_forces_full_replay_and_does_not_delete_journal(tmp_path):
    torch.manual_seed(101)
    base=Tiny()
    e=make(tmp_path,'alice',base)
    train_and_seal(e)
    capture(e)
    base_bytes=e.journal.total_bytes()
    e.journal.db.execute('DELETE FROM events')
    e.journal.db.commit()
    assert restore(e.journal,e,max_state_bytes=16*1024**2) is None
    assert e.journal.total_bytes()==base_bytes  # cache never edits journal data
    e.journal.close()


def test_bad_missing_snapshot_and_unsafe_path_do_not_load(tmp_path):
    torch.manual_seed(102)
    base=Tiny()
    e=make(tmp_path,'alice',base)
    train_and_seal(e)
    capture(e)
    p=root_for(e.journal)/'latest.json'
    info=json.loads(p.read_text())
    (root_for(e.journal)/'manifests'/(info['manifest']+'.fmanifest')).unlink()
    assert restore(e.journal,e,max_state_bytes=16*1024**2) is None
    e.journal.close()


def test_old_private_cache_gc_keeps_new_and_signed_archive(tmp_path):
    torch.manual_seed(103)
    base=Tiny()
    e=make(tmp_path,'alice',base)
    train_and_seal(e)
    capture(e)
    first=json.loads((root_for(e.journal)/'latest.json').read_text())
    train_and_seal(e)
    capture(e)
    latest=json.loads((root_for(e.journal)/'latest.json').read_text())
    assert latest['manifest']!=first['manifest']
    assert not (root_for(e.journal)/'manifests'/(first['manifest']+'.fmanifest')).exists()
    assert (root_for(e.journal)/'manifests'/(latest['manifest']+'.fmanifest')).exists()
    assert e.journal.count()==2 and e.journal.log_path.is_file()
    e.journal.close()


def test_virtual_factor_ledger_materialized_restart(tmp_path):
    from test_factor_ledger_v62 import build
    from patchlib.streaming_base import materialize_meta_from_safetensors
    from patchlib.layer_offload import install_disk_linear_offload
    from patchlib.factor_ledger_v62 import attach_factor_ledger
    torch.manual_seed(38)
    source,m,a,ledger,reference,b=build(tmp_path/'built')
    j=EventJournal(tmp_path/'virtual.sqlite3',G,max_bytes=0)
    eng=ContinualElasticEngine(a,'alice',G,j,reinit_every=2,unsafe=True)
    x=torch.randn(2,10)
    opt=torch.optim.AdamW(a.trainable_parameters(),lr=.01)
    m(x).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
    eng.seal_local(1)
    expected=m(x).detach().clone()
    assert ledger.stats()['ledger_groups']==2
    capture(eng)
    j.close()
    # Relaunch from the same pristine read-only pretrained model mapping.
    with torch.device('meta'):
        m2=type(m)()
    materialize_meta_from_safetensors(m2,source,writable=False)
    a2=UniversalAdapter(m2,'test','v62',kind='loha',rank=3,target_scope='auto')
    install_disk_linear_offload(m2,a2,tmp_path/'new-disk',min_weight_mib=0.)
    attach_factor_ledger(m2,a2,scratch_dir=tmp_path/'new-scratch')
    j2=EventJournal(tmp_path/'virtual.sqlite3',G,max_bytes=0)
    eng2=ContinualElasticEngine(a2,'alice',G,j2,reinit_every=2,unsafe=True)
    assert restore(j2,eng2,max_state_bytes=16*1024**2)==1
    assert eng2.replay_suffix(1)==0
    torch.testing.assert_close(m2(x),expected,atol=2e-5,rtol=2e-5)
    j2.close()


def test_retired_generation_source_requires_durable_gc_proof(tmp_path):
    from patchlib.epoch_trainer_boot_v52 import CertifiedEpochBoot,verify_boot_carry
    from patchlib.epoch_carry_v46 import CrossEpochCarryArchive
    from patchlib.core import _canonical_json
    prev='a'*64; nxt='b'*64; manifest='c'*64
    carry=CrossEpochCarryArchive(tmp_path/'carry.sqlite3',source_genesis=prev,
        target_genesis=nxt,certified_manifest=manifest,epoch=1,cutoffs={})
    source=tmp_path/'deleted-source.sqlite3'
    target=tmp_path/'successor.sqlite3'
    boot=CertifiedEpochBoot(1,nxt,prev,manifest,{'cutoffs':{},'cut':hashlib.sha256(b'').hexdigest()},
        tmp_path/'manifest',tmp_path/'store',target,carry.journal.path,source)
    marker={'epoch':1,'manifest':manifest,'new_journal':str(target.resolve())}
    Path(str(source)+'.retired.json').write_bytes(_canonical_json(marker))
    receipt={'format':'kagami-local-carry-attestation-v1',
             'source_genesis':prev,'target_genesis':nxt,
             'certified_manifest':manifest,'epoch':1,'source_rowid':0,
             'total_archived':0,'archive_digest':hashlib.sha256(b'').hexdigest(),
             'archive_genesis':carry.genesis}
    Path(str(carry.journal.path)+'.handoff.json').write_bytes(_canonical_json(receipt))
    assert verify_boot_carry(boot,carry,old_journal=None)
    Path(str(source)+'.retired.json').write_bytes(b'{}')
    with pytest.raises(PatchError,match='tombstone'):
        verify_boot_carry(boot,carry,old_journal=None)
    carry.close()
