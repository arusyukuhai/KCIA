"""Destructive, real-payload tests for certified compact v2 old-journal GC."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from patchlib.core import PatchError, _canonical_json
from patchlib.elastic import EventJournal
from patchlib.epoch_carry_v46 import CrossEpochCarryArchive
from patchlib.epoch_live_rebase_v52 import StreamingCertifiedRebase
from patchlib.epoch_hot_swap_v45 import durable_pointer
from patchlib.epoch_trainer_boot_v52 import resolve_certified_boot, verify_boot_carry
from patchlib.live_epoch_switch_v48 import certified_genesis
from patchlib.compact_journal_gc_v88 import collect_certified_compact_source
from patchlib.shards import export_state_sharded, read_manifest, semantic_shard_sha256
from test_epoch_live_rebase_v52 import setup


async def setup_certified(tmp_path, *, with_carry=2):
    source,unused,emit,maker=setup(tmp_path)
    unused.journal.close()
    root=tmp_path/'epochs'
    initial=tmp_path/'initial.fmanifest'
    export_state_sharded(source.adapter.model,tmp_path/'original-store',initial,
        model_id='tiny',revision='v1',peer_id='alice',seq=0,adapter=source.adapter,
        shard_bytes=65536)
    digest=hashlib.sha256(initial.read_bytes()).hexdigest()
    view=root/'checkpoints'/'views'/digest
    view.mkdir(parents=True)
    shutil.copy2(initial,view/'checkpoint.fmanifest')
    shutil.copytree(tmp_path/'original-store'/'objects',view/'objects')
    proposal={'room':'lab','epoch':1,'parent':source.genesis,'manifest':digest,
        'semantic':semantic_shard_sha256(read_manifest(initial)),
        'cut':hashlib.sha256(b'').hexdigest(),'cutoffs':{},'replicas':['alice','bob','charlie']}
    genesis=certified_genesis(proposal)
    target=maker('next',genesis,'bob')
    new_journal=EventJournal(root/'journals'/'epoch-00000001.sqlite3',genesis,max_bytes=0)
    carry=CrossEpochCarryArchive(root/'carries'/'epoch-00000001.sqlite3',
        source_genesis=source.genesis,target_genesis=genesis,
        certified_manifest=digest,epoch=1,cutoffs={})
    assert carry.replay(target)==0
    for i in range(with_carry):emit()
    rebaser=StreamingCertifiedRebase(source.journal,carry,target,proposal)
    rebaser.validate_cut()
    assert (await rebaser.finish_under_ingress_barrier())['total_archived']==with_carry
    durable_pointer(root/'active-epoch.json',epoch=1,genesis=genesis,checkpoint_hash=digest)
    class Ledger:
        genesis=source.genesis
        def certified_epoch(self,i):return {'proposal':proposal} if i==1 else None
    boot=resolve_certified_boot(root=root,journal_base=source.journal.path,
        ledger=Ledger(),root_genesis=source.genesis)
    assert verify_boot_carry(boot,carry,old_journal=source.journal)
    return source,target,carry,boot,new_journal,emit


def close_all(source,target,carry,new_journal):
    carry.close();new_journal.close();target.journal.close();source.journal.close()


@pytest.mark.asyncio
async def test_signed_suffix_survives_old_log_removal_and_restart(tmp_path):
    source,target,carry,boot,new_journal,emit=await setup_certified(tmp_path)
    original_path=source.journal.path
    close_all(source,target,carry,new_journal)
    assert collect_certified_compact_source(boot,active_pointer=tmp_path/'epochs'/'active-epoch.json')
    assert not original_path.exists()
    assert not Path(str(original_path)+'.fplog').exists()
    assert Path(str(original_path)+'.retired.json').is_file()
    # Enforce that the certified old journal cannot be reopened and recreated.
    with pytest.raises(PatchError,match='retired'):
        EventJournal(original_path,boot.previous_genesis,max_bytes=0)
    carry=CrossEpochCarryArchive(boot.carry_path,source_genesis=boot.previous_genesis,
        target_genesis=boot.current_genesis,certified_manifest=boot.manifest_hash,
        epoch=boot.epoch,cutoffs=boot.proposal['cutoffs'])
    assert verify_boot_carry(boot,carry,old_journal=None)
    # Late signed old-genesis updates can be archived AFTER the original
    # handoff and no longer invalidate its certified immutable prefix.
    carry.journal.append('alice',3,b'late-signed-carry-test')
    assert verify_boot_carry(boot,carry,old_journal=None)
    carry.close()
    assert collect_certified_compact_source(boot,active_pointer=tmp_path/'epochs'/'active-epoch.json')


@pytest.mark.asyncio
async def test_gc_does_not_touch_active_writer_or_uncovered_carry(tmp_path):
    source,target,carry,boot,new_journal,emit=await setup_certified(tmp_path)
    path=source.journal.path
    assert not collect_certified_compact_source(boot,active_pointer=tmp_path/'epochs'/'active-epoch.json')
    assert path.is_file() and not Path(str(path)+'.retired.json').exists()
    # Missing a post-cut event from the archive must cause fail-closed GC.
    carry.journal.db.execute('DELETE FROM events WHERE seq=2')
    carry.journal.db.commit()
    close_all(source,target,carry,new_journal)
    with pytest.raises(PatchError,match='archive|carry|mutated|uncovered'):
        collect_certified_compact_source(boot,active_pointer=tmp_path/'epochs'/'active-epoch.json')
    assert path.is_file() and not Path(str(path)+'.retired.json').exists()


@pytest.mark.asyncio
async def test_gc_refuses_certified_pointer_substitution(tmp_path):
    source,target,carry,boot,new_journal,emit=await setup_certified(tmp_path)
    close_all(source,target,carry,new_journal)
    root=tmp_path/'epochs'
    (root/'active-epoch.json').write_text(json.dumps({'format':'kcia-hot-pointer-v1',
        'epoch':1,'genesis':'f'*64,'checkpoint':boot.manifest_hash}))
    with pytest.raises(PatchError,match='pointer'):
        collect_certified_compact_source(boot,active_pointer=root/'active-epoch.json')
    assert boot.source_path.is_file()
