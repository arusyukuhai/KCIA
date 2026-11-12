"""v96: do not replay historical signed carry bodies after materialization."""
import pytest
import json
import torch
from pathlib import Path
from patchlib.fast_resume_v85 import save, restore, root_for
from patchlib.epoch_carry_v46 import CrossEpochCarryArchive
from patchlib.identity import signer_from_pem
from test_epoch_live_rebase_v52 import setup


def archive_for(path, source_genesis, target_genesis):
    return CrossEpochCarryArchive(path, source_genesis=source_genesis,
        target_genesis=target_genesis, certified_manifest='c'*64,
        epoch=1, cutoffs={})


def capture(engine, archive):
    j=engine.journal
    return save(j, engine, cursor=j.fast_resume_cursor(),
        carry_archive=archive, carry_cursor=archive.journal.fast_resume_cursor(),
        max_state_bytes=16*1024**2, shard_bytes=65536, min_free_bytes=0)


def get_base(engine):
    return engine.adapter.model.layer.parametrizations.weight.original.detach().clone()


def current_update(tmp_path, maker, genesis):
    signer=signer_from_pem(tmp_path/'p.key')
    current=maker('current-writer',genesis,'alice',signer)
    opt=torch.optim.AdamW(current.adapter.trainable_parameters(),lr=.01)
    current.adapter.model().backward();opt.step();opt.zero_grad()
    wire=current.seal_local(1)
    return current,wire


def prepare(tmp_path, *, with_current=False):
    old,target,emit,maker=setup(tmp_path)
    archive=archive_for(tmp_path/'carry.db',old.genesis,target.genesis)
    assert archive.replay(target)==0
    assert archive.accept(target,emit())
    current=None
    if with_current:
        current,wire=current_update(tmp_path,maker,target.genesis)
        assert target.ingest(wire)
    capture(target,archive)
    expected=get_base(target)
    archive.close();target.journal.close()
    return old,maker,expected,current


def resume(tmp_path, old, maker):
    recovered=maker('target', 'b'*64, 'bob')
    archived=archive_for(tmp_path/'carry.db',old.genesis,recovered.genesis)
    return recovered,archived


def test_nonempty_carry_zero_replay_and_same_model(tmp_path):
    old,maker,expected,current=prepare(tmp_path,with_current=True)
    recovered,archived=resume(tmp_path,old,maker)
    selected=restore(recovered.journal,recovered,carry_archive=archived,max_state_bytes=16*1024**2)
    assert selected == (1,1)
    assert archived.resume_suffix(recovered,selected[1])==0
    assert recovered.replay_suffix(selected[0])==0
    torch.testing.assert_close(get_base(recovered),expected,rtol=0,atol=0)
    assert archived._replayed and archived._replayed_count==1
    archived.close();recovered.journal.close();old.journal.close();current.journal.close()


def test_new_carry_suffix_replayed_without_old_bodies(tmp_path):
    old,maker,expected,current=prepare(tmp_path)
    # A later signed old-generation event arrives after checkpoint publication.
    late=emit_later(old)
    archived_write=archive_for(tmp_path/'carry.db',old.genesis,'b'*64)
    writer=maker('target','b'*64,'bob')
    assert archived_write.replay(writer)==1
    assert archived_write.accept(writer,late)
    target_expected=get_base(writer)
    archived_write.close();writer.journal.close()
    recovered,archived=resume(tmp_path,old,maker)
    selected=restore(recovered.journal,recovered,carry_archive=archived,max_state_bytes=16*1024**2)
    assert selected==(0,1)
    assert archived.resume_suffix(recovered,selected[1])==1
    assert recovered.replay_suffix(selected[0])==0
    torch.testing.assert_close(get_base(recovered),target_expected,atol=1e-6,rtol=1e-6)
    archived.close();recovered.journal.close();old.journal.close()


def emit_later(old):
    opt=torch.optim.AdamW(old.adapter.trainable_parameters(),lr=.01)
    old.adapter.model().backward();opt.step();opt.zero_grad()
    return old.seal_local(1)


def test_both_lanes_changed_falls_back_before_mutation(tmp_path):
    old,maker,expected,current=prepare(tmp_path)
    oldwire=emit_later(old)
    newwriter,newwire=current_update(tmp_path,maker,'b'*64)
    archived_write=archive_for(tmp_path/'carry.db',old.genesis,'b'*64)
    write=maker('target','b'*64,'bob')
    assert archived_write.replay(write)==1
    assert archived_write.accept(write,oldwire)
    assert write.ingest(newwire)
    archived_write.close();write.journal.close()
    recovered,archived=resume(tmp_path,old,maker)
    pristine=get_base(recovered)
    assert restore(recovered.journal,recovered,carry_archive=archived,max_state_bytes=16*1024**2) is None
    torch.testing.assert_close(get_base(recovered),pristine,rtol=0,atol=0)
    archived.close();recovered.journal.close();old.journal.close();newwriter.journal.close()


def test_old_unbound_cache_cannot_assume_carry_empty(tmp_path):
    old,maker,expected,current=prepare(tmp_path)
    p=root_for(type('J',(),{'path':tmp_path/'target.db'})())/'latest.json'
    pointer=json.loads(p.read_text());pointer.pop('carry');p.write_text(json.dumps(pointer))
    recovered,archived=resume(tmp_path,old,maker)
    assert restore(recovered.journal,recovered,carry_archive=archived,max_state_bytes=16*1024**2) is None
    archived.close();recovered.journal.close();old.journal.close()


def test_carry_index_mutation_invalidates_snapshot(tmp_path):
    old,maker,expected,current=prepare(tmp_path)
    recovered,archived=resume(tmp_path,old,maker)
    archived.journal.db.execute('DELETE FROM events')
    archived.journal.db.commit()
    assert restore(recovered.journal,recovered,carry_archive=archived,max_state_bytes=16*1024**2) is None
    archived.close();recovered.journal.close();old.journal.close()


def test_signed_carry_receipt_verified_prefix_is_persisted_and_skips_scan(tmp_path):
    import hashlib
    from patchlib.elastic import EventJournal
    from patchlib.core import _canonical_json, PatchError
    from patchlib.compact_journal_gc_v88 import verify_receipt_prefix
    archive=EventJournal(tmp_path/'prefix.db','c'*64,max_bytes=0)
    entries=[]
    for seq in range(1,5):
        data=(f'signed-test-fixture-{seq}').encode()
        assert archive.append('alice',seq,data)
        entries.append(('alice',seq,hashlib.sha256(data).hexdigest()))
    digest=hashlib.sha256()
    for row in entries:
        digest.update(_canonical_json(list(row))+b'\n')
    receipt={'total_archived':len(entries),'archive_digest':digest.hexdigest()}
    assert verify_receipt_prefix(archive.db,receipt)
    archive.close()
    reopened=EventJournal(tmp_path/'prefix.db','c'*64,max_bytes=0)
    statements=[]
    reopened.db.set_trace_callback(statements.append)
    assert verify_receipt_prefix(reopened.db,receipt)
    reopened.db.set_trace_callback(None)
    assert not any('ORDER BY origin,seq' in statement for statement in statements)
    assert reopened.append('alice',5,b'later signed bytes')
    assert verify_receipt_prefix(reopened.db,receipt) # append-only suffix is fine
    reopened.db.execute('DELETE FROM events WHERE seq=2')
    reopened.db.commit()
    with pytest.raises(PatchError,match='mutated'):
        verify_receipt_prefix(reopened.db,receipt)
    reopened.close()

@pytest.mark.asyncio
async def test_frozen_source_certified_cut_witness_skips_old_history(tmp_path,monkeypatch):
    from test_compact_journal_gc_v88 import setup_certified, close_all
    from patchlib.epoch_trainer_boot_v52 import verify_boot_carry
    from patchlib.core import PatchError
    from patchlib import epoch_live_rebase_v52 as rebase
    source,target,carry,boot,new_journal,emit=await setup_certified(tmp_path,with_carry=2)
    # setup_certified() already verified the cut, publishing a local witness.
    def fail_rehash(*_args,**_kwargs):
        raise AssertionError('O(history) frozen-source audit was repeated')
    with monkeypatch.context() as m:
        m.setattr(rebase,'verify_certified_cut',fail_rehash)
        assert verify_boot_carry(boot,carry,old_journal=source.journal)
    # If the frozen source unexpectedly receives another event, its last
    # rowid changes: cache MUST invalidate; boot MUST refuse to adopt it.
    emit()
    with pytest.raises(PatchError,match='changed after committed epoch handoff'):
        verify_boot_carry(boot,carry,old_journal=source.journal)
    close_all(source,target,carry,new_journal)

@pytest.mark.parametrize("simulated_mps", [False, True])
@pytest.mark.asyncio
async def test_fast_resume_carry_export_on_model_worker_keeps_sqlite_thread_affinity(tmp_path, monkeypatch, simulated_mps):
    """Regression: v97 called carry.sqlite from a model-task worker thread.

    EventJournal connections use SQLite's default check_same_thread=True.
    The caller captures both cursors on its original thread; the model task
    must do file/tensor I/O only, never reach back into either SQLite DB.
    """
    from patchlib import mps_runtime_v91 as runtime
    from patchlib.mps_runtime_v91 import model_task
    if simulated_mps:
        # Exercise the dedicated, thread-affine MPS executor on Linux too.
        monkeypatch.setattr(runtime, 'uses_mps', lambda **_kw: True)
        monkeypatch.setattr(runtime, 'synchronize_mps', lambda: None)

    old, target, emit, _maker = setup(tmp_path)
    archive = archive_for(tmp_path / 'carry.db', old.genesis, target.genesis)
    try:
        assert archive.replay(target) == 0
        assert archive.accept(target, emit())
        current_point = target.journal.fast_resume_cursor()
        carry_point = archive.journal.fast_resume_cursor()
        # This mirrors elasticnet.save_fast_resume_cache() on CPU; MPS uses
        # an equally foreign thread with a dedicated executor.
        saved = await model_task(
            save, target.journal, target, device='cpu', model=target.adapter.model,
            max_state_bytes=16 * 1024**2, shard_bytes=65536,
            min_free_bytes=0, cursor=current_point,
            carry_archive=archive, carry_cursor=carry_point,
        )
        assert saved['carry']['cursor'] == carry_point
        assert saved['cursor'] == current_point
        # The validation after worker completion belongs on the owner thread.
        assert target.journal.fast_resume_cursor() == current_point
        assert archive.journal.fast_resume_cursor() == carry_point
    finally:
        archive.close()
        target.journal.close()
        old.journal.close()
