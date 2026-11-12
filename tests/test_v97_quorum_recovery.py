"""Recovery tests for a single-peer candidate stuck after durable vote commit."""
import hashlib
import sqlite3
import threading
from types import SimpleNamespace

import pytest
import torch

from patchlib.core import PatchError
from patchlib.epoch_autonomous_v56 import AutonomousPagedEpoch
from patchlib.epoch_certificate_v46 import CompactEpochLedger
from patchlib.epoch_export_v57 import export_at_step_boundary
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem


def make_solo(tmp_path):
    private, public = tmp_path / 'local.pem', tmp_path / 'local.pub'
    generate_keypair(private, public)
    signer, verifier = signer_from_pem(private), verifier_from_pem(public)
    db = sqlite3.connect(tmp_path/'events.sqlite3')
    db.execute('CREATE TABLE events(origin TEXT, seq INTEGER, digest TEXT, PRIMARY KEY(origin,seq))')
    db.execute('INSERT INTO events VALUES (?,?,?)', ('local',1,hashlib.sha256(b'first').hexdigest()))
    db.commit()
    class Journal:
        def __init__(self, db): self.db=db
        def count(self): return self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    model=torch.nn.Linear(4,2)
    adapter=SimpleNamespace(model_id='tiny',revision='v1',groups=[],lock=threading.RLock())
    p,pages,receipt=export_at_step_boundary(
        journal=Journal(db), model=model, adapter=adapter,
        store=tmp_path/'store',incoming=tmp_path/'incoming',
        room='solo-room',epoch=1,parent='a'*64,
        replicas=['local'],signer=signer,peer='local',
        max_state_bytes=1024**2,shard_bytes=65536)
    ledger_path=tmp_path/'ledger.sqlite3';stage_path=tmp_path/'stage.sqlite3'
    def open_pair():
        ledger=CompactEpochLedger(ledger_path,room='solo-room',peer='local',
            committee=['local'],signer=signer,trusted={'local':verifier},genesis='a'*64)
        controller=AutonomousPagedEpoch(ledger,stage_path,
            local_checkpoint=lambda _p:(tmp_path/'incoming'/(p['manifest']+'.fmanifest'),tmp_path/'store',db),
            local_export_verified=lambda _p: True,defer_votes=True)
        return ledger,controller
    return db,p,pages,receipt,open_pair


def test_reconcile_crash_after_stage_vote_before_certificate(tmp_path):
    db,p,pages,receipt,open_pair=make_solo(tmp_path)
    ledger,control=open_pair()
    try:
        control.receive_export_receipt(receipt,p)
        control.offer(p,pages)
        assert control.vote_progress(1)['stage']=='eligible-for-local-vote'
        vote=ledger.vote(p,pages=pages,journal_db=db,
            manifest_path=tmp_path/'incoming'/(p['manifest']+'.fmanifest'),
            shard_store=tmp_path/'store')
        original_accept=ledger.accept
        def crash(_cert):
            raise OSError('simulated crash just after durable stage vote')
        ledger.accept=crash
        with pytest.raises(OSError):
            control.receive('local',{'kind':'compact-vote','data':vote})
        assert control.vote_progress(1)['stage']=='quorum-votes-persisted-certificate-recovery'
        ledger.accept=original_accept
    finally:
        control.close();ledger.close()
    # Reopen using the exact same signed bytes, without generating another vote.
    ledger,control=open_pair()
    try:
        assert ledger.certified_epoch(1) is None
        assert control.reconcile_quorum() is True
        assert ledger.certified_epoch(1)['proposal']==p
        assert control.reconcile_quorum() is False
        assert control.vote_progress(1)['stage']=='certified-awaiting-activation'
    finally:
        control.close();ledger.close();db.close()


def test_incomplete_quorum_is_never_synthesized(tmp_path):
    db,p,pages,receipt,open_pair=make_solo(tmp_path)
    ledger,control=open_pair()
    try:
        control.receive_export_receipt(receipt,p)
        control.offer(p,pages)
        assert control.reconcile_quorum() is False
        assert ledger.certified_epoch(1) is None
        assert control.vote_progress(1)['votes']==0
    finally:
        control.close();ledger.close();db.close()


def test_logged_pressure_stage_contains_real_vote_progress():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'patchlib'/'elasticnet.py').read_text()
    assert "generation_compact_control.reconcile_quorum()" in source
    assert "generation_compact_control.vote_progress(" in source
    assert "local-vote-worker-running=" in source
