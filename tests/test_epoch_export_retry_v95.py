"""Regressions for the single-peer 80%-full certified rollover stall."""
import hashlib
import sqlite3
import threading
from types import SimpleNamespace

import pytest
import torch

from patchlib.epoch_export_retry_v95 import EpochExportRetry
from patchlib.epoch_export_v57 import export_at_step_boundary
from patchlib.epoch_autonomous_v56 import AutonomousPagedEpoch
from patchlib.epoch_certificate_v46 import CompactEpochLedger
from patchlib.epoch_compact_trainer_v57 import CertifiedCompactTrainerLedger
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem


def test_hold_retry_without_any_more_journal_bytes_or_training_steps():
    retry = EpochExportRetry()
    delay = retry.failed(100., ValueError('temporary disk pressure'))
    assert delay == 5.0
    assert not retry.can_retry(104.99)
    # No additional events or log bytes are needed; V89 HOLD may be active.
    assert retry.can_retry(105.0)
    assert 'temporary disk pressure' in retry.status(103.)
    assert retry.failed(106., 'model snapshot transient failure') == 10.0
    assert not retry.can_retry(115.99)
    assert retry.can_retry(116.0)
    retry.started()
    retry.succeeded()
    assert retry.failures == 0 and not retry.can_retry(500.)


def test_retry_delay_is_bounded_and_does_not_mask_failure():
    retry = EpochExportRetry()
    for index in range(15):
        delay = retry.failed(index * 200., RuntimeError('export incomplete'))
    assert delay == 120.0
    assert retry.stage == 'retry-scheduled'
    assert 'export incomplete' in retry.last_error


def test_single_pinned_peer_exports_votes_certifies_without_network(tmp_path):
    private, public = tmp_path / 'one.key', tmp_path / 'one.pub'
    generate_keypair(private, public)
    signer, verifier = signer_from_pem(private), verifier_from_pem(public)
    root = '1' * 64
    journal = sqlite3.connect(tmp_path / 'journal.db')
    journal.execute('CREATE TABLE events(origin TEXT,seq INTEGER,digest TEXT, PRIMARY KEY(origin,seq))')
    journal.execute('INSERT INTO events VALUES(?,?,?)',
                    ('solo', 1, hashlib.sha256(b'durable event').hexdigest()))
    journal.commit()
    class Source:
        db = journal
        def count(self):
            return self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    model = torch.nn.Linear(4, 2)
    adapter = SimpleNamespace(model_id='tiny', revision='v1', groups=[],lock=threading.RLock())
    ledger = CompactEpochLedger(tmp_path / 'compact.sqlite3',room='solo-room',peer='solo',
                               committee=['solo'],signer=signer,trusted={'solo':verifier},genesis=root)
    holder = {'proposal': None}
    control = AutonomousPagedEpoch(ledger,tmp_path / 'stage.sqlite3',
        local_checkpoint=lambda p: (
            tmp_path/'incoming'/(p['manifest']+'.fmanifest'), tmp_path/'objects', journal),
        local_export_verified=lambda p: holder['proposal'] == p,
        defer_votes=True)
    try:
        p, pages, receipt = export_at_step_boundary(
            journal=Source(),model=model,adapter=adapter,store=tmp_path/'objects',
            incoming=tmp_path/'incoming',room='solo-room',epoch=1,parent=root,
            replicas=['solo'],signer=signer,peer='solo',
            shard_bytes=65536,max_state_bytes=1024**2)
        holder['proposal'] = p
        control.receive_export_receipt(receipt, p)
        control.offer(p, pages)
        assert ledger.certified_epoch(1) is None
        ready = control.ready_for_vote()
        assert ready is not None
        vote = ledger.vote(*(),p=p,pages=pages,journal_db=journal,
                           manifest_path=tmp_path/'incoming'/(p['manifest']+'.fmanifest'),
                           shard_store=tmp_path/'objects')
        # Same callback as the GUI loop; no network peer is required.
        control.receive('local',{'kind':'compact-vote','data':vote})
        assert ledger.certified_epoch(1)['proposal'] == p
        projected = CertifiedCompactTrainerLedger(ledger,control).certified_epoch(1)
        assert projected is not None
    finally:
        control.close(); ledger.close(); journal.close()


def test_compact_export_failures_are_retried_in_main_loop_source():
    # A minimal integration assertion catches removing the essential retry
    # branch during future refactors, without running the full training GUI.
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/'patchlib'/'elasticnet.py').read_text()
    assert 'compact_retry.can_retry(loop.time())' in source
    assert 'delay=compact_retry.failed(loop.time(),exc)' in source
    assert 'await maybe_propose_compact_epoch()' in source
