"""v94: Old-genesis carries must not shadow current-genesis signed IDs.

Use real Ed25519 signatures, real SQLite/sidecar journals and tiny LoHA model.
No external network or GPU required.
"""
import asyncio
from copy import deepcopy

import torch

from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem
from patchlib.epoch_carry_v46 import CrossEpochCarryArchive
from patchlib.elastic import unpack_event
from tests.test_epoch_carry_v46 import Tiny, make_engine


OLD = 'a' * 64
NEW = 'b' * 64
MANIFEST = 'c' * 64


def _scenario(tmp_path):
    torch.manual_seed(100)
    base = Tiny()
    previous_model = deepcopy(base)
    current_model = deepcopy(base)
    private = tmp_path / 'self.pem'
    public = tmp_path / 'self.pub'
    generate_keypair(private, public)
    trusted = {'alice': verifier_from_pem(public)}
    signer = signer_from_pem(private)
    old, old_adapter = make_engine(previous_model, tmp_path / 'old.db', OLD, 'alice', trusted, signer)
    opt = torch.optim.AdamW(old_adapter.trainable_parameters(), lr=.005)
    previous_model().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    old_wire = old.seal_local(1)
    new, _ = make_engine(current_model, tmp_path / 'new.db', NEW, 'alice', trusted, signer)
    archive = CrossEpochCarryArchive(tmp_path / 'carry.db', source_genesis=OLD,
        target_genesis=NEW, certified_manifest=MANIFEST, epoch=1, cutoffs={})
    return old, old_wire, new, archive


def _close(old, new, archive):
    if archive is not None:
        archive.close()
    new.journal.close()
    old.journal.close()


def test_replayed_self_carry_does_not_shadow_first_new_epoch_event(tmp_path):
    async def run():
        old, old_wire, new, archive = _scenario(tmp_path)
        try:
            assert archive.journal.append('alice', 1, old_wire)
            factors = [[p.detach().clone() for p in g.overlay.local.parts]
                       for g in new.adapter.groups]
            anchors = [[getattr(g.overlay.anchor, f'part{j}').detach().clone()
                        for j in range(len(g.overlay.local.parts))]
                       for g in new.adapter.groups]
            assert archive.replay(new) == 1
            assert new.journal.count() == 0
            assert new.journal.next_seq('alice') == 1
            # Carry updates the GLOBAL effective weight, not the new-generation
            # local LoHA optimizer coordinates/anchors.
            for g, initial, initial_anchor in zip(new.adapter.groups, factors, anchors):
                assert all(torch.equal(p, before) for p, before in zip(g.overlay.local.parts, initial))
                assert all(torch.equal(getattr(g.overlay.anchor, f'part{j}'), before)
                           for j, before in enumerate(initial_anchor))
            assert ('alice', 1) not in new.applied
            wire = await new.seal_local_responsive(1)
            assert unpack_event(wire, trusted={}, unsafe=True)[1]['seq'] == 1
            assert new.journal.count() == 1
            assert new.journal.has('alice', 1)[0] != archive.journal.has('alice', 1)[0]
            assert ('alice', 1) in new.applied
        finally:
            _close(old, new, archive)
    asyncio.run(run())


def test_live_carry_after_current_event_with_same_identity_is_applied_once(tmp_path):
    old, old_wire, new, archive = _scenario(tmp_path)
    try:
        # Current-generation (alice, 1) already exists; carry (alice, 1)
        # belongs to OLD epoch and must be independently accepted and applied.
        assert archive.replay(new) == 0
        current_wire = new.seal_local(1)
        before = new.adapter.model.layer.parametrizations.weight.original.detach().clone()
        assert archive.accept(new, old_wire) is True
        after = new.adapter.model.layer.parametrizations.weight.original.detach().clone()
        assert not torch.equal(before, after)
        assert archive.accept(new, old_wire) is False
        assert torch.equal(after, new.adapter.model.layer.parametrizations.weight.original)
        assert new.journal.count() == 1 and archive.journal.count() == 1
        assert new.journal.get('alice', 1) == current_wire
        assert archive.journal.get('alice', 1) == old_wire
    finally:
        _close(old, new, archive)


def test_recovery_carry_then_current_same_sequence_restores_model(tmp_path):
    old, old_wire, new, archive = _scenario(tmp_path)
    try:
        assert archive.replay(new) == 0
        assert archive.accept(new, old_wire)
        current_wire = new.seal_local(1)
        expected = new.adapter.model.layer.parametrizations.weight.original.detach().clone()
        # Reconstruct exact same seeded base, then BOTH distinct source epochs.
        torch.manual_seed(100)
        fresh = Tiny()
        fresh_engine, _ = make_engine(fresh, tmp_path / 'recovery.db', NEW, 'alice', new.trusted)
        # A retired trainer is closed before a fresh process reopens the carry DB.
        archive.close()
        archive = None
        recovery_archive = CrossEpochCarryArchive(tmp_path / 'carry.db', source_genesis=OLD,
            target_genesis=NEW, certified_manifest=MANIFEST, epoch=1, cutoffs={})
        try:
            assert recovery_archive.replay(fresh_engine) == 1
            fresh_engine.journal.append('alice', 1, current_wire)
            fresh_engine.replay()
            actual = fresh_engine.adapter.model.layer.parametrizations.weight.original.detach()
            assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
            assert fresh_engine.journal.has('alice', 1) is not None
        finally:
            recovery_archive.close(); fresh_engine.journal.close()
    finally:
        _close(old, new, archive)


def test_async_inbound_accepts_self_origin_old_epoch_carry_with_same_current_seq(tmp_path):
    from patchlib.async_ingest import AsyncInboundPipeline

    async def run():
        old, old_wire, new, archive = _scenario(tmp_path)
        received = []
        async def ack(info, wire, sender, fresh):
            received.append((info.genesis, info.seq, fresh))
        assert archive.replay(new) == 0
        pipeline = AsyncInboundPipeline(new, ack, carry_archive=archive)
        try:
            now_wire = await new.seal_local_responsive(1)
            pipeline.start()
            assert pipeline.submit('neighbor', old_wire)
            await asyncio.wait_for(pipeline.drain(), timeout=10)
            assert pipeline.submit('neighbor', old_wire)
            await asyncio.wait_for(pipeline.drain(), timeout=10)
            assert received == [(OLD, 1, True), (OLD, 1, False)]
            assert archive.journal.get('alice', 1) == old_wire
            assert new.journal.get('alice', 1) == now_wire
            assert new.journal.count() == archive.journal.count() == 1
        finally:
            if pipeline.task:
                await pipeline.stop()
            _close(old, new, archive)
    asyncio.run(run())


def test_old_reinit_boundary_does_not_reset_new_generation_local_factors(tmp_path):
    old, first_wire, new, archive = _scenario(tmp_path)
    try:
        # A signed old-genesis seq=20 has reinit_after=True, but that cannot
        # reset freshly initialized current-generation optimizer coordinates.
        last_wire = first_wire
        for _ in range(19):
            last_wire = old.seal_local(1)
        assert unpack_event(last_wire, {}, unsafe=True)[1]['seq'] == 20
        assert unpack_event(last_wire, {}, unsafe=True)[1]['reinit_after'] is True
        before = [[p.detach().clone() for p in group.overlay.local.parts]
                  for group in new.adapter.groups]
        assert archive.replay(new) == 0
        assert archive.accept(new, last_wire)
        assert ('alice', 20) not in new.applied
        for group, initial in zip(new.adapter.groups, before):
            assert all(torch.equal(p, oldp) for p, oldp in zip(group.overlay.local.parts, initial))
    finally:
        _close(old, new, archive)


def test_exact_prior_error_missing_journal_digest_is_not_generated_by_replayed_carry(tmp_path):
    """v93 fails with journal_sha256=missing, the observed user's error."""
    async def run():
        old, old_wire, new, archive = _scenario(tmp_path)
        try:
            assert archive.journal.append('alice', 1, old_wire)
            assert archive.replay(new) == 1
            # Current-generation journal has *no* event 1; old carry did.
            assert new.journal.has('alice', 1) is None
            emitted = await new.seal_local_responsive(1)
            assert unpack_event(emitted, {}, unsafe=True)[1]['seq'] == 1
            assert new.journal.has('alice', 1) is not None
        finally:
            _close(old, new, archive)
    asyncio.run(run())
