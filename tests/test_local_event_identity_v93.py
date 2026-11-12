"""v93 regression: protect self-origin signed event IDs from asynchronous collisions.

Only CPU and throwaway sqlite journals are used. No real network, HF or MPS needed.
"""
import asyncio

import pytest

from patchlib.core import PatchError
from patchlib.elastic import unpack_event
from patchlib.async_ingest import AsyncInboundPipeline
from test_continual_elastic import make_engine


def test_concurrent_local_responsive_seals_take_distinct_sequences(tmp_path):
    async def scenario():
        engine = make_engine(tmp_path, 'alice')
        try:
            # Both coroutines start before either worker can complete export.
            one, two = await asyncio.gather(
                engine.seal_local_responsive(1),
                engine.seal_local_responsive(1),
            )
            first = unpack_event(one, {}, unsafe=True)[1]['seq']
            second = unpack_event(two, {}, unsafe=True)[1]['seq']
            assert (first, second) == (1, 2)
            assert engine.journal.count() == 2
            assert engine.journal.next_seq(engine.self_id) == 3
        finally:
            engine.journal.close()
    asyncio.run(scenario())


def test_remote_cannot_install_new_event_under_our_own_identity(tmp_path):
    async def scenario():
        target = make_engine(tmp_path / 'legit', 'alice')
        clone = make_engine(tmp_path / 'clone', 'alice')
        async def ack(*_args):
            pass
        pipeline = AsyncInboundPipeline(target, ack)
        try:
            leaked = clone.seal_local(1)
            pipeline.start()
            assert pipeline.submit('relay-bob', leaked)
            with pytest.raises(PatchError, match='duplicate peer identity'):
                await asyncio.wait_for(pipeline.task, timeout=8)
            assert target.journal.count() == 0
            assert target.journal.next_seq('alice') == 1
            # A rejected relay must not poison local seq allocation.
            saved = await target.seal_local_responsive(1)
            assert unpack_event(saved, {}, unsafe=True)[1]['seq'] == 1
        finally:
            if pipeline.task and not pipeline.task.done():
                pipeline.task.cancel()
                await asyncio.gather(pipeline.task, return_exceptions=True)
            clone.journal.close()
            target.journal.close()
    asyncio.run(scenario())


def test_aggregate_cannot_install_new_local_origin(tmp_path):
    async def scenario():
        target = make_engine(tmp_path / 'target', 'alice')
        clone = make_engine(tmp_path / 'clone', 'alice')
        bob = make_engine(tmp_path / 'bob', 'bob')
        async def ack(*_args):
            pass
        pipeline = AsyncInboundPipeline(target, ack)
        try:
            own_from_other = clone.seal_local(1)
            remote = bob.seal_local(1)
            with pytest.raises(PatchError, match='duplicate peer identity'):
                await pipeline._ingest_aggregate('relay-bob', [own_from_other, remote])
            assert target.journal.count() == 0
        finally:
            bob.journal.close()
            clone.journal.close()
            target.journal.close()
    asyncio.run(scenario())


def test_echo_of_existing_self_event_remains_idempotent(tmp_path):
    async def scenario():
        engine = make_engine(tmp_path, 'alice')
        acked = []
        async def ack(info, wire, sender, fresh):
            acked.append((info.seq, sender, fresh))
        pipeline = AsyncInboundPipeline(engine, ack)
        try:
            known = await engine.seal_local_responsive(1)
            pipeline.start()
            assert pipeline.submit('relay-bob', known)
            await asyncio.wait_for(pipeline.queue.join(), timeout=8)
            pipeline.check()
            assert acked == [(1, 'relay-bob', False)]
            assert engine.journal.count() == 1
        finally:
            if pipeline.task:
                pipeline.task.cancel()
                await asyncio.gather(pipeline.task, return_exceptions=True)
            engine.journal.close()
    asyncio.run(scenario())


def test_base_elastic_concurrent_seals_use_distinct_sequences(tmp_path):
    from test_elastic import make_pair
    async def scenario():
        engines, _, _ = make_pair(tmp_path)
        try:
            alice = engines['alice']
            left, right = await asyncio.gather(
                alice.seal_local_responsive(1),
                alice.seal_local_responsive(1),
            )
            assert [unpack_event(w, alice.trusted)[1]['seq'] for w in (left,right)] == [1,2]
            assert alice.journal.count() == 2
        finally:
            for engine in engines.values():
                engine.journal.close()
    asyncio.run(scenario())


def test_existing_local_identity_collision_stops_without_rewriting_journal(tmp_path):
    async def scenario():
        engine = make_engine(tmp_path, 'alice')
        try:
            engine.applied.add(('alice', 1))  # simulate model index ahead of disk
            with pytest.raises(PatchError, match='conflicting event ID: origin=alice seq=1'):
                await engine.seal_local_responsive(1)
            assert engine.journal.count() == 0
        finally:
            engine.journal.close()
    asyncio.run(scenario())
