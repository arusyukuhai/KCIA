"""v132 regression: paced WAN streams, preemptible huge patches and sharded reconnects."""
import asyncio
import hashlib
import json

import pytest

from patchlib.p2p import BulkPacer, bulk_rate, transmit
from patchlib.snapshot_live import LiveSnapshotLink


class RecordingChannel:
    def __init__(self):
        self.readyState = 'open'
        self.bufferedAmount = 0
        self.sent = []

    def send(self, data):
        self.sent.append(data)


def test_v132_pacer_rejects_invalid_env(monkeypatch):
    monkeypatch.setenv('KCIA_SNAPSHOT_RATE_MIB', '0')
    with pytest.raises(ValueError, match='KCIA_SNAPSHOT_RATE_MIB'):
        bulk_rate('KCIA_SNAPSHOT_RATE_MIB')
    monkeypatch.setenv('KCIA_SNAPSHOT_RATE_MIB', '1.5')
    assert bulk_rate('KCIA_SNAPSHOT_RATE_MIB').bytes_per_second == 1.5*1024**2


def test_v132_patch_pauses_between_fragments_without_corrupting_frames():
    async def scenario():
        channel=RecordingChannel()
        frozen=[False]
        data=b'X' * (16*1024*8 + 123)
        def capture(buf):
            channel.sent.append(buf)
            if isinstance(buf,bytes) and sum(isinstance(b,bytes) for b in channel.sent)==2:
                frozen[0]=True
        channel.send=capture
        sender=asyncio.create_task(transmit(channel,data,pacer=BulkPacer(32),
                           pause_when=lambda:frozen[0]))
        for _ in range(500):
            if frozen[0]:break
            await asyncio.sleep(.002)
        assert frozen[0]
        sent_before=len(channel.sent)
        await asyncio.sleep(.05)
        assert len(channel.sent)==sent_before
        frozen[0]=False
        await asyncio.wait_for(sender,5)
        chunks=[c for c in channel.sent if isinstance(c,bytes)]
        assert b''.join(chunks)==data
        assert json.loads(channel.sent[0])['type']=='start'
        assert json.loads(channel.sent[-1])['type']=='end'
    asyncio.run(scenario())


def test_v132_pacer_cooperatively_yields():
    async def scenario():
        limiter=BulkPacer(1)
        ran=[]
        async def heartbeat():
            for _ in range(5):
                ran.append('tick')
                await asyncio.sleep(.01)
        async def transfer():
            for _ in range(5):
                await limiter.pace(32*1024)
                ran.append('packet')
        await asyncio.gather(heartbeat(),transfer())
        assert ran.index('tick')<ran.index('packet')
        assert ran.count('tick')==5
        assert ran.count('packet')==5
    asyncio.run(scenario())


class LinkedChannel:
    def __init__(self):
        self.readyState='open'
        self.bufferedAmount=0
        self.handlers={}
        self.other=None
    def on(self,event):
        def register(func):
            self.handlers[event]=func
            return func
        return register
    def send(self,frame):
        if self.readyState!='open': raise ConnectionError('closed')
        # Simulate ordered SCTP forwarding, while keeping callbacks asynchronous.
        asyncio.get_running_loop().call_soon(self.other.handlers['message'],frame)
    def close(self):
        if self.readyState=='closed':return
        self.readyState='closed'
        self.handlers['close']()


def test_v132_verified_small_shards_survive_disconnect(tmp_path,monkeypatch):
    from torch import nn
    from patchlib.core import UniversalAdapter
    from patchlib.shards import export_state_sharded,read_manifest,shard_map
    from patchlib.roomid import auto_identity

    monkeypatch.setenv('KCIA_SNAPSHOT_RATE_MIB','8')

    async def scenario():
        peer,signer,_,verify=auto_identity(tmp_path/'identity.pem')
        model=nn.Linear(1024,1024)
        adapter=UniversalAdapter(model,'tiny','rev',kind='loha',rank=2)
        store=tmp_path/'src-objects'
        manifest=tmp_path/'snapshot.fmanifest'
        export_state_sharded(model,store,manifest,model_id='tiny',revision='rev',
            peer_id=peer,adapter=adapter,shard_bytes=1024**2)
        assert len(shard_map(read_manifest(manifest)))>=4
        source=LiveSnapshotLink(store,tmp_path/'sender-cache',16*1024**2,lambda:verify)
        target=LiveSnapshotLink(tmp_path/'target-objects',tmp_path/'receiver-cache',16*1024**2,lambda:verify)
        try:
            for attempt in (1,2):
                a,b=LinkedChannel(),LinkedChannel()
                a.other=b
                b.other=a
                source.attach(a)
                target.attach(b)
                source.on_authenticated(signer)
                a.handlers['open']()
                b.handlers['open']()
                assert source.bulk_transfer_active is False
                digest=source.offer(manifest,signer)
                assert source.bulk_transfer_active
                if attempt==1:
                    for _ in range(1000):
                        if target.session.received_objects>=1: break
                        await asyncio.sleep(.01)
                    assert target.session.received_objects>=1
                    a.close(); b.close()
                    await asyncio.sleep(.04)
                else:
                    finished=await asyncio.wait_for(target.completed.get(),20)
                    assert hashlib.sha256(finished.read_bytes()).hexdigest()==digest
                    assert target.session.reused_objects>=1
                    assert target.session.remote_complete
                    a.close(); b.close()
                    await asyncio.sleep(.04)
        finally:
            if source.worker: source.worker.cancel()
            if target.worker: target.worker.cancel()
    asyncio.run(scenario())
