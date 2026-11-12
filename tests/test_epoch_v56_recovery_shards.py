import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from patchlib.core import PatchError
from patchlib.online_generation import _digest
from patchlib.epoch_multihop_v56 import CertifiedChain,MultiEpochCatchup,DurableOfflineOutbox


class Ledger:
    genesis='0'*64
    def __init__(self,count=4):
        self.commits={}
        parent=self.genesis
        for epoch in range(1,count+1):
            p={'epoch':epoch,'parent':parent,'manifest':f'{epoch:064x}'}
            self.commits[epoch]={'proposal':p}
            parent=_digest(p)
    def certified_epoch(self,e):return self.commits.get(e)
    def latest(self):return self.commits[max(self.commits)] if self.commits else None


class Router:
    def __init__(self):self.current=SimpleNamespace(epoch=0)


class Service:
    def __init__(self,ledger):
        self.ledger=ledger;self.router=Router();self.fail_at=None;self.calls=[]
    async def activate_next(self):
        n=self.router.current.epoch+1
        self.calls.append(n)
        if self.fail_at==n:raise ConnectionError('missing checkpoint shard')
        self.router.current.epoch=n
        return True


@pytest.mark.asyncio
async def test_three_epoch_offline_recovery_adjacent_only():
    ledger=Ledger(4);service=Service(ledger)
    recovery=MultiEpochCatchup(service,max_hops_per_poll=2)
    assert await recovery.poll()==2
    assert service.router.current.epoch==2
    service.fail_at=3
    with pytest.raises(ConnectionError):await recovery.poll()
    assert service.router.current.epoch==2
    service.fail_at=None
    assert await recovery.poll()==2
    assert service.calls==[1,2,3,3,4]


@pytest.mark.asyncio
async def test_missing_or_broken_certificate_blocks_recovery():
    ledger=Ledger(3);service=Service(ledger)
    ledger.commits[2]['proposal']['parent']='f'*64
    with pytest.raises(PatchError,match='broken parent'):
        await MultiEpochCatchup(service).poll()
    assert not service.calls
    ledger=Ledger(3);service=Service(ledger)
    del ledger.commits[2]
    with pytest.raises(PatchError,match='missing authenticated'):
        await MultiEpochCatchup(service).poll()
    assert not service.calls


def test_quarantine_never_drops_unacknowledged_old_wire(tmp_path):
    q=DurableOfflineOutbox(tmp_path,max_bytes=1024)
    assert q.hold(epoch=0,origin='a',seq=1,wire=b'a'*600)
    assert not q.hold(epoch=0,origin='a',seq=1,wire=b'a'*600)
    with pytest.raises(PatchError,match='equivocating'):
        q.hold(epoch=0,origin='a',seq=1,wire=b'wrong')
    with pytest.raises(PatchError,match='capacity'):
        q.hold(epoch=0,origin='a',seq=2,wire=b'b'*600)
    with pytest.raises(PatchError):q.acknowledge(epoch=0,origin='a',seq=1,verified_archive=True)
    assert list(q.iter_epoch(0))==[('a',1,b'a'*600)]
    q.close()
    q=DurableOfflineOutbox(tmp_path,max_bytes=1024)
    assert list(q.iter_epoch(0))==[('a',1,b'a'*600)]
    q.close()






