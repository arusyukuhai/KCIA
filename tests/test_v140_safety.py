"""v140: actual zero-recipient WAN accounting and preflight GUI safety."""
from pathlib import Path
import pytest

from patchlib.wan_bandit_v139 import WanCostBandit
from patchlib.webview_controller import WebviewController
from patchlib.roomid import create_room_id


def test_offline_updates_cannot_invent_traffic_or_train_policy(tmp_path):
    path=tmp_path/'peer.sqlite3.wan-bandit-v139.json'
    b=WanCostBandit(baseline=32,ceiling=128,wire_cost=.5,path=path)
    for event in range(15):
        for i in range(32):
            b.step(5-i*.02,inbound_applied=0)
        assert not b.observe(steps=32,signed_bytes=160*1024**2,
                             measured_egress_bytes=160*1024**2,
                             recipients=0,inbound_applied=0)
    assert b.events==0
    assert b.samples==0
    assert b.wire_estimate_mib==0
    assert b.selected==32
    assert not path.exists()
    # Once a real authenticated recipient is present, actual cost is scored.
    for i in range(32):b.step(5-i*.02,inbound_applied=0)
    assert b.observe(steps=32,signed_bytes=160*1024**2,
                     recipients=1,inbound_applied=0)
    assert b.samples==1
    assert b.wire_estimate_mib==160
    assert path.is_file()


def test_invalid_connection_type_not_counted_as_recipient():
    b=WanCostBandit(baseline=32,ceiling=128)
    for recipients in (False, True, -1, 9, 1.5, '1'):
        with pytest.raises(ValueError):
            b.observe(steps=32,signed_bytes=1,inbound_applied=0,
                      recipients=recipients)


def test_double_start_never_allocates_profile_or_writes_sidecars(tmp_path,monkeypatch):
    controller=WebviewController(tmp_path)
    room=create_room_id()
    controller.save({'room':room,'model':'sshleifer/tiny-gpt2','revision':'main',
                     'task':'causal-lm','initialization':'pretrained',
                     'sources':[{'repo':'org/example','revision':'a'*40,
                         'config':None,'split':'train',
                         'columns':{'text':'text'},'weight':1.0}]})
    saved=controller._room_path(room).read_bytes()
    monkeypatch.setattr(controller.train,'status',lambda: True)
    def forbidden_profile(_):
        raise AssertionError('An already active trainer must be rejected before profile allocation')
    monkeypatch.setattr('patchlib.webview_lifecycle_v110.allocate_profile',forbidden_profile,
                        raising=False)
    from patchlib import webview_lifecycle_v110 as lifecycle
    with pytest.raises(RuntimeError,match='already online'):
        lifecycle.start_training(controller,{'room':room,'ws':'wss://signal.kcia.tech/ws',
                             'mixed_training':{'enabled':True}},
                             profile_allocator=forbidden_profile)
    assert controller._room_path(room).read_bytes()==saved
    assert not list(controller._room_path(room).parent.glob('*.launch.json'))
    assert not list(controller._room_path(room).parent.glob('*mixed*'))
    assert not controller.profiles


def test_fanout_cost_counts_only_current_authenticated_recipients():
    from types import SimpleNamespace
    from patchlib.wan_bandit_v139 import live_push_recipients
    class Event:
        def __init__(self, is_open):self.is_open=is_open
        def is_set(self):return self.is_open
    def slot(*, verified=100., ready=True, connected=True,
             disabled=False, bridge=False):
        return SimpleNamespace(disabled=disabled,node=SimpleNamespace(
            verified_at=verified,ready=ready,bridge_only=bridge,
            connected=Event(connected)))
    slots={str(i):slot() for i in range(8)}
    assert live_push_recipients(slots,fanout=2,now=110.,peer_timeout=30.)==2
    assert live_push_recipients(slots,fanout=1,now=110.,peer_timeout=30.)==1
    assert live_push_recipients(slots,fanout=8,now=110.,peer_timeout=30.)==8
    slots={'disconnected':slot(connected=False),'unauth':slot(verified=0.),
           'stale':slot(verified=20.),'bridge':slot(bridge=True),
           'notready':slot(ready=False),'disabled':slot(disabled=True)}
    assert live_push_recipients(slots,fanout=2,now=110.,peer_timeout=30.)==0
    slots['valid']=slot()
    assert live_push_recipients(slots,fanout=2,now=110.,peer_timeout=30.)==1
    for bad in (0,9,True):
        with pytest.raises(ValueError):
            live_push_recipients(slots,fanout=bad,now=110.,peer_timeout=30.)


def test_policy_cache_cleanup_failure_cannot_stop_training(tmp_path,monkeypatch):
    from pathlib import Path
    from unittest.mock import patch
    p=tmp_path/'peer.wan-bandit-v139.json'
    b=WanCostBandit(baseline=32,path=p)
    original=Path.unlink
    def cannot_cleanup(self,*args,**kwargs):
        if '.tmp-' in self.name:
            raise PermissionError('unwritable cache directory')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',cannot_cleanup)
    # Cache is best effort. Even cleanup failure cannot abort an optimizer.
    b.persist()
    for i in range(32):b.step(4.0-i*.001,inbound_applied=0)
    assert b.observe(steps=32,signed_bytes=1024,
                     recipients=1,inbound_applied=0)
