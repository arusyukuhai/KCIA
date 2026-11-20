"""v141: per-RTC-generation egress accounting, consistent gossip eligibility,
short loss windows and strict advisory cache validation. No network required.
"""
from __future__ import annotations

import gc
import json
import weakref
from types import SimpleNamespace

import pytest

from patchlib.wan_bandit_v139 import (
    WanCostBandit, WireEgressTracker, eligible_live_push_slots,
    live_push_recipients,
)


class Node:
    def __init__(self, *, push=0, repair=0, verified=10.0,
                 ready=True, online=True, bridge=False):
        self.push_sent_wire_bytes=push
        self.repair_sent_wire_bytes=repair
        self.verified_at=verified
        self.ready=ready
        self.bridge_only=bridge
        self.connected=SimpleNamespace(is_set=lambda:online)


def slot(peer, **node_args):
    return SimpleNamespace(peer=peer, disabled=False, node=Node(**node_args))


def test_egress_tracks_new_rtc_generations_without_counter_rollover():
    tracker=WireEgressTracker()
    old=slot('peer',push=200_000_000,repair=10_000_000)
    slots={'peer':old}
    assert tracker.poll(slots)==210_000_000
    old.node.push_sent_wire_bytes+=50_000_000
    assert tracker.poll(slots)==50_000_000

    # ICE reconnection replaces the whole RTC node. The previous v140 global
    # sum approach would report 0 until the new peer exceeded 260 MB.
    oldref=weakref.ref(old.node)
    slots['peer']=slot('peer',push=3_000_000,repair=2_000_000)
    del old
    gc.collect()
    assert oldref() is None
    assert tracker.poll(slots)==5_000_000
    slots['peer'].node.repair_sent_wire_bytes+=6_000_000
    assert tracker.poll(slots)==6_000_000
    assert tracker.poll(slots)==0
    del slots['peer']
    gc.collect()
    assert len(tracker._last)==0  # not a memory leak during churn


def test_egress_rejects_invalid_counter_and_handles_one_counter_reset():
    tr=WireEgressTracker()
    s=slot('x',push=100,repair=50)
    assert tr.poll({'x':s})==150
    s.node.push_sent_wire_bytes=5  # in-place counter reset must not go negative
    s.node.repair_sent_wire_bytes=60
    assert tr.poll({'x':s})==15
    s.node.repair_sent_wire_bytes=-1
    with pytest.raises(ValueError,match='wire counters'):
        tr.poll({'x':s})


def test_body_recipients_match_bandit_cost_predicate():
    slots={
        'valid-a':slot('valid-a',verified=100),
        'valid-b':slot('valid-b',verified=100),
        'joining':slot('joining',ready=False),
        'stale':slot('stale',verified=15),
        'bridge':slot('bridge',bridge=True),
        'offline':slot('offline',online=False),
        'unsigned':slot('unsigned',verified=0),
    }
    slots['disabled']=slot('disabled')
    slots['disabled'].disabled=True
    eligible=eligible_live_push_slots(slots,now=110,peer_timeout=30)
    assert set(eligible)=={'valid-a','valid-b'}
    assert live_push_recipients(slots,fanout=8,now=110,peer_timeout=30)==2
    assert live_push_recipients(slots,fanout=1,now=110,peer_timeout=30)==1
    assert 'stale' not in eligible_live_push_slots(slots,now=120,peer_timeout=30)
    with pytest.raises(ValueError):eligible_live_push_slots(slots,now=float('inf'),peer_timeout=30)


def test_eight_minibatch_block_uses_disjoint_loss_windows():
    b=WanCostBandit(baseline=8,ceiling=32,wire_cost=0)
    for i in range(8):
        b.step(8.0-i,inbound_applied=0)
    assert b.observe(steps=8,signed_bytes=2048,inbound_applied=0)
    assert b.last_improvement>0  # v140 scored zero due to complete overlap


def test_malformed_float_arm_cache_is_not_loaded(tmp_path):
    path=tmp_path/'peer.wan-bandit-v139.json'
    b=WanCostBandit(baseline=32,ceiling=128,path=path)
    for i in range(32):b.step(5-i*.001,inbound_applied=0)
    assert b.observe(steps=32,signed_bytes=12_000,inbound_applied=0)
    data=json.loads(path.read_text())
    data['history'][0]['arm']=32.0  # JSON float is not a valid discrete action
    path.write_text(json.dumps(data))
    assert WanCostBandit(baseline=32,ceiling=128,path=path).samples==0
    data['history'][0]['arm']=32
    data['selected']=32.0
    path.write_text(json.dumps(data))
    assert WanCostBandit(baseline=32,ceiling=128,path=path).samples==0
