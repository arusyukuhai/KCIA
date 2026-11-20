"""v143: measured training work, bounded validated WAN directory metrics."""
import json
import math
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from patchlib.roomid import RoomDiscovery,list_public_rooms

ROOM='R'*32
PEER='p'+'a'*32


def test_report_is_monotonic_and_excludes_invalid_or_idle_time():
    r=RoomDiscovery('wss://signal.kcia.tech/ws',ROOM,PEER)
    first=r.training_report()
    assert re.fullmatch('[0-9a-f]{32}',first['run'])
    assert first['compute_ms']==0
    for bad in (float('nan'), float('inf'), -1.0, None, True, 4000):
        r.note_training(bad)
    assert r.training_report()['compute_ms']==0
    r.note_training(1.5)
    r.note_training(0.25)
    assert r.training_report()['compute_ms']==1750
    assert r.training_report()['run']==first['run']
    other=RoomDiscovery('wss://signal.kcia.tech/ws',ROOM,PEER)
    assert other.training_report()['run']!=first['run']
    roster={'type':'group-members','topology':'bounded-v2','degree':8,
            'revision':1,'online':1,'peers':[]}
    r.accept(roster)
    assert not r._usage_supported, 'v142 servers must never receive new metrics frames'
    r.accept({**roster,'usage_v1':True})
    assert r._usage_supported


class Reply:
    status=200
    def __init__(self,body):
        self.body=body
        self.headers={'Content-Length':str(len(body))}
    def read(self,n=-1):return self.body[:n]
    def __enter__(self):return self
    def __exit__(self,*exc):pass


def lookup(monkeypatch,entry):
    body=json.dumps({'rooms':[entry]}).encode()
    monkeypatch.setattr('urllib.request.urlopen',lambda *args,**kw:Reply(body))
    return list_public_rooms('wss://signal.kcia.tech/ws')


def test_metrics_are_validated_and_legacy_server_remains_supported(monkeypatch):
    e=dict(room_id=ROOM,model='company/model',online=4096,datasets=[],revision='main')
    assert lookup(monkeypatch,e)==[e]
    e['cumulative_participants']=12000
    e['training_seconds']=35281.125
    assert lookup(monkeypatch,e)==[e]
    for bad in (-1,'100',None,True,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='directory metric'):
            lookup(monkeypatch,{**e,'training_seconds':bad})
    for bad in ('4',False,3.5, -1, 1_000_000_000_000_001):
        with pytest.raises(ValueError,match='directory metric'):
            lookup(monkeypatch,{**e,'cumulative_participants':bad})
    with pytest.raises(ValueError,match='invalid room listing data'):
        lookup(monkeypatch,{**e,'online':4097})


def test_nim_usage_is_persistent_bounded_not_in_immutable_manifest():
    src=Path(__file__).resolve().parents[1].joinpath('src/main.nim').read_text()
    for feature in ('proc groupTrainingReport','of "training-report": groupTrainingReport(ws, doc)',
                    'let key = c.peerId & ":" & run',
                    'if value <= prev.ms: return', 'maybePersistUsage(room)',
                    'proc loadRoomUsage()', 'cumulative_participants',
                    'training_seconds', 'metrics_source',
                    'if c.trainingRun.len > 0 and c.trainingRun != run: return'):
        assert feature in src
    assert 'let folder = roomDataDir / "_usage"' in src
    assert 'if now - roomUsageLastSaved.getOrDefault(room,0\'i64) >= 60:' in src
    assert 'var roomUsageLastSaved' in src
