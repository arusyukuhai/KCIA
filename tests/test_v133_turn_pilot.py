"""WSS/ICE pilot route isolation regressions, no public Internet required."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from patchlib.p2p import PatchNode, _relay_only_sdp
from scripts import start_wan_pt


ROOM='test-room-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234'
SHA='a'*40


def test_relay_flag_propagates_to_elastic_patchnode():
    with patch.dict(os.environ, {'KCIA_TURN_TRANSPORT':'tcp', 'KCIA_FORCE_TURN_RELAY':'1'}):
        n=PatchNode('wss://signal.kcia.tech/ws',ROOM)
        assert n.turn_only and n.turn_transport=='tcp'
        assert 'typ host' not in _relay_only_sdp('a=candidate:1 1 udp 1 10.0.0.1 1000 typ host\r\na=candidate:2 1 udp 1 1.2.3.4 2000 typ relay\r\n')
    with patch.dict(os.environ, {'KCIA_FORCE_TURN_RELAY':'0'}):
        n=PatchNode('wss://signal.kcia.tech/ws',ROOM)
        assert not n.turn_only
    with patch.dict(os.environ, {'KCIA_FORCE_TURN_RELAY':'wat'}):
        with pytest.raises(ValueError,match='KCIA_FORCE_TURN_RELAY'):
            PatchNode('wss://signal.kcia.tech/ws',ROOM)


def test_launcher_injects_safe_pilot_defaults_and_correct_role(monkeypatch,capsys):
    captured={}
    def fake_call(command,*,cwd,env):
        captured.update(command=command,cwd=cwd,env=env)
        return 0
    monkeypatch.setattr(start_wan_pt.subprocess,'call',fake_call)
    # mac skips torch CUDA device check
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop('KCIA_TURN_TRANSPORT',None)
        os.environ.pop('KCIA_FORCE_TURN_RELAY',None)
        assert start_wan_pt.main(['run','--role','mac','--room-id',ROOM,'--revision',SHA])==0
    assert captured['env']['KCIA_TURN_TRANSPORT']=='auto'
    assert captured['env']['KCIA_FORCE_TURN_RELAY']=='0'
    assert captured['env']['KCIA_SNAPSHOT_PRIORITY']=='0'
    assert captured['command'][captured['command'].index('--shard-mb')+1]=='1'
    assert 'ICE route: auto relay_only=0' in capsys.readouterr().out


def test_colab_notebook_preserves_distinct_role():
    path=Path(start_wan_pt.ROOT/'notebooks/KCIA_Cloudflare_Mac_Colab.ipynb')
    body=json.dumps(json.loads(path.read_text()))
    assert "'--role', 'colab'" in body or "'--role','colab'" in body
    assert 'KCIA_FORCE_TURN_RELAY' in body
