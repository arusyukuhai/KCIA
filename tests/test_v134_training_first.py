"""WAN pilot cadence and training-first semantics; no live cloud service needed."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import start_wan_pt
from patchlib.adaptive import DeviceProfile, initial_policy, AdaptiveController

ROOM='abcdeFGHIJKLMNOPQRSTU123456'
SHA='1'*40


def test_new_defaults_give_eightfold_fewer_event_publications():
    p=argparse.Namespace(role='colab',room_id=ROOM,revision=SHA,ws=start_wan_pt.WS,
                         device=None,rounds=0,local_steps=32,snapshot_shard_mb=1)
    cmd=start_wan_pt.make_command(p)
    assert cmd[cmd.index('--local-steps')+1]=='32'
    assert cmd[cmd.index('--reinit-every')+1]=='20'
    assert '--signing-key' not in cmd
    assert 640//32 == 20
    assert 640//4 == 160
    assert (640//4) // (640//32) == 8


def test_custom_sync_64_and_reinit_unchanged():
    p=argparse.Namespace(role='mac',room_id=ROOM,revision=SHA,ws=start_wan_pt.WS,
                         device=None,rounds=0,local_steps=64,snapshot_shard_mb=1)
    argv=start_wan_pt.make_command(p)
    assert argv[argv.index('--local-steps')+1]=='64'
    assert argv[argv.index('--reinit-every')+1]=='20'


def test_launcher_training_snapshot_mode_can_be_restored(monkeypatch,tmp_path,capsys):
    called=[]
    monkeypatch.setattr(start_wan_pt,'ROOT',tmp_path)
    monkeypatch.setattr(start_wan_pt.subprocess,'call',lambda argv,*,cwd,env:called.append((argv,env)) or 0)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop('KCIA_TURN_TRANSPORT',None)
        os.environ.pop('KCIA_FORCE_TURN_RELAY',None)
        assert start_wan_pt.main(['run','--role','mac','--room-id',ROOM,'--revision',SHA])==0
    assert len(called)==1
    argv,env=called[0]
    assert argv[argv.index('--local-steps')+1]=='32'
    assert env['KCIA_TURN_TRANSPORT']=='auto' and env['KCIA_FORCE_TURN_RELAY']=='0'
    assert env['KCIA_SNAPSHOT_PRIORITY']=='0'
    assert 'snapshot_priority=training' in capsys.readouterr().out
    called.clear()
    assert start_wan_pt.main(['run','--role','mac','--room-id',ROOM,'--revision',SHA,
                               '--local-steps','64','--snapshot-priority','snapshot'])==0
    assert called[0][1]['KCIA_SNAPSHOT_PRIORITY']=='1'


@pytest.mark.parametrize('bad',['3','5','65','0','256'])
def test_bad_cadence_is_rejected(bad):
    with pytest.raises(SystemExit):
        start_wan_pt.main(['run','--role','mac','--room-id',ROOM,'--revision',SHA,'--local-steps',bad,'--dry-run'])


def test_colab_autotuner_honors_32_and_64_with_two_peers():
    profile=DeviceProfile(cpu_threads=8,ram_free_gib=12,accelerator='cuda',vram_free_gib=8,
                          available_storage_gib=30)
    for requested in (32,64):
        policy=initial_policy(profile, requested_local_steps=requested, max_batch=1)
        assert policy.local_steps==requested
        control=AdaptiveController(profile,policy)
        # No peer tier telemetry; controller should not shorten these intervals.
        assert control.tune(now=20., timeout=60.).local_steps==requested


def test_colab_notebook_no_forced_turn_relay_and_passes_new_default():
    nb=json.loads((Path(__file__).resolve().parents[1]/'notebooks/KCIA_Cloudflare_Mac_Colab.ipynb').read_text())
    body=''.join(''.join(c['source']) for c in nb['cells'])
    assert "os.environ['KCIA_TURN_TRANSPORT'] = 'auto'" in body
    assert "os.environ['KCIA_FORCE_TURN_RELAY'] = '0'" in body
    assert "'--local-steps','32'" in body
    assert "'--snapshot-priority','training'" in body


def test_snapshot_priority_gate_remains_opt_in():
    content=(Path(__file__).resolve().parents[1]/'patchlib/elasticnet.py').read_text()
    assert "os.environ.get('KCIA_SNAPSHOT_PRIORITY','0') == '1'" in content
    assert "f'publish_target={publish_target} grad_accum={args.grad_accum} '" in content
