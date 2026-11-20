"""v135: production WAN defaults, GUI-only local debug and v134 training policy.

All tests are offline and must not open sockets or load a real model.
"""
from pathlib import Path
import json
import os
import subprocess

import pytest

from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import compose_html
from patchlib.gui import DesktopController
from patchlib.roomid import create_room_id
from patchlib.network_defaults import WAN_SIGNALING_URL, LOCAL_DEBUG_SIGNALING_URL

ROOT=Path(__file__).resolve().parents[1]


def make_room(ctrl):
    rid=create_room_id()
    ctrl.save({'room':rid,'model':'sshleifer/tiny-gpt2','revision':'main','task':'causal-lm',
        'seed':42,'initialization':'pretrained',
        'sources':[{'repo':'roneneldan/TinyStories','revision':'main','config':None,
                    'split':'train','columns':{'text':'text'},'weight':1.0}]})
    return rid


def test_defaults_are_wan_and_public_directory(tmp_path):
    ctrl=WebviewController(tmp_path)
    settings=ctrl.settings()
    assert settings['ws']==WAN_SIGNALING_URL
    assert settings['publish_mode']=='wan'
    assert settings['snapshot_priority']=='training'
    assert settings['turn_transport']=='auto'
    assert settings['relay_only'] is False
    assert settings['local_steps']==32
    assert settings['reinit_every']==20
    html=compose_html()
    assert 'id="networkMode"' in html
    assert '<option value="internet" selected>' in html
    assert f'value="{WAN_SIGNALING_URL}"' in html
    assert '<option value="public" selected' in html
    for element in ('trainPublishMode','trainLocalSteps','trainSnapshotPriority',
                    'trainTurnTransport','trainRelayOnly'):
        assert f'id="{element}"' in html
    js=(ROOT/'web/desktop_webview.js').read_text()
    assert 'const DEFAULT_WAN_WS=' in js
    assert "void api('publicModels',{ws:url})" in js
    assert 'selectedPublishSteps(plan)' in js
    assert 'min_publish_steps:manual?steps:0' in js
    assert 'shard_mb:1' in js


def test_old_local_default_migrates_to_wan_and_debug_is_session_only(tmp_path):
    ctrl=WebviewController(tmp_path)
    path=tmp_path/'.kcia-gui'/'network.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'version':1,'ws':LOCAL_DEBUG_SIGNALING_URL}))
    assert WebviewController(tmp_path).settings()['ws']==WAN_SIGNALING_URL
    ctrl.configure_network({'ws':LOCAL_DEBUG_SIGNALING_URL})
    # Local debug can be selected during a session, but is never the next GUI boot default.
    assert json.loads(path.read_text())=={'version':2,'ws':WAN_SIGNALING_URL}
    assert WebviewController(tmp_path).settings()['ws']==WAN_SIGNALING_URL
    ctrl.configure_network({'ws':WAN_SIGNALING_URL})
    assert WebviewController(tmp_path).settings()['ws']==WAN_SIGNALING_URL


def test_gui_launch_includes_v134_policy_without_changing_signed_room(tmp_path,monkeypatch):
    ctrl=WebviewController(tmp_path)
    room=make_room(ctrl)
    launched=[]
    monkeypatch.setattr(ctrl.train,'start',lambda cmd, logfile: launched.append(list(cmd)) or 7123)
    opts={'room':room, 'ws':WAN_SIGNALING_URL,'kind':'loha','rank':32,
          'local_steps':32,'min_publish_steps':32,'snapshot_priority':'training',
          'turn_transport':'auto','relay_only':False,'shard_mb':1,
          'reinit_every':20,'rounds':0}
    result=ctrl.start(opts)
    cmd=launched[-1]
    def opt(name):return cmd[cmd.index(name)+1]
    assert result['pid']==7123
    assert opt('--ws')==WAN_SIGNALING_URL
    assert opt('--room-id')==room
    assert opt('--local-steps')=='32'
    assert opt('--min-publish-steps')=='32'
    assert opt('--snapshot-priority')=='training'
    assert opt('--turn-transport')=='auto'
    assert '--no-relay-only' in cmd
    assert opt('--shard-mb')=='1'
    assert opt('--reinit-every')=='20'
    assert '--room-public' not in cmd
    # Changed peer-only training policy must not be embedded in signed room manifest.
    raw=ctrl._room_path(room).read_text()
    assert 'min_publish_steps' not in raw
    assert 'snapshot_priority' not in raw


def test_gui_custom_cadence_and_debug_transport_are_explicit(tmp_path,monkeypatch):
    ctrl=WebviewController(tmp_path)
    room=make_room(ctrl)
    launched=[]
    monkeypatch.setattr(ctrl.train,'start',lambda cmd, logfile: launched.append(list(cmd)) or 44)
    ctrl.start({'room':room, 'ws':LOCAL_DEBUG_SIGNALING_URL, 'kind':'loha','rank':32,
        'local_steps':64,'min_publish_steps':64,'snapshot_priority':'snapshot',
        'turn_transport':'tcp','relay_only':True,'shard_mb':2})
    cmd=launched[0]
    assert cmd[cmd.index('--min-publish-steps')+1]=='64'
    assert cmd[cmd.index('--snapshot-priority')+1]=='snapshot'
    assert cmd[cmd.index('--turn-transport')+1]=='tcp'
    assert '--relay-only' in cmd
    assert cmd[cmd.index('--shard-mb')+1]=='2'


def test_command_rejects_invalid_per_peer_policy(tmp_path):
    ctrl=WebviewController(tmp_path)
    room=make_room(ctrl)
    manifest=ctrl._room_path(room)
    params=dict(room=room,model='sshleifer/tiny-gpt2',revision='main',base=manifest.parent/'genesis.fstate',
                manifest=manifest,ws=WAN_SIGNALING_URL,task='causal-lm',kind='loha',rank=32,
                journal=manifest.parent/'events.sqlite3',auto_state=True)
    for broken in ({'min_publish_steps':-1}, {'min_publish_steps':10001},
                   {'snapshot_priority':'no'}, {'turn_transport':'plain'},
                   {'relay_only':'false'}, {'shard_mb':0}, {'shard_mb':65}):
        with pytest.raises(ValueError):
            DesktopController.build_command(**params,**broken)


def test_gui_training_preferences_persist_and_reject_corruption(tmp_path):
    ctrl=WebviewController(tmp_path)
    params={'publish_mode':'custom','local_steps':64,'snapshot_priority':'snapshot',
            'turn_transport':'tcp','relay_only':True}
    assert ctrl.configure_training_preferences(params)==params
    path=tmp_path/'.kcia-gui'/'training-preferences.json'
    assert path.stat().st_mode & 0o077 == 0
    reopened=WebviewController(tmp_path)
    for key,value in params.items():
        assert reopened.settings()[key]==value
    for bad in ({**params,'local_steps':3},{**params,'local_steps':True},
                {**params,'publish_mode':'invalid'},{**params,'relay_only':'false'}):
        with pytest.raises(ValueError):ctrl.configure_training_preferences(bad)
    assert reopened.training_preferences()==params
    path.write_text('not json')
    assert reopened.training_preferences()['publish_mode']=='wan'
