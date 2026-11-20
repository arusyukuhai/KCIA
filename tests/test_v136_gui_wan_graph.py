"""WAN-first directory source classification and graph freshness regressions.

All tests run offline; no signaling sockets or user state are touched.
"""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from patchlib.webview_controller import WebviewController
from patchlib.roomid import create_room_id
from patchlib.network_defaults import WAN_SIGNALING_URL,LOCAL_DEBUG_SIGNALING_URL

ROOT=Path(__file__).resolve().parents[1]


def make_room(controller):
    room=create_room_id()
    controller.save({'room':room,'model':'sshleifer/tiny-gpt2','revision':'main','task':'causal-lm',
        'seed':42,'initialization':'pretrained',
        'sources':[{'repo':'roneneldan/TinyStories','revision':'main','config':None,
                    'split':'train','columns':{'text':'text'},'weight':1.0}]})
    return room


def test_legacy_local_debug_settings_never_override_wan(tmp_path):
    cfg=tmp_path/'.kcia-gui'/'network.json'
    cfg.parent.mkdir(parents=True)
    for version in (1,2,3):
        cfg.write_text(json.dumps({'version':version,'ws':LOCAL_DEBUG_SIGNALING_URL}))
        assert WebviewController(tmp_path).settings()['ws']==WAN_SIGNALING_URL
    ctrl=WebviewController(tmp_path)
    resp=ctrl.configure_network({'ws':LOCAL_DEBUG_SIGNALING_URL})
    assert resp['mode']=='local'
    assert json.loads(cfg.read_text())['ws']==WAN_SIGNALING_URL
    assert WebviewController(tmp_path).settings()['ws']==WAN_SIGNALING_URL


def test_wan_receipt_only_after_real_server_ack_and_matches_host(tmp_path,monkeypatch):
    ctrl=WebviewController(tmp_path)
    room=make_room(ctrl)
    listing=lambda ws:ctrl.local_models({'ws':ws})['models'][0]
    assert listing(WAN_SIGNALING_URL)['wan_registered'] is False
    def fail(*args,**kwargs):raise ConnectionError('server down')
    monkeypatch.setattr('patchlib.webview_controller.register_room_manifest',fail)
    with pytest.raises(ConnectionError):
        ctrl.register_room({'room':room,'ws':WAN_SIGNALING_URL,'room_public':True})
    assert listing(WAN_SIGNALING_URL)['wan_registered'] is False
    async def ok(*args,**kwargs):return None
    monkeypatch.setattr('patchlib.webview_controller.register_room_manifest',ok)
    result=ctrl.register_room({'room':room,'ws':WAN_SIGNALING_URL,'room_public':True})
    assert result['registered'] is True
    assert listing(WAN_SIGNALING_URL)['wan_registered'] is True
    assert listing(WAN_SIGNALING_URL)['registered_public'] is True
    assert listing('wss://other.example/ws')['wan_registered'] is False
    assert listing(LOCAL_DEBUG_SIGNALING_URL)['wan_registered'] is False
    marker=ctrl._wan_registration_path(room)
    assert marker.is_file() and marker.stat().st_mode&0o077==0
    assert 'token' not in marker.read_text()
    assert 'secrets' not in marker.read_text()


def test_local_debug_registration_does_not_acquire_wan_badge(tmp_path,monkeypatch):
    ctrl=WebviewController(tmp_path)
    room=make_room(ctrl)
    async def ok(*args,**kwargs):return None
    monkeypatch.setattr('patchlib.webview_controller.register_room_manifest',ok)
    ctrl.register_room({'room':room,'ws':LOCAL_DEBUG_SIGNALING_URL,'room_public':True})
    assert ctrl.local_models({'ws':WAN_SIGNALING_URL})['models'][0]['wan_registered'] is False


def test_frontend_source_and_chart_cadence():
    js=(ROOT/'web/desktop_webview.js').read_text()
    html=(ROOT/'web/desktop_webview.html').read_text()
    # Regression: every localhost saved room was previously shown in WAN scope.
    assert "x.local&&get('networkMode')==='internet'" not in js
    assert 'x.on_wan||x.wan_registered' in js
    assert "const local=await api('localModels',{ws:url})" in js
    assert '<option value="public" selected data-t="public">WANルーム</option>' in html
    assert "setInterval(()=>{if(!document.hidden)void poll();},900);" in js
    assert '800-(Date.now()-lastChartPaint)' in js
    assert 'lastHistoryFetch<12000' in js
    assert 'requestAnimationFrame' in js
    assert 'points.length>360' in js
    # New/old public cards are displayed truthfully instead of local defaults.
    assert "item.wan_registered?t('registeredWan'):t('localScope')" in js


def test_frontend_scope_filter_executes_on_sample_rooms():
    """Exercise actual JS filter, not a Python copy of its logic."""
    js=(ROOT/'web/desktop_webview.js').read_text()
    src=js[js.index('function modelFiltered(){'):js.index('function statusClass(')]
    code='''
const models=[
 {room_id:'a',model:'a',datasets:[],local:true,on_wan:false,wan_registered:false},
 {room_id:'b',model:'b',datasets:[],local:true,on_wan:false,wan_registered:true},
 {room_id:'c',model:'c',datasets:[],local:false,on_wan:true,wan_registered:true}
];
let scope='public',mode='internet';
function get(name){return name==='modelScope'?scope:name==='networkMode'?mode:'';}
'''+src+'''
function check(s,m,expected){scope=s;mode=m;const keys=modelFiltered().map(x=>x.room_id).sort().join('');
if(keys!==expected)throw Error(s+' '+m+': '+keys+' != '+expected);}
check('public','internet','bc');
check('public','local','');
check('local','internet','a');
check('all','internet','abc');
'''
    result=subprocess.run(['node','-e',code],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
