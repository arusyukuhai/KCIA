"""Anonymous public signaling and explicit live-GUI behavior regression coverage.

These are offline tests; VPS Docker, TURN/ICE and real Cloudflare are tested separately.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import KCIABridge, compose_html

ROOT=Path(__file__).resolve().parents[1]


def test_anonymous_signaling_maintains_bounded_admission():
    server=(ROOT/'src/main.nim').read_text()
    compose=(ROOT/'deploy/vps/compose.yml').read_text()
    startup=(ROOT/'deploy/vps/setup_env.py').read_text()
    for content in (server, compose, startup):
        assert 'APP_TOKEN' not in content
        assert 'ROOM_PUBLISH_TOKEN' not in content
    assert 'MaxMessagesPerTenSeconds' in server
    assert 'MaxWsMessageBytes' in server
    assert 'MaxStoredRooms' in server and 'MaxListedRooms' in server
    assert 'TURN_SECRET' in compose and 'turnSecret' in server
    assert 'if publish and publishSecret' not in server


def test_signaling_persistence_without_client_credentials(tmp_path):
    ctrl=WebviewController(tmp_path)
    bridge=KCIABridge(ctrl)
    assert bridge.defaults()['ws']=='wss://signal.kcia.tech/ws'
    result=bridge.configure_network({'ws':'wss://signal.example.com/ws'})
    assert result['mode']=='internet'
    saved=tmp_path/'.kcia-gui'/'network.json'
    assert json.loads(saved.read_text())=={'version':2,'ws':'wss://signal.example.com/ws'}
    assert saved.stat().st_mode & 0o077 == 0
    second=WebviewController(tmp_path)
    assert second.settings()['ws']=='wss://signal.example.com/ws'
    for bad in ['ws://example.org/ws','https://example.org/ws','wss://user:pass@example.org/ws',
                'wss://signal.example.com/other','wss://signal.example.com/ws?token=x']:
        with pytest.raises(ValueError):bridge.configure_network({'ws':bad})
    assert second.settings()['ws']=='wss://signal.example.com/ws'


def test_active_training_prevents_network_switch(tmp_path,monkeypatch):
    ctrl=WebviewController(tmp_path)
    monkeypatch.setattr(ctrl.train,'status',lambda: True)
    with pytest.raises(RuntimeError,match='Disconnect'):
        ctrl.configure_network({'ws':'wss://signal.example.com/ws'})
    assert not (tmp_path/'.kcia-gui'/'network.json').exists()


def test_network_probe_requires_actual_websocket_pong(tmp_path,monkeypatch):
    import patchlib.webview_controller as vc
    import websockets
    ctrl=WebviewController(tmp_path)
    sent=[]
    class FakeWebsocket:
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        async def send(self,s):sent.append(json.loads(s))
        async def recv(self):return '{"type":"pong"}'
    monkeypatch.setattr(websockets,'connect',lambda url,**kwargs: FakeWebsocket())
    monkeypatch.setattr(vc,'list_public_rooms',lambda ws,timeout:[{'room_id':'a'*32}])
    reply=ctrl.probe_network({'ws':'wss://signal.example.org/ws'})
    assert reply['connected'] and reply['public_rooms']==1
    assert sent==[{'type':'ping'}]


def test_backend_generates_id_when_not_supplied(tmp_path):
    ctrl=WebviewController(tmp_path)
    sources=[{'repo':'org/data','revision':'main','config':None,'split':'train',
              'columns':{'text':'text'},'weight':1.0}]
    result=ctrl.save({'model':'org/model','revision':'main','task':'causal-lm',
                      'sources':sources,'seed':42})
    assert len(result['room'])>=24
    assert ctrl._room_path(result['room']).exists()
    assert 'APP_TOKEN' not in json.dumps(result)


def test_webview_has_no_room_field_or_token_required():
    html=compose_html()
    script=(ROOT/'web/desktop_webview.js').read_text()
    assert 'id="room" type="hidden"' in html
    assert 'id="newRoom"' not in html
    assert 'id="networkMode"' in html and 'id="connectNetwork"' in html
    assert 'id="joinInvite"' in html and 'id="copyInvite"' in html
    assert 'id="ws"' in html
    assert 'APP_TOKEN' not in html
    assert 'void registerSavedModel(saved.room' in script
    assert 'await api(\'register\',{room,ws,room_public:roomPublic})' in script
    assert "await api('join',{room,ws})" in script
    assert 'await prepareNewDraft()' in script
