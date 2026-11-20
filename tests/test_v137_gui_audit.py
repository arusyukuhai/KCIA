"""v137: regression coverage for costly GUI polling and concurrent WAN probes."""
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from patchlib.webview_controller import WebviewController
from patchlib.live_telemetry import recent_samples
from patchlib.webview_gui import compose_html


def test_telemetry_reader_survives_atomic_rotation(tmp_path):
    path=tmp_path/'history.jsonl'
    path.write_text(json.dumps({'steps':1,'elapsed_s':1.0,'state':'training'})+'\n')
    assert recent_samples(path)[0]['steps']==1
    real_open=Path.open
    def missing(self,*args,**kwargs):
        if self==path:
            raise FileNotFoundError('replaced during rotation')
        return real_open(self,*args,**kwargs)
    with patch.object(Path,'open',missing):
        assert recent_samples(path)==[]


def test_parallel_wan_probe_shorter_than_sequential(tmp_path,monkeypatch):
    """HTTP directory and WSS probe must not occupy two timeout windows."""
    import sys
    class Socket:
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        async def send(self,message):
            assert json.loads(message)=={'type':'ping'}
        async def recv(self):
            await asyncio.sleep(.22)
            return json.dumps({'type':'pong'})
    class WebSockets:
        def connect(self,*a,**kw):return Socket()
    monkeypatch.setitem(sys.modules,'websockets',WebSockets())
    def list_rooms(*a,**kw):
        time.sleep(.22)
        return [dict(room_id='placeholder')]
    monkeypatch.setattr('patchlib.webview_controller.list_public_rooms',list_rooms)
    start=time.monotonic()
    result=WebviewController(tmp_path).probe_network({'ws':'wss://signal.kcia.tech/ws'})
    elapsed=time.monotonic()-start
    assert result['connected'] is True
    assert result['public_rooms']==1
    assert elapsed < .40, 'Sequential WAN probes would exceed 0.44 seconds'


@pytest.fixture
def page():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                                    args=['--no-sandbox'])
        try:
            p=browser.new_page()
            p.route('https://fonts.googleapis.com/**', lambda route:route.abort())
            p.route('https://use.typekit.net/**',lambda route:route.abort())
            p.set_content(compose_html(),wait_until='load')
            yield p
        finally:
            browser.close()


def test_repeated_status_does_not_rewrite_unchanged_dom_or_graph(page):
    outcome=page.evaluate('''async ()=>{
      const output={};
      const el=document.getElementById('monitorLoss');
      let mutations=0;
      const monitor=new MutationObserver(changes=>mutations+=changes.length);
      monitor.observe(el,{childList:true,characterData:true,subtree:true});
      setText('monitorLoss','5.4321');
      await Promise.resolve();
      setText('monitorLoss','5.4321');
      await Promise.resolve();
      output.mutations=mutations;
      run={running:true};
      latestTelemetry=[{steps:2,connected_peers:['peer1']}];
      drawGraph();const initial=document.getElementById('networkGraph').firstChild;
      drawGraph();output.reusedGraph=initial===document.getElementById('networkGraph').firstChild;
      const initialSig=chartSignature();
      latestTelemetry=[{steps:2,connected_peers:['peer2']}];
      output.chartStatusDoesNotInvalidate=initialSig===chartSignature();
      latestTelemetry=[{steps:3,loss:4.1}];
      output.chartLossInvalidates=initialSig!==chartSignature();
      return output;
    }''')
    assert outcome == {'mutations':1,'reusedGraph':True,'chartStatusDoesNotInvalidate':True,
                       'chartLossInvalidates':True}


def test_compatibility_check_shared_across_rooms_and_refresh(page):
    result=page.evaluate('''async ()=>{
      let calls=0;
      window.pywebview={api:{
        defaults:async()=>({}),
        model_compatibility:async ()=>{calls++;await new Promise(r=>setTimeout(r,25));return {status:'candidate',parameters:123};}
      }};
      models=[
        {room_id:'room-one',model:'org/same',revision:'main',datasets:[],local:true,online:0},
        {room_id:'room-two',model:'org/same',revision:'main',datasets:[],local:true,online:0}
      ];
      queueCompatibility(models[0]);queueCompatibility(models[1]);
      await new Promise(r=>setTimeout(r,120));
      const initial=calls;
      cachedLocalRooms=[{room_id:'room-three',model:'org/same',revision:'main',datasets:[],local:true,online:0}];
      cachedPublicRooms=[];
      updateModelDirectory();
      const cached=models[0]?.compatibility?.parameters;
      await new Promise(r=>setTimeout(r,30));
      return {initial,calls,cached};
    }''')
    assert result == {'initial':1,'calls':1,'cached':123}


def test_out_of_order_model_plan_cannot_apply_to_new_selection(page):
    outcome=page.evaluate('''async ()=>{
      let resolvePlan;
      window.pywebview={api:{defaults:async()=>({}),recommend_training:()=>new Promise(r=>resolvePlan=r)}};
      document.getElementById('room').value='old-room';
      document.getElementById('model').value='org/old';
      document.getElementById('revision').value='main';
      document.getElementById('kind').value='loha';
      const pending=refreshAutoPlan(true).then(()=>({ok:true}),e=>({ok:false,error:e.message}));
      await new Promise(r=>setTimeout(r,15));
      document.getElementById('model').value='org/new';
      resolvePlan({rank:32,local_steps:4,max_length:128,can_train:true});
      return await pending;
    }''')
    assert outcome['ok'] is False
    assert 'changed' in outcome['error']
