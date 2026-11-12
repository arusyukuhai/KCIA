"""v120: GUI never waits on external services or heavy device probing.

All assertions here are offline, and the Chromium test deliberately leaves
external Python bridge calls unresolved (a common failure mode on macOS).
"""
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import compose_html, run_webview


def test_startup_defaults_restore_publish4_reinit20(tmp_path):
    controller = WebviewController(tmp_path)
    defaults = controller.settings()
    assert defaults['reinit_every'] == 20
    assert defaults['local_steps'] == 4
    html = compose_html()
    assert 'id="trainReinitEvery" min="0" max="100000" value="20"' in html
    assert 'Default 20 for new rooms' in html
    assert "$('trainReinitEvery').value='20'" in html
    assert "window.addEventListener('load'" in html and "stylesheet.rel = 'stylesheet'" in html  # after paint, nonblocking


def test_busy_status_does_not_queue_behind_a_long_disk_operation(tmp_path):
    ctrl = WebviewController(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def hold_room_lock():
        with ctrl.lock:
            started.set()
            release.wait(timeout=3)

    worker = threading.Thread(target=hold_room_lock)
    worker.start()
    try:
        assert started.wait(1)
        start = time.monotonic()
        result = ctrl.status()
        assert time.monotonic() - start < .1
        assert result['busy'] is True
        assert result['running'] is False
    finally:
        release.set()
        worker.join(timeout=2)


def test_frameless_default_and_native_recovery(tmp_path,monkeypatch):
    creations=[]
    class FakeWindow:
        def destroy(self): pass
    class FakeWebview:
        settings={}
        def create_window(self,*args,**kwargs):
            creations.append(kwargs)
            return FakeWindow()
        def start(self,**kwargs): pass
    monkeypatch.delenv('KCIA_GUI_SAFE',raising=False)
    run_webview(workdir=tmp_path,webview_module=FakeWebview())
    assert creations[-1]['frameless'] is True
    monkeypatch.setenv('KCIA_GUI_SAFE','1')
    run_webview(workdir=tmp_path,webview_module=FakeWebview())
    assert creations[-1]['frameless'] is False
    assert 'native-titlebar' in creations[-1]['html']


def test_renderer_unresponsive_bridge_does_not_block_clicks():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1380,'height':860})
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_content(compose_html(),wait_until='domcontentloaded')
        page.evaluate('''() => {
          window.__statusCalls=0;
          window.__rpc={
             defaults: async()=>({room:'R'.repeat(32),model:'roneneldan/TinyStories-1M',revision:'main',task:'causal-lm',ws:'wss://offline.example/ws',seed:42,kind:'loha',initialization:'random',reinit_every:20,lr:0.0001,local_steps:4,disk_cache_fraction:.6,wide_lora_rank:4,push_fanout:2,max_journal_gb:4,streaming_base:false,sources:[]}),
             training_status: async()=>{window.__statusCalls++;return {running:false,telemetry:[],logs:''};},
             local_models: async()=>({models:[]}),
             list_rooms: ()=>new Promise(()=>{}),
             device_capabilities: ()=>new Promise(()=>{}),
             recommend_training: ()=>new Promise(()=>{}),
             eval_status: async()=>({running:false}),
             available_checkpoints: async()=>({checkpoints:[]}),
             new_room: async()=>({room:'W'.repeat(32)}),
          };
          window.pywebview={api:new Proxy(window.__rpc,{get(target,key){if(key==='then')return;return target[key]||(async()=>({}));}})};
          window.dispatchEvent(new Event('pywebviewready'));
        }''')
        page.locator('#bridgeState.online').wait_for(timeout=6000)
        assert page.locator('#trainReinitEvery').input_value()=='20'
        for tab in ['new','training','models','network','new']:
            start=time.monotonic()
            page.locator(f'button[data-tab="{tab}"]').click(timeout=2500)
            assert page.locator(f'#panel-{tab}.active').count()==1
            assert time.monotonic()-start<1.5
        assert page.locator('#save').is_enabled()
        assert not errors,errors
        browser.close()


def test_renderer_chart_renders_are_bounded_by_time():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1380,'height':860})
        page.set_content(compose_html(),wait_until='domcontentloaded')
        # Static WebKit stress: plot 1600 actual samples, not an empty chart.
        result=page.evaluate('''() => {
            currentTab='training';
            document.querySelectorAll('.screen').forEach(p=>p.classList.toggle('active',p.id==='panel-training'));
            chartTelemetry=Array.from({length:1600},(_,i)=>({steps:i+1,loss:10-i*.003,accuracy:i/1600,pt_accuracy:i/1600,sft_accuracy:i/1600,ppo_reward:i/1000,ppo_kl:i/10000,ppo_critic_loss:3-i/1000,elapsed_s:i}));
            let calls=0;
            const original=renderObjectivePlots;
            renderObjectivePlots=function(){calls++;return original();};
            const start=performance.now();
            for(let i=0;i<200;i++)scheduleChartPaint();
            return new Promise(resolve=>setTimeout(()=>resolve({calls,elapsed:performance.now()-start}),600));
        }''')
        assert result['calls']==1,result
        assert result['elapsed']<3000,result
        browser.close()
