"""Run with python tests/browser_create_model_v114.py; requires Chromium+Playwright.
Tests actual interactive assembled HTML/JS, not regex or a simulated DOM.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from patchlib.webview_gui import compose_html

BRIDGE = r'''
(() => {
  const savedModels=[];
  const base={room:'R'.repeat(32),model:'roneneldan/TinyStories-1M',revision:'77f1b168e219585646439073245fe87e56b3023e',task:'causal-lm',ws:'wss://signal.example.com/ws',seed:42,kind:'loha',initialization:'random',reinit_every:20,lr:1e-4, disk_cache_fraction:.6,wide_lora_rank:4,push_fanout:2,max_journal_gb:4,streaming_base:false,sources:[{repo:'roneneldan/TinyStories',revision:'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',config:null,split:'train',columns:{text:'text'},weight:1}]};
  const methods={
    defaults:async()=>base,
    new_room:async()=>({room:'S'.repeat(32)}),
    save_room:async(data)=>{window.__savedPayload=data; savedModels.push({room_id:data.room,model:data.model,revision:data.revision,datasets:data.sources.map(x=>x.repo),local:true,task:data.task});return {room:data.room,sha256:'a'.repeat(64)};},
    register_room:async()=>({public:false}),
    local_models:async()=>({models:savedModels}),
    list_rooms:async()=>({rooms:[]}),
    model_compatibility:async()=>({status:'unknown'}),
    training_status:async()=>({running:false,logs:'',telemetry:[]}),
    device_capabilities:async()=>({device:'cpu',ram_bytes:16*1073741824,vram_bytes:0,mps_available:false,cuda_available:false}),
    recommend_training:async()=>({}),
    discover_hub_assets:async({role,task})=>({candidates:[{id:'example/tiny-llm',kind:'model',revision:'b'.repeat(40),downloads:999,reason:'metadata-only · task='+task,gated:false}],role}),
    mixed_settings:async()=>({enabled:false}),
    eval_status:async()=>({running:false}),
    available_checkpoints:async()=>({checkpoints:[]}),
    training_history:async()=>({telemetry:[]}),
    storage_paths:async()=>({}),
  };
  window.pywebview={api:new Proxy(methods,{get(obj,key){ if(key==='then')return undefined;return obj[key] || (async()=>({}));}})};
  window.dispatchEvent(new Event('pywebviewready'));
})()
'''

def test_browser():
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=browser.new_page(viewport={'width':1440,'height':900})
        errors=[]
        page.on('pageerror',lambda e: errors.append(str(e)))
        page.set_content(compose_html(),wait_until='domcontentloaded')
        page.evaluate(BRIDGE)
        page.locator('#bridgeState.online').wait_for(timeout=7000)
        page.locator('[data-tab="new"]').click()
        assert page.locator('#pickBaseModel').is_visible()
        assert page.locator('#model').input_value()=='roneneldan/TinyStories-1M'
        assert page.locator('#newDatasetTags').inner_text().find('TinyStories')>=0
        # No manual room_id; default dataset is preattached.
        assert page.locator('#room').get_attribute('type')=='hidden'
        page.locator('#pickBaseModel').click()
        page.locator('#hubPicker').wait_for(state='visible',timeout=5000)
        page.locator('.hub-candidate').first.wait_for(timeout=5000)
        page.locator('.hub-candidate').first.click()
        assert page.locator('#hubPicker').is_hidden()
        assert page.locator('#model').input_value()=='example/tiny-llm'
        assert page.locator('#revision').input_value()=='b'*40
        assert page.locator('#modelInitialization').input_value()=='pretrained'
        page.locator('#save').click()
        page.locator('#panel-models.active').wait_for(timeout=9000)
        payload=page.evaluate('window.__savedPayload')
        assert payload['model']=='example/tiny-llm'
        assert payload['room']=='R'*32
        assert payload['sources'][0]['repo']=='roneneldan/TinyStories'
        assert page.locator('#modelCards').inner_text().find('tiny-llm')>=0
        assert not errors,errors
        print('PASS: initialized GUI, model picker, SHA pin, create model, saved room and model list')
        # The VPS may be offline; local creation must still finish and expose recovery feedback.
        offline=browser.new_page(viewport={'width':1200,'height':800})
        offline_errors=[]
        offline.on('pageerror',lambda e:offline_errors.append(str(e)))
        offline.set_content(compose_html(),wait_until='domcontentloaded')
        offline.evaluate(BRIDGE.replace('register_room:async()=>({public:false}),', "register_room:async()=>{throw Error('VPS offline')},"))
        offline.locator('#bridgeState.online').wait_for(timeout=7000)
        offline.locator('[data-tab="new"]').click()
        offline.locator('#save').click()
        offline.locator('#panel-models.active').wait_for(timeout=9000)
        assert offline.evaluate('window.__savedPayload.model')=='roneneldan/TinyStories-1M'
        offline.locator('#feedback').get_by_text('VPS offline').wait_for(timeout=5000)
        # Error path: failed saves remain on the creation form and re-enable button.
        offline.locator('[data-tab="new"]').click()
        offline.evaluate("(()=>{window.pywebview.api.save_room=async()=>{throw Error('backend disk full')};})()")
        offline.locator('#save').click()
        offline.wait_for_function("document.querySelector('#newModelStatus').textContent.includes('backend disk full')",timeout=6000)
        assert offline.locator('#panel-new').get_attribute('class').find('active')>=0
        assert offline.locator('#save').is_enabled()
        assert not offline_errors,offline_errors
        print('PASS: VPS-offline local save, failed-save visible feedback, button re-enabled')
        browser.close()
if __name__=='__main__':test_browser()
