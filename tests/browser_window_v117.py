"""GUI click and drag-hitbox regression in Chromium; native Cocoa movement untestable here."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from patchlib.webview_gui import compose_html

JS_BRIDGE='''(() => {
  window.__calls=[];
  const base={room:'R'.repeat(32),model:'roneneldan/TinyStories-1M',revision:'main',task:'causal-lm',ws:'ws://127.0.0.1:8080/ws',seed:42,kind:'loha',initialization:'random',reinit_every:20,lr:0.0001,disk_cache_fraction:.6,wide_lora_rank:4,push_fanout:2,max_journal_gb:4,streaming_base:false,sources:[]};
  const apis={
    defaults:async()=>base,
    window_action:({action})=>{window.__calls.push(action);return new Promise(()=>{});},
    device_capabilities:async()=>({device:'cpu',ram_bytes:8000000000,vram_bytes:0,mps_available:false,cuda_available:false}),
    training_status:async()=>({running:false,telemetry:[],logs:''}),
    local_models:async()=>({models:[]}),
    list_rooms:async()=>({rooms:[]}),
    recommend_training:async()=>({rank:8,can_train:true,kind:'loha'}),
    eval_status:async()=>({running:false}),
    available_checkpoints:async()=>({checkpoints:[]}),
    training_history:async()=>({telemetry:[]}),
  };
  window.pywebview={api:new Proxy(apis,{get(obj,k){if(k==='then')return;return obj[k] || (async()=>({}));}})};
  window.dispatchEvent(new Event('pywebviewready'));
})()'''

def test_browser():
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1380,'height':860},device_scale_factor=1)
        errors=[]
        page.on('pageerror',lambda e: errors.append(str(e)))
        page.set_content(compose_html(),wait_until='domcontentloaded')
        page.evaluate(JS_BRIDGE)
        page.locator('#bridgeState.online').wait_for(timeout=7000)
        for name in ['windowMinimize','windowMaximize','windowClose']:
            page.locator('#'+name).click()
        assert page.evaluate('window.__calls')==['minimize','toggle_maximize','close']
        # Each OS operation is deliberately pending forever in this mock.
        # App navigation and the Create Model action remain clickable.
        page.locator('[data-tab="new"]').click()
        assert page.locator('#panel-new.active').count()==1
        assert page.locator('#save').is_visible()
        page.locator('[data-tab="models"]').click()
        assert page.locator('#panel-models.active').count()==1
        # Drag hit boxes must resolve to the draggable parent, not SVG/text.
        brand=page.locator('.brand-logo')
        rect=brand.bounding_box()
        assert rect and rect['width']>100 and rect['height']>30
        brand_hit=page.evaluate('''([x,y])=>document.elementFromPoint(x,y)?.closest('.pywebview-drag-region')?.className''',
                   [rect['x']+20,rect['y']+rect['height']/2])
        assert 'brand-logo' in brand_hit,brand_hit
        title=page.locator('.window-drag')
        rect=title.bounding_box()
        assert rect and rect['width']>140
        title_hit=page.evaluate('''([x,y])=>document.elementFromPoint(x,y)?.closest('.pywebview-drag-region')?.className''',
                   [rect['x']+80,rect['y']+rect['height']/2])
        assert 'window-drag' in title_hit,title_hit
        control=page.locator('#windowClose')
        r=control.bounding_box()
        control_hit=page.evaluate('''([x,y])=>document.elementFromPoint(x,y)?.id''',
                 [r['x']+r['width']/2,r['y']+r['height']/2])
        assert control_hit=='windowClose',control_hit
        assert page.evaluate("getComputedStyle(document.querySelector('.brand-logo')).justifyContent")=='flex-start'
        # The SVG left edge must align with the parent content area, no centering.
        svg=page.locator('.kcia-logo').bounding_box()
        assert abs(svg['x']-(brand.bounding_box()['x']+10))<1.5,(svg,brand.bounding_box())  # v119 restored 10px brand padding
        assert not errors,errors
        page.screenshot(path='/mnt/data/kcia_v117_gui_preview.png')
        print('PASS: 3 window controls dispatched without waiting, tab navigation responsive; native drag hit boxes, left-aligned logo')
        browser.close()

if __name__=='__main__': test_browser()
