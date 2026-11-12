"""Click-level Chromium regression for deliberate, non-automatic reset controls."""
from pathlib import Path

from patchlib.webview_gui import compose_html


def test_full_reset_dialog_requires_preview_and_explicit_confirmation():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                                  args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1380,'height':860})
        errors=[]
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.set_content(compose_html(),wait_until='domcontentloaded')
        page.evaluate('''() => {
          window.__started=0; window.__resets=0; window.__windows=[];
          const defaults={room:'A'.repeat(32),model:'test/model',revision:'main',
            task:'causal-lm',ws:'ws://127.0.0.1:8080/ws',seed:42,kind:'loha',
            initialization:'pretrained',reinit_every:20,lr:0.0001,local_steps:4,
            disk_cache_fraction:.6,wide_lora_rank:4,push_fanout:2,max_journal_gb:4,
            streaming_base:false,sources:[]};
          window.pywebview={api:new Proxy({
            defaults:async()=>defaults,
            training_status:async()=>({running:false,telemetry:[],logs:''}),
            local_models:async()=>({models:[]}),
            list_rooms:async()=>({rooms:[]}),
            start_training:async()=>{window.__started++;return {pid:123};},
            preview_full_reset:async()=>({preview_id:'test-preview',root:'/data',
              total_bytes:4096,total_entries:1,
              targets:[{path:'/data/.kcia-gui',entries:1}]}),
            request_full_reset:async payload=>{window.__resets++;return {scheduled:true};},
            window_action:async payload=>{window.__windows.push(payload);return {ok:true};},
          },{get:(obj,key)=>key==='then'?undefined:obj[key]||(async()=>({}))})};
          window.dispatchEvent(new Event('pywebviewready'));
        }''')
        page.locator('#bridgeState.online').wait_for(timeout=5000)
        assert page.evaluate('window.__started')==0, 'GUI boot cannot start training'
        page.locator('#resetAllData').click()
        assert page.locator('#resetDialog').is_visible()
        assert not page.locator('#resetExecute').is_enabled()
        page.locator('#resetConfirmation').fill('DELETE KCIA')
        assert not page.locator('#resetExecute').is_enabled(), 'preview required'
        page.locator('#resetPreview').click()
        page.wait_for_function("document.querySelector('#resetPreviewResult').textContent.includes('削除対象:')")
        assert not page.locator('#resetExecute').is_enabled(), 'typed confirm resets after preview'
        page.locator('#resetConfirmation').fill('DELETE KCIA')
        assert page.locator('#resetExecute').is_enabled()
        page.locator('#resetIncludeCli').check()
        assert not page.locator('#resetExecute').is_enabled(), 'changed scope requires new preview'
        page.locator('#resetPreview').click()
        page.wait_for_function("document.querySelector('#resetPreviewResult').textContent.includes('削除対象:')")
        page.locator('#resetConfirmation').fill('DELETE KCIA')
        page.locator('#resetExecute').click()
        page.wait_for_function('window.__windows.length===1')
        assert page.evaluate('window.__started')==0
        assert page.evaluate('window.__resets')==1
        assert page.evaluate('window.__windows[0].action')=='close'
        assert not errors,errors
        browser.close()
