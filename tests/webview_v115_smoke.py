from patchlib.webview_gui import compose_html
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1380,'height':860})
 errors=[]
 page.on('pageerror',lambda exc:errors.append(str(exc)))
 page.set_content(compose_html(),wait_until='domcontentloaded')
 page.evaluate('''window.__windowActions=[];
 window.pywebview={api:{
  defaults:async()=>({room:'rtest',model:'roneneldan/TinyStories-1M',revision:'main',task:'causal-lm',ws:'ws://127.0.0.1:8080/ws',seed:42,kind:'loha',initialization:'random',reinit_every:20,lr:0.0001,disk_cache_fraction:0.6,wide_lora_rank:4,push_fanout:2,max_journal_gb:4,streaming_base:false,sources:[]}),
  window_action:async p=>{window.__windowActions.push(p.action);return {ok:true};},
  device_capabilities:async()=>({device:'cpu',ram_bytes:8000000000,vram_bytes:0,mps_available:false,cuda_available:false}),
  training_status:async()=>({running:false,telemetry:[],logs:''}),
  local_models:async()=>({models:[]}),
  list_rooms:async()=>({rooms:[]}),
  recommend_training:async()=>({rank:16,local_training_rank_estimate:8,kind:'loha',max_length:64,local_steps:4,can_train:true,wide_lora_rank:4}),
 }};''')
 page.wait_for_timeout(750)
 for x in ['windowMinimize','windowMaximize','windowClose']:
  assert page.locator('#'+x).is_visible(),x
  page.locator('#'+x).click()
 page.wait_for_timeout(250)
 print('window controls:',page.evaluate('window.__windowActions'))
 print('page errors:',errors[:5])
 assert page.evaluate('window.__windowActions')==['minimize','toggle_maximize','close']
 assert not errors,errors
 browser.close()
