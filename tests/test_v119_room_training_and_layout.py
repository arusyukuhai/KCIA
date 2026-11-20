"""v119: immutable room datasets, PT fallback, refresh cadence and GUI spacing."""
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from patchlib.room_manifest import make_hf_manifest,validate_manifest,manifest_digest,PatchError
from patchlib.room_training_v119 import canonical_training,for_local_peer,shared_policy
from patchlib.mix_gui_v66 import normalize,objective_policy,atomic_json
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import compose_html

SFT_SHA='a'*40
SOURCES=[{'repo':'roneneldan/TinyStories','revision':'main','config':None,
          'split':'train','columns':{'text':'text'},'weight':1.0}]

def room():
    manifest=make_hf_manifest('roneneldan/TinyStories-1M','main',SOURCES)
    manifest['training']=canonical_training({'enabled':True,'sft_hf_repo':'demo/sft',
                                              'sft_hf_revision':SFT_SHA,'ppo_weight':0.0})
    manifest['optimization']={'reinit_every':4}
    return validate_manifest(manifest)

def test_created_gui_room_pins_public_mixed_intent_and_reinit():
    with TemporaryDirectory() as d:
        c=WebviewController(Path(d))
        saved=c.save({'room':'x'*32,'model':'roneneldan/TinyStories-1M',
             'revision':'main','task':'causal-lm','sources':SOURCES,'seed':42,
             'initialization':'pretrained',
             'mixed_training':{'enabled':True,'sft_hf_repo':'demo/sft',
                               'sft_hf_revision':SFT_SHA,'ppo_weight':0.0},
             'reinit_every':4})
        m=saved['manifest']
        assert m['training']['options']['sft_hf_revision']==SFT_SHA
        assert m['optimization']['reinit_every']==4
        settings=c.mixed_settings({'room':'x'*32})
        assert settings['enabled'] is True
        assert settings['sft_hf_repo']=='demo/sft'
        assert settings['pinned_policy']['objectives']==[True,False]
        assert settings['room_reinit_every']==4

def test_pt_only_peer_preserves_signed_policy_without_private_dataset():
    m=room()
    chosen=for_local_peer(m,{'enabled':False})
    assert chosen['enabled'] is True and not chosen['sft_hf_repo']
    normalized=normalize(chosen,task='causal-lm',allow_pt_only=True)
    with TemporaryDirectory() as d:
        pin=Path(d)/'policy.json'
        atomic_json(pin,shared_policy(m))
        assert objective_policy(normalized,pinned=pin)==shared_policy(m)
    assert for_local_peer(m,{'enabled':True})['sft_hf_repo']=='demo/sft'

def test_rejects_changed_repo_or_private_paths():
    m=room()
    with pytest.raises(ValueError,match='40-character|pin a 40'):
        for_local_peer(m,{'enabled':True,'sft_hf_repo':'other/another'})
    # A pinned, peer-specific HF dataset can implement the same signed SFT
    # objective without replacing the room's immutable shared descriptor.
    different=for_local_peer(m,{'enabled':True,'sft_hf_repo':'other/another',
                                     'sft_hf_revision':'b'*40})
    assert different['sft_hf_repo']=='other/another'
    assert shared_policy(m)['weights']==[1.,1.,0.]
    with pytest.raises(ValueError,match='private'):
        canonical_training({'enabled':True,'sft_jsonl':'/Users/me/.ssh/id.jsonl'})
    malformed=dict(m)
    malformed['training']={'version':1,'options':{'enabled':True,'sft_hf_repo':'other/bad'}}
    with pytest.raises(PatchError,match='invalid room training'):
        validate_manifest(malformed)


def test_legacy_v2_manifest_stays_valid_without_training_fields():
    m=make_hf_manifest('roneneldan/TinyStories-1M','main',SOURCES)
    assert manifest_digest(validate_manifest(m))==manifest_digest(m)
    assert 'optimization' not in m and 'training' not in m


def test_browser_spacing_and_moved_training_panel():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                                  args=['--no-sandbox'])
        for width in (1440,1120,800):
            page=browser.new_page(viewport={'width':width,'height':900})
            page.set_content(compose_html(),wait_until='domcontentloaded')
            assert page.locator('#mixedPanel').evaluate('(x)=>x.parentElement.id')=='objectiveDialogBody'
            page.locator('button[data-tab="training"]').click()
            coords=page.evaluate('''() => {
               const metrics=document.querySelector('.metrics').getBoundingClientRect();
               const charts=document.querySelector('.charts').getBoundingClientRect();
               const plot=[...document.querySelectorAll('.chart-box')].slice(0,2).map(x=>x.getBoundingClientRect().top);
               const toolbar=document.querySelector('.charts-toolbar').getBoundingClientRect();
               return {metricsBottom:metrics.bottom,chartsTop:charts.top,
                       toolbarHeight:toolbar.height,plot};
            }''')
            assert coords['metricsBottom']+8 <= coords['chartsTop'],(width,coords)
            assert coords['toolbarHeight']>=40,(width,coords)
            assert coords['plot'][1]-coords['plot'][0]>=170,(width,coords)
            page.close()
        browser.close()
