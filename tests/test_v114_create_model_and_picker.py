"""v114: source-preserving logo, real browser create and backend metadata picker tests."""
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from patchlib.hub_discovery_v70 import discover_hub_assets
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import compose_html

ROOT=Path(__file__).resolve().parents[1]

class FakeHub:
    def __init__(self):self.calls=[]
    def list_models(self, **kwargs):
        self.calls.append(kwargs)
        return [NS(id='sample/tiny-gpt',sha='e'*40,downloads=100,pipeline_tag=kwargs.get('pipeline_tag','text-generation'),tags=['text-generation','gpt'],gated=False)]
    def list_datasets(self, **kwargs):
        raise AssertionError('base model picker must request HF models, not datasets')

@pytest.mark.parametrize('task,pipeline',[('causal-lm','text-generation'),('masked-lm','fill-mask'),('seq2seq-lm','text2text-generation'),('sequence-classification','text-classification')])
def test_model_picker_uses_task_specific_filter(task,pipeline):
    api=FakeHub()
    result=discover_hub_assets({'role':'model','task':task,'query':'tiny','scope':'recommended'},api=api)
    assert api.calls==[{'search':'tiny','sort':'downloads','limit':20,'pipeline_tag':pipeline}]
    assert result['candidates'][0]['id']=='sample/tiny-gpt'
    assert result['candidates'][0]['revision']=='e'*40
    assert result['candidates'][0]['requires_inspection']

@pytest.mark.parametrize('task',["", "arbitrary", {}, 123])
def test_model_picker_rejects_invalid_task(task):
    with pytest.raises(ValueError,match='task'):
        discover_hub_assets({'role':'model','task':task},api=FakeHub())


def test_create_selected_pretrained_model_persists_without_vps(tmp_path):
    controller=WebviewController(tmp_path)
    preset=controller.settings()
    assert preset['sources'] and preset['model']
    data={'room':preset['room'],'model':'sample/tiny-gpt','revision':'e'*40,
          'task':'causal-lm','initialization':'pretrained','creation_mode':'weights',
          'sources':preset['sources'],'seed':42,'prevent_duplicate':True,
          'adapter_kind':'loha','adapter_scope':'auto'}
    saved=controller.save(data)
    assert saved['room']==preset['room']
    assert saved['manifest']['model']['id']=='sample/tiny-gpt'
    assert saved['manifest']['model']['revision']=='e'*40
    assert controller._room_path(saved['room']).is_file()


def test_exact_logo_is_bundled_and_inline_and_ui_loaded():
    original=(ROOT/'web/assets/kcia.svg').read_text()
    assert '<title>KCIA</title>' in original
    html=compose_html()
    assert 'class="kcia-logo"' in html
    assert 'id="pickBaseModel"' in html
    assert '<title>KCIA</title>' in html
    assert 'id="newModelStatus"' in html
    assert 'Cannot access \'safe\' before initialization' not in html
    assert 'function safe(title,call)' in html
