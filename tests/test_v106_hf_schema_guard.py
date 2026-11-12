"""v106: early SFT/PPO HF schema verification and non-destructive mapping."""
from types import SimpleNamespace
from pathlib import Path
import pytest
from patchlib.hf_mixed_datasets_v68 import HFRowCycle, HFMixedDatasetError
from patchlib.ppo_pipeline_v72 import PrefetchedRows
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, MixedLearningError
from test_mixed_learning_v65 import make_args, ToyTokenizer

REV='a'*40

def spec(role='sft',prompt='prompt',other=None):
    label='response' if role=='sft' else 'reference'
    return {'repo':'org/data','revision':REV,'config':None,'split':'train',
            'prompt_column':prompt,label+'_column':other or label}

def factory(rows):
    return lambda *_args,**_kwargs: list(rows)

def test_alias_mapping_sft_and_ppo_and_first_row_not_lost():
    s=HFRowCycle(spec(), 'sft', dataset_factory=factory([{'instruction':'Q','output':'A'}]))
    assert s.peek()=={'prompt':'Q','response':'A'}
    assert s.next()=={'prompt':'Q','response':'A'}
    assert s.next()=={'prompt':'Q','response':'A'}
    assert s._resolved_mapping==('columns',('instruction','output'))
    s.close()
    p=HFRowCycle(spec('ppo'),'ppo',dataset_factory=factory([{'question':'Q','answer':'A'}]))
    assert p.peek()=={'prompt':'Q','reference':'A'}
    assert p.next()=={'prompt':'Q','reference':'A'}
    p.close()

def test_chat_turn_projection_keeps_answer_out_of_prompt():
    row={'messages':[{'role':'system','content':'Rules'},
                     {'role':'user','content':'Hello'},
                     {'role':'assistant','content':'World'}]}
    s=HFRowCycle(spec(),'sft',dataset_factory=factory([row]))
    result=s.peek()
    assert result['response']=='World'
    assert 'World' not in result['prompt']
    assert 'user: Hello' in result['prompt']
    assert s.next()==result
    s.close()

def test_ambiguous_or_explicit_invalid_mapping_fails_with_keys():
    rows=[{'instruction':'a','input':'extra','output':'c'}]
    with pytest.raises(HFMixedDatasetError,match='ambiguous.*Available columns:.*instruction'):
        HFRowCycle(spec(),'sft',dataset_factory=factory(rows)).peek()
    with pytest.raises(HFMixedDatasetError,match='Missing Hub column.*Available columns:'):
        HFRowCycle(spec(prompt='explicit_missing',other='output'),'sft',dataset_factory=factory(rows)).peek()

def test_missing_second_row_still_fails_no_silent_empty_label():
    s=HFRowCycle(spec(),'sft',dataset_factory=factory([
        {'question':'Hello','answer':'World'}, {'question':'bad'}]))
    assert s.peek()['response']=='World'
    assert s.next()['response']=='World'
    with pytest.raises(HFMixedDatasetError,match='Missing Hub column'):
        s.next()
    s.close()

def test_ppo_external_reward_allows_prompt_only_but_ground_truth_does_not(monkeypatch,tmp_path):
    import patchlib.mixed_learning_v65 as mixed
    orig=mixed.HFRowCycle
    monkeypatch.setattr(mixed,'HFRowCycle',lambda spec,role:orig(spec,role,dataset_factory=factory([{'question':'Q'}])))
    args=make_args(tmp_path,ppo_hf_repo='org/data',ppo_hf_revision=REV,
                   ppo_reward='token_match',sft_weight=0.,pt_weight=0.)
    with pytest.raises(MixedLearningError,match='before Reward/Critic model loading.*reference'):
        MixedObjectiveTrainer(args,ToyTokenizer())

def test_bad_sft_dataset_fails_before_reward_and_critic_download(monkeypatch,tmp_path):
    import patchlib.mixed_learning_v65 as mixed
    import patchlib.reward_model_v67 as reward
    original=mixed.HFRowCycle
    monkeypatch.setattr(mixed,'HFRowCycle',lambda spec,role:original(spec,role,dataset_factory=factory([{'wrong':'data'}] if role=='sft' else [{'prompt':'Q'}])))
    called=[]
    monkeypatch.setattr(reward,'HFRewardScorer',lambda args:called.append('reward'))
    args=make_args(tmp_path,sft_hf_repo='org/sft',sft_hf_revision=REV,
                   ppo_hf_repo='org/ppo',ppo_hf_revision=REV,
                   ppo_reward='hf_model',reward_model_id='org/reward',reward_model_revision='b'*40,
                   reward_model_loader='sequence_classification',reward_model_output='logits',
                   reward_model_format='plain',reward_model_index=0,
                   reward_model_scale=1.,reward_model_device='cpu',reward_model_max_length=512)
    with pytest.raises(MixedLearningError,match='SFT Hub dataset preflight failed.*Available columns'):
        MixedObjectiveTrainer(args,ToyTokenizer())
    assert called==[]

def test_prefetch_peek_does_not_consume_first_row_and_errors_preserved():
    class Source:
        i=0
        def next(self):
            self.i+=1
            if self.i>2:raise RuntimeError('boom')
            return {'prompt':str(self.i)}
        def close(self):pass
    p=PrefetchedRows(Source())
    assert p.peek()=={'prompt':'1'}
    assert p.peek()=={'prompt':'1'}
    assert p.next()=={'prompt':'1'}
    assert p.next()=={'prompt':'2'}
    with pytest.raises(RuntimeError,match='boom'):p.next()

def test_gui_probe_reports_only_field_names(monkeypatch,tmp_path):
    from patchlib.webview_controller import WebviewController
    # Call the method without initializing networking/windows; its logic is pure.
    from patchlib.webview_controller import WebviewController
    import patchlib.hf_mixed_datasets_v68 as hf
    orig=hf.HFRowCycle
    monkeypatch.setattr(hf,'HFRowCycle',lambda spec,role:orig(spec,role,dataset_factory=factory([
        {'question':'PRIVATE QUESTION','answer':'PRIVATE ANSWER'}])))
    result=WebviewController.probe_mixed_dataset(object(),{
        'role':'sft','repo':'org/data','revision':REV,'config':'','split':'train',
        'prompt_column':'prompt','response_column':'response'})
    assert result['ready'] and result['prompt_column']=='question'
    assert result['response_column']=='answer'
    assert 'PRIVATE' not in str(result)
    js=Path(__file__).parents[1]/'web/desktop_webview.js'
    assert "mixProbeDataset:'probe_mixed_dataset'" in js.read_text()
    html=Path(__file__).parents[1]/'web/desktop_webview.html'
    assert 'mixSftHfProbe' in html.read_text()
