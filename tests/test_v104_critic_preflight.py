"""Regression: incompatible pinned HF Critic must not download large weights."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from patchlib.hf_value_critic_v69 import HFValueError, preflight_pretrained_scalar_head
from patchlib.mix_gui_v66 import DEFAULT, normalize, command_options, objective_policy

SHA='a'*40

def opts(**kw):
    d=dict(critic_kind='hf', critic_hf_repo='Org/TwoLabelCritic',
           critic_hf_revision=SHA, critic_hf_loader='sequence_classification',
           critic_hf_device='cpu', critic_hf_max_length=128,
           critic_hf_head_init='pretrained')
    d.update(kw)
    return SimpleNamespace(**d)

def config(tmp_path, data):
    path=tmp_path/'config.json'
    path.write_text(json.dumps(data))
    return path

def test_two_label_config_fails_before_weight_shards(tmp_path):
    cfg=config(tmp_path, {'model_type':'bert', 'id2label':{'0':'NEGATIVE','1':'POSITIVE'}})
    calls=[]
    def fetch(**kw):
        calls.append(kw);return cfg
    with pytest.raises(HFValueError,match=r'PRE-FLIGHT:.*id2label=2.*No HF weight shards'):
        preflight_pretrained_scalar_head(opts(),download_config=fetch)
    assert calls==[{'repo_id':'Org/TwoLabelCritic','revision':SHA,'filename':'config.json'}]

def test_new_head_and_local_critic_never_query_hub(tmp_path):
    def forbidden(**kw):raise AssertionError('should not download')
    assert preflight_pretrained_scalar_head(opts(critic_hf_head_init='new'),download_config=forbidden) is None
    assert preflight_pretrained_scalar_head(opts(critic_kind='mlp'),download_config=forbidden) is None

def test_single_label_or_unstated_labels_not_misdiagnosed(tmp_path):
    a=config(tmp_path, {'id2label':{'0':'VALUE'}})
    assert preflight_pretrained_scalar_head(opts(),download_config=lambda **kw:a)==[('id2label',1)]
    b=config(tmp_path, {'model_type':'bert'})
    assert preflight_pretrained_scalar_head(opts(),download_config=lambda **kw:b)==[]

def test_metadata_network_failure_does_not_block_existing_cached_weights(tmp_path):
    assert preflight_pretrained_scalar_head(opts(),download_config=lambda **kw: (_ for _ in ()).throw(OSError('offline'))) is None

def test_gui_fresh_room_default_is_new_but_legacy_policy_unchanged(tmp_path):
    from patchlib.webview_gui import compose_html
    html=compose_html()
    assert '<option value="new" selected>' in html
    js=Path('web/desktop_webview.js').read_text()
    assert "critic_hf_max_length:512,critic_hf_head_init:'new'" in js
    assert DEFAULT['critic_hf_head_init']=='pretrained'
    row=tmp_path/'ppo.jsonl';row.write_text('{"prompt":"a","reference":"b"}\n')
    d={**DEFAULT,'enabled':True,'ppo_jsonl':str(row),'sft_weight':0.,
       'critic_kind':'hf','critic_hf_repo':'Org/TwoLabelCritic','critic_hf_revision':SHA}
    old=normalize(d,task='causal-lm')
    assert 'head_init' not in objective_policy(old)['critic']['hf']
    new=normalize({**d,'critic_hf_head_init':'new'},task='causal-lm')
    assert objective_policy(new)['critic']['hf']['head_init']=='new-scalar-zero-v102'
    cli=command_options(new,state_dir=tmp_path/'state')
    assert cli[cli.index('--critic-hf-head-init')+1]=='new'

def test_metadata_failure_occurs_before_reward_model_load(monkeypatch,tmp_path):
    from patchlib import mixed_learning_v65 as mixed
    from patchlib import hf_value_critic_v69 as value
    row=tmp_path/'ppo.jsonl';row.write_text('{"prompt":"a","reference":"b"}\n')
    args=opts(ppo_jsonl=row,sft_jsonl=None,ppo_reward='hf_model')
    monkeypatch.setattr(mixed,'validate_args',lambda x:None)
    observed=[]
    def fail_before_large_model(x):
        observed.append('critic-config')
        raise HFValueError('HF Critic PRE-FLIGHT: detected 2 labels')
    monkeypatch.setattr(value,'preflight_pretrained_scalar_head',fail_before_large_model)
    with pytest.raises(HFValueError,match='PRE-FLIGHT'):
        mixed.MixedObjectiveTrainer(args,tokenizer=None)
    assert observed==['critic-config']
