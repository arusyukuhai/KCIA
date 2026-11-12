"""Regression for ordinary HF classification checkpoint used as PPO Critic.

No network: simulate the actual Transformers output_loading_info contract,
including mismatched scalar head and missing backbone guardrails.
"""
import json
from types import SimpleNamespace
import pytest
import torch
from patchlib.hf_value_critic_v69 import HFValueCritic, HFValueError, critic_spec
from patchlib.hf_critic_state_v69 import save,load,snapshot_file
from patchlib.mix_gui_v66 import DEFAULT,normalize,command_options,objective_policy
from test_hf_value_critic_v69 import Tokenizer,ValueModel,settings,SHA


def _patch_load(monkeypatch, *, missing=None, mismatched=None):
    from patchlib import hf_load_v85, hf_disk_preflight_v101
    calls=[]
    def load_model(cls,repo,**kwargs):
        calls.append(dict(kwargs))
        # Classifier stored as 2-labels; new mode must explicitly request 1.
        n=kwargs.get('num_labels',2)
        return ValueModel(n),{'missing_keys': list(missing or []),
                               'mismatched_keys': list(mismatched or [])}
    monkeypatch.setattr(hf_load_v85,'load_hf_model',load_model)
    monkeypatch.setattr(hf_load_v85,'load_hf_tokenizer',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(hf_disk_preflight_v101,'basic_space_check',lambda *a,**kw:None)
    import sys
    # Only class symbols are accessed, network access is mocked above.
    from types import SimpleNamespace as NS
    monkeypatch.setitem(sys.modules,'transformers',NS(AutoModelForSequenceClassification=object,
        AutoModelForTokenClassification=object))
    return calls


def test_default_rejects_generic_two_label_classification_checkpoint(tmp_path,monkeypatch):
    calls=_patch_load(monkeypatch)
    with pytest.raises(HFValueError,match='num_labels=1'):
        HFValueCritic(settings(tmp_path))
    assert calls[0].get('num_labels') is None


def test_fresh_scalar_head_is_explicit_and_trains_and_restores(tmp_path,monkeypatch):
    calls=_patch_load(monkeypatch,mismatched=[('score.weight',(2,4),(1,4))],missing=['score.bias'])
    args=settings(tmp_path,critic_hf_head_init='new')
    critic=HFValueCritic(args)
    assert calls[0]['num_labels']==1 and calls[0]['ignore_mismatched_sizes'] is True
    assert critic.spec['head_init']=='new-scalar-zero-v102'
    assert critic.head.weight.abs().sum().item()==0
    frozen=critic.model.body.weight.detach().clone()
    optim=torch.optim.AdamW(critic.trainable_parameters(),lr=.02)
    v=critic.values(['short','longer'],actor_device='cpu')
    loss=((v-torch.tensor([1.5,-.5]))**2).mean()
    loss.backward()
    assert critic.head.weight.grad is not None
    optim.step()
    assert torch.equal(critic.model.body.weight,frozen)
    assert not torch.equal(critic.head.weight,torch.zeros_like(critic.head.weight))
    save(tmp_path,critic,optim,identity='room',steps=1)
    restored=HFValueCritic(args)
    o2=torch.optim.AdamW(restored.trainable_parameters(),lr=.02)
    assert load(snapshot_file(tmp_path),restored,o2,identity='room')==1
    for a,b in zip(critic.head.parameters(),restored.head.parameters()):
        assert torch.equal(a,b)
    # A different initialization policy is not allowed to reuse the checkpoint.
    plain=HFValueCritic(settings(tmp_path),model=ValueModel(),tokenizer=Tokenizer())
    with pytest.raises(HFValueError,match='mismatch'):
        load(snapshot_file(tmp_path),plain,torch.optim.AdamW(plain.trainable_parameters()),identity='room')


def test_bootstrap_rejects_missing_backbone_tensor(tmp_path,monkeypatch):
    _patch_load(monkeypatch,missing=['body.weight','score.weight'])
    with pytest.raises(HFValueError,match='backbone'):
        HFValueCritic(settings(tmp_path,critic_hf_head_init='new'))


def test_gui_cli_and_policy_pin_new_head_without_changing_legacy(tmp_path):
    row=tmp_path/'ppo.jsonl'
    row.write_text('{"prompt":"abc","reference":"abcd"}\n')
    opts={**DEFAULT,'enabled':True,'ppo_jsonl':str(row),'sft_weight':0.0,
          'critic_kind':'hf','critic_hf_repo':'Org/critic','critic_hf_revision':SHA}
    legacy=normalize(opts,task='causal-lm')
    fresh=normalize({**opts,'critic_hf_head_init':'new'},task='causal-lm')
    assert 'head_init' not in objective_policy(legacy)['critic']['hf']
    assert objective_policy(fresh)['critic']['hf']['head_init']=='new-scalar-zero-v102'
    assert objective_policy(fresh)!=objective_policy(legacy)
    command=command_options(fresh,state_dir=tmp_path/'state')
    assert command[command.index('--critic-hf-head-init')+1]=='new'
    from patchlib.webview_gui import compose_html
    assert 'mixCriticHfHeadInit' in compose_html()
    from pathlib import Path
    assert "critic_hf_head_init:'mixCriticHfHeadInit'" in Path('web/desktop_webview.js').read_text()
    with pytest.raises(ValueError,match='head init'):
        normalize({**opts,'critic_hf_head_init':'anything'},task='causal-lm')


def test_new_head_ppo_rollout_changes_critic_and_survives_resume(tmp_path, monkeypatch):
    from test_hf_value_critic_v69 import Actor
    from patchlib.mixed_learning_v65 import MixedObjectiveTrainer
    _patch_load(monkeypatch, missing=['score.weight','score.bias'])
    (tmp_path/'ppo.jsonl').write_text('{"prompt":"abc","reference":"abcd"}\n')
    args=settings(tmp_path,critic_hf_head_init='new',critic_state_dir=tmp_path/'state')
    actor=Actor()
    # Control the trajectory to ensure at least one verifiable reward match.
    monkeypatch.setattr(torch, 'multinomial', lambda probs, count: torch.full((probs.shape[0],count),2,device=probs.device,dtype=torch.long))
    mix=MixedObjectiveTrainer(args,Tokenizer())
    initial={k:v.clone() for k,v in mix.critic.snapshot_weights().items()}
    actor_loss,value_loss,_=mix.ppo_loss(actor,torch.device('cpu'))
    assert actor_loss.requires_grad and value_loss.requires_grad
    (actor_loss+value_loss).backward()
    assert any(p.grad is not None for p in mix.critic.trainable_parameters())
    mix.critic_optim.step()
    mix.critic_steps+=1
    evolved=mix.critic.snapshot_weights()
    assert any(not torch.equal(initial[k],evolved[k]) for k in initial)
    mix.close()
    mix2=MixedObjectiveTrainer(args,Tokenizer())
    assert mix2.critic_steps==1
    assert all(torch.equal(v,mix2.critic.snapshot_weights()[k]) for k,v in evolved.items())
    mix2.close()
