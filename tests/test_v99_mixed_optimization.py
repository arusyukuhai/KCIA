"""v99 PPO stability and reference-free preference: gradients, flags, signed policy.

CPU-only deterministic regression tests. Not a claim about real-world reward gains.
"""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patchlib.mixed_learning_v65 import (
    MixedObjectiveTrainer, MixedLearningError, _normalize_advantages,
    _response_mean_logp, spec, validate_args)
from patchlib.mix_gui_v66 import DEFAULT, normalize, command_options
from patchlib.cli import add_train_args
import argparse


class ToyToken:
    eos_token_id=None
    def encode(self,text,add_special_tokens=False):
        return [3+(ord(x)%19) for x in text]


class ToyCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(30,16)
        self.lin=nn.Linear(16,30)
    def forward(self,input_ids,attention_mask=None):
        x=self.embed(input_ids).cumsum(1)
        return {'logits':self.lin(x)}


def file(tmp_path,name,row):
    p=tmp_path/name
    p.write_text(json.dumps(row)+'\n',encoding='utf-8')
    return p


def config(tmp_path,**opts):
    d=dict(sft_jsonl=None,ppo_jsonl=None,pt_weight=0.,sft_weight=1.,ppo_weight=.2,
           ppo_clip=.2,ppo_kl_coef=.02,ppo_value_coef=.5,ppo_gamma=.99,
           ppo_gae_lambda=.95,ppo_max_tokens=5,ppo_reuse=3,ppo_reward='token_match',
           ppo_kv_cache='off',max_length=48,hf_task='causal-lm',loss_policy='auto',
           critic_kind='mlp',critic_hidden=32,critic_lr=.001,input_map='',device='cpu')
    d.update(opts)
    return SimpleNamespace(**d)


def test_rms_advantage_keeps_direction_of_one_rollout():
    advantage=torch.tensor([2.,2.,2.,2.])
    assert torch.count_nonzero(_normalize_advantages(advantage,'legacy'))==4 # legacy no centering for zero std
    torch.testing.assert_close(_normalize_advantages(advantage,'rms'),torch.ones(4))
    torch.testing.assert_close(_normalize_advantages(-advantage,'rms'),-torch.ones(4))
    one=torch.tensor([-0.75])
    assert _normalize_advantages(one,'rms').item()==pytest.approx(-1)


def test_preference_math_prefers_chosen():
    ids=torch.tensor([[1,2,3]])
    logits=torch.zeros((1,3,5),requires_grad=True)
    with torch.no_grad():
        logits[0,0,2]=2
        logits[0,1,3]=3
    val=_response_mean_logp(logits,ids,1)
    assert val.requires_grad
    val.backward()
    assert logits.grad[0,1,3]>0


def test_reference_free_preference_trains_and_masks(tmp_path):
    dataset=file(tmp_path,'pairs.jsonl',{'prompt':'Hello','response':'Good','rejected':'Bad'})
    args=config(tmp_path,sft_jsonl=dataset,sft_pref_coef=.5,sft_pref_beta=2.,sft_pref_margin=.3,sft_label_smoothing=.05)
    m=MixedObjectiveTrainer(args,ToyToken())
    model=ToyCausal()
    try:
        loss=m.sft_loss(model,torch.device('cpu'))
        assert loss.ndim==0 and loss.requires_grad and torch.isfinite(loss)
        loss.backward()
        assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
        assert m.counts['sft']==1
    finally:m.close()


def test_preference_missing_rejected_fails_closed(tmp_path):
    dataset=file(tmp_path,'pairs.jsonl',{'prompt':'Hello','response':'Good'})
    m=MixedObjectiveTrainer(config(tmp_path,sft_jsonl=dataset,sft_pref_coef=.4),ToyToken())
    try:
        with pytest.raises(MixedLearningError,match='rejected'):
            m.sft_loss(ToyCausal(),torch.device('cpu'))
    finally:m.close()


def test_nonfinite_and_invalid_advanced_options(tmp_path):
    dataset=file(tmp_path,'pairs.jsonl',{'prompt':'Hi','response':'Yo','rejected':'No'})
    for opts, pattern in [
        ({'sft_pref_coef':float('nan')},'sft_pref_coef'),
        ({'ppo_target_kl':-1},'ppo_target_kl'),
        ({'ppo_kl_horizon':0},'ppo_kl_horizon'),
        ({'ppo_adv_norm':'banana'},'ppo_adv_norm'),
        ({'ppo_adaptive_kl':True},'adaptive KL'),
        ({'sft_label_smoothing':.7},'sft_label_smoothing'),
    ]:
        with pytest.raises(MixedLearningError,match=pattern):
            validate_args(config(tmp_path,sft_jsonl=dataset,**opts))


def test_stale_rollout_recollected_on_small_target_kl(tmp_path):
    p=file(tmp_path,'ppo.jsonl',{'prompt':'Hi','reference':'yes'})
    m=MixedObjectiveTrainer(config(tmp_path,ppo_jsonl=p,ppo_target_kl=.01,
                     ppo_adv_norm='rms',ppo_entropy_coef=.001,ppo_adaptive_kl=True),ToyToken())
    model=ToyCausal()
    try:
        first,c,metrics=m.ppo_loss(model,torch.device('cpu'))
        assert first.requires_grad and c.requires_grad
        assert 'clipfrac' in metrics and 'ratio_mean' in metrics
        assert metrics['kl']<1e-4
        assert m.reuses==1
        old=m.cached
        old['old_lp']-=.6  # moderate, not the extreme fail-closed case
        first2,c2,met2=m.ppo_loss(model,torch.device('cpu'))
        assert m.cached is not old
        assert met2['kl']<1e-4
        assert m.reuses==1
        assert first2.requires_grad and c2.requires_grad
        assert m.kl_coef>0
    finally:m.close()


def test_adaptive_coefficient_moves_with_proxy_kl(tmp_path):
    p=file(tmp_path,'ppo.jsonl',{'prompt':'Hi','reference':'yes'})
    args=config(tmp_path,ppo_jsonl=p,ppo_target_kl=.5,ppo_adaptive_kl=True,ppo_kl_horizon=4)
    m=MixedObjectiveTrainer(args,ToyToken())
    model=ToyCausal()
    try:
        before=m.kl_coef
        m.ppo_loss(model,torch.device('cpu'))
        assert m.kl_coef==before # fresh rollout must not artificially shrink KL
        m.cached['old_lp']-=.1
        m.ppo_loss(model,torch.device('cpu'))
        assert m.kl_coef<before # small but nonzero divergence: below target
    finally:m.close()


def test_signing_compat_and_policy_pins_options(tmp_path):
    dataset=file(tmp_path,'sft.jsonl',{'prompt':'Q','response':'A','rejected':'B'})
    old=config(tmp_path,sft_jsonl=dataset)
    old_policy=spec(old)
    assert 'ppo_stability_v99' not in old_policy
    assert 'sft_ref_free_preference_v99' not in old_policy
    new=config(tmp_path,sft_jsonl=dataset,ppo_target_kl=.02,sft_pref_coef=.3,ppo_adv_norm='rms')
    new_policy=spec(new)
    assert new_policy['ppo_stability_v99']['ppo_target_kl']==.02
    assert new_policy['sft_ref_free_preference_v99']['sft_pref_coef']==.3
    assert str(tmp_path) not in json.dumps(new_policy)
    policyfile=tmp_path/'policy.json';policyfile.write_text(json.dumps(new_policy))
    assert spec(config(tmp_path,sft_jsonl=dataset,mix_policy=policyfile,
                       ppo_target_kl=.02,sft_pref_coef=.3,ppo_adv_norm='rms')) == new_policy
    with pytest.raises(MixedLearningError,match='disagree'):
        spec(config(tmp_path,sft_jsonl=dataset,mix_policy=policyfile,
                    ppo_target_kl=.01,sft_pref_coef=.3,ppo_adv_norm='rms'))


def test_gui_mapping_and_cli_flags(tmp_path):
    sft=file(tmp_path,'sft.jsonl',{'prompt':'Hello','response':'Good','rejected':'Bad'})
    settings={**DEFAULT,'enabled':True,'sft_jsonl':str(sft),'ppo_weight':0,
              'sft_pref_coef':.4,'ppo_adv_norm':'rms','ppo_target_kl':.02,
              'ppo_adaptive_kl':True,'ppo_entropy_coef':.005,'ppo_critic_grad_norm':1.}
    normalized=normalize(settings,task='causal-lm')
    flags=command_options(normalized,state_dir=tmp_path/'critic')
    assert '--ppo-adaptive-kl' in flags
    assert flags[flags.index('--sft-pref-coef')+1]=='0.4'
    assert flags[flags.index('--ppo-target-kl')+1]=='0.02'
    assert flags[flags.index('--ppo-adv-norm')+1]=='rms'
    assert flags[flags.index('--ppo-entropy-coef')+1]=='0.005'
    assert flags[flags.index('--ppo-critic-grad-norm')+1]=='1.0'
    # SFT preference requires per-example rejected column; not supported by HFRowCycle.
    with pytest.raises(ValueError,match='local JSONL'):
        normalize({**settings,'sft_jsonl':'','sft_hf_repo':'org/data','sft_hf_revision':'e'*40},task='causal-lm')


def test_cli_flags_registered():
    parser=argparse.ArgumentParser()
    # add_train_args registers the existing command's model flags, too.
    add_train_args(parser)
    args=parser.parse_args(['--model','toy','--ppo-adv-norm','rms','--sft-pref-coef','.2',
                            '--ppo-target-kl','.01','--ppo-adaptive-kl'])
    assert args.ppo_adaptive_kl is True
    assert args.ppo_adv_norm=='rms'
    assert args.sft_pref_coef==.2


def test_actor_gradient_norm_clipped_only_when_enabled(tmp_path):
    from patchlib.mixed_learning_v65 import clip_actor_grads
    class Adapter:
        def __init__(self):
            self.p=nn.Parameter(torch.ones(5))
            self.p.grad=torch.ones(5)*100
        def trainable_parameters(self):
            return [self.p]
    a=Adapter()
    assert clip_actor_grads(a,SimpleNamespace(mixed_actor_grad_norm=0.)) is None
    assert float(a.p.grad.norm())>10
    before=clip_actor_grads(a,SimpleNamespace(mixed_actor_grad_norm=.5))
    assert float(before)>200
    assert float(a.p.grad.norm())<=.500001
    a.p.grad[0]=float('inf')
    with pytest.raises(RuntimeError):
        clip_actor_grads(a,SimpleNamespace(mixed_actor_grad_norm=1.))


def test_mixed_actor_norm_policy_and_gui(tmp_path):
    sft=file(tmp_path,'sft.jsonl',{'prompt':'X','response':'Y'})
    from patchlib.mix_gui_v66 import objective_policy
    vals=normalize({**DEFAULT,'enabled':True,'sft_jsonl':str(sft),
                    'ppo_weight':0,'mixed_actor_grad_norm':.7},task='causal-lm')
    options=command_options(vals,state_dir=tmp_path/'crit')
    assert options[options.index('--mixed-actor-grad-norm')+1]=='0.7'
    assert objective_policy(vals)['mixed_actor_grad_norm_v99']==.7
