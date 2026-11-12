"""Regression tests: real PPO updates, sparse rewards, MPS-safe IS and policies.

GPU test is skipped on non-Mac environments; CPU tests verify both math and
that the optimizer changes Actor weights, not merely the value head.
"""
import json
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, MixedLearningError, spec
from patchlib.offpolicy_critic_v79 import exact_trajectory_value_surrogate
from patchlib.mix_gui_v66 import DEFAULT, normalize, command_options

class Tok:
    eos_token_id=None
    def encode(self,text,add_special_tokens=False):
        return [3+ord(ch)%47 for ch in text]
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(65+int(i)%26) for i in ids)

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(64,24)
        self.head=nn.Linear(24,64)
    def forward(self,input_ids,attention_mask=None):
        return {'logits':self.head(self.emb(input_ids))}

def args(path,**kw):
    values=dict(sft_jsonl=None,ppo_jsonl=path,pt_weight=0.,sft_weight=0.,ppo_weight=1.,
           ppo_clip=.2,ppo_kl_coef=0.,ppo_value_coef=.5,ppo_gamma=.99,
           ppo_gae_lambda=.95,ppo_max_tokens=3,ppo_reuse=1,ppo_reward='exact',
           ppo_kv_cache='off',ppo_adv_norm='rms',ppo_reference_aux_coef=3.,
           max_length=32,hf_task='causal-lm',loss_policy='auto',critic_kind='linear',
           critic_hidden=32,critic_lr=0.001,input_map='',device='cpu')
    values.update(kw)
    return SimpleNamespace(**values)

def test_sparse_zero_reward_still_trains_actor_with_explicit_guidance(tmp_path):
    torch.manual_seed(42)
    path=tmp_path/'ppo.jsonl'
    # Exact reward is guaranteed zero: target length 6, sample cap 3.
    path.write_text(json.dumps({'prompt':'Q','reference':'AAAAAA'})+'\n')
    conf=args(path)
    model=TinyLM(); trainer=MixedObjectiveTrainer(conf,Tok())
    optim=torch.optim.AdamW(model.parameters(),lr=.035)
    prompt=torch.tensor([[Tok().encode('Q')[0]]])
    ref=torch.tensor(Tok().encode('AAA'))
    def nll():
        ids=torch.cat((prompt,ref[None]),dim=1)
        scores=model(ids)['logits'][0,:-1]
        return float(torch.nn.functional.cross_entropy(scores,ref).detach())
    before=nll()
    try:
        for _ in range(18):
            optim.zero_grad(set_to_none=True)
            actor,critic,metrics=trainer.ppo_loss(model,torch.device('cpu'))
            assert metrics['zero_reward'] and metrics['reward']==0
            assert metrics['reference_nll'] is not None
            actor.backward()
            assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.parameters())
            optim.step()
        assert nll()<before-0.5, f'Actor reference loss did not improve from {before} to {nll()}'
    finally:
        trainer.close()

def test_mps_safe_importance_weights_are_differentiable_and_match_cpu_reference():
    values=torch.tensor([.2,-.3,.5],requires_grad=True)
    old=torch.tensor([-.8,-.7,-.6]);current=old+torch.tensor([.1,-.2,.08])
    loss,metrics=exact_trajectory_value_surrogate(values,old,current,.9,.99)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(values.grad).all()
    assert metrics['trajectory_ratio']>0
    # No normalization clipping or silent detaching of the critic gradient.
    delta=(current-old).double();occ=torch.cat([torch.ones(1,dtype=torch.double),delta.cumsum(0)[:-1].exp()])
    complete=delta.sum().exp();target=(.99**torch.arange(2,-1,-1,dtype=torch.double))*.9
    expected=((occ*values.detach().double()-complete*target)/3).float()
    torch.testing.assert_close(values.grad,expected)

def test_new_reference_setting_pinned_and_legacy_room_unchanged(tmp_path):
    path=tmp_path/'r.jsonl';path.write_text('{"prompt":"Q","reference":"AAAAAA"}\n')
    default=spec(args(path,ppo_reference_aux_coef=0))
    new=spec(args(path,ppo_reference_aux_coef=.3))
    assert 'ppo_reference_aux_coef' not in str(default)
    assert new['ppo_stability_v99']['ppo_reference_aux_coef']==.3
    with pytest.raises(MixedLearningError, match='reference auxiliary'):
        MixedObjectiveTrainer(args(path,ppo_reward='hf_model',ppo_reference_aux_coef=.3),Tok())
    settings={**DEFAULT,'enabled':True,'ppo_jsonl':str(path),'pt_weight':0.,'sft_weight':0.,'ppo_reference_aux_coef':.3}
    data=normalize(settings,task='causal-lm')
    flags=command_options(data,state_dir=tmp_path/'critic')
    assert flags[flags.index('--ppo-reference-aux-coef')+1]=='0.3'

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires real Apple MPS')
def test_real_mps_distributed_critic_forward_backward_and_actor_update(tmp_path):
    torch.manual_seed(7)
    path=tmp_path/'ppo.jsonl';path.write_text('{"prompt":"Q","reference":"AAAAAA"}\n')
    conf=args(path,distributed_critic=True,critic_offpolicy='trajectory_is')
    model=TinyLM().to('mps')
    trainer=MixedObjectiveTrainer(conf,Tok())
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    try:
        before=[p.detach().clone() for p in model.parameters()]
        actor,critic,metrics=trainer.ppo_loss(model,torch.device('mps'))
        assert actor.isfinite().all() and critic.isfinite().all()
        (actor+critic).backward()
        opt.step()
        torch.mps.synchronize()
        assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
    finally:trainer.close()

def test_ppo_only_cli_step_updates_signed_lora_weights_with_zero_reward(tmp_path):
    from patchlib.core import UniversalAdapter
    from patchlib.cli import run_step
    class AdaptedLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=nn.Embedding(64,12)
            self.linear=nn.Linear(12,12)
            self.head=nn.Linear(12,64)
        def forward(self,input_ids,attention_mask=None):
            return {'logits':self.head(torch.tanh(self.linear(self.emb(input_ids))))}
    torch.manual_seed(51)
    path=tmp_path/'ppo.jsonl'
    path.write_text('{"prompt":"Q","reference":"AAAAAA"}\n')
    conf=args(path,ppo_reference_aux_coef=3.,grad_accum=1,telemetry_file=None)
    model=AdaptedLM()
    adapter=UniversalAdapter(model,'toy','revision',kind='lora',rank=2,target_scope='linear')
    optim=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=.02)
    conf._v65_mixer=MixedObjectiveTrainer(conf,Tok())
    prior=[p.detach().clone() for p in adapter.trainable_parameters()]
    try:
        for _ in range(7):
            optim.zero_grad(set_to_none=True)
            loss=run_step(model,adapter,optim,{},conf)
            assert torch.isfinite(torch.tensor(loss))
            assert conf._v65_ppo_metrics['actor_grad_norm']>0
            assert conf._v65_ppo_metrics['zero_reward']
            optim.step()
        assert any(not torch.equal(before,p) for before,p in zip(prior,adapter.trainable_parameters()))
        assert conf._v65_mixer.counts['ppo']==7
    finally:conf._v65_mixer.close()
