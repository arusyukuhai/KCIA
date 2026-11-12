"""v109: real long-supervision windows, without dropping response tokens."""
from types import SimpleNamespace
import json
import pytest
import torch
from torch import nn
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer,effective_context_length

class RecordingTokenizer:
    eos_token_id=2
    model_max_length=2048
    def __init__(self):
        self.asked_verbose=[]
    def encode(self,text,add_special_tokens=False,verbose=True):
        self.asked_verbose.append(verbose)
        return [3+(ord(c)%21) for c in text]

class RecordingModel(nn.Module):
    def __init__(self,cap=2048):
        super().__init__()
        self.config=SimpleNamespace(max_position_embeddings=cap)
        self.emb=nn.Embedding(32,8)
        self.head=nn.Linear(8,32)
        self.seen=[]
    def forward(self,input_ids,attention_mask=None,labels=None):
        assert input_ids.shape[1]<=self.config.max_position_embeddings
        self.seen.append((input_ids.detach().cpu().clone(),
                          None if labels is None else labels.detach().cpu().clone()))
        return {'logits':self.head(self.emb(input_ids))}

def args(tmp_path, **overrides):
    data=dict(sft_jsonl=None,ppo_jsonl=None,pt_weight=0.,sft_weight=1.,ppo_weight=.2,
              ppo_clip=.2,ppo_kl_coef=.02,ppo_value_coef=.5,ppo_gamma=.99,
              ppo_gae_lambda=.95,ppo_max_tokens=3,ppo_reuse=2,ppo_reward='token_match',
              max_length=4096,hf_task='causal-lm',loss_policy='auto',device='cpu',
              grad_accum=1,telemetry_file=None,input_map='',batch_size=1)
    data.update(overrides)
    return SimpleNamespace(**data)

def write(tmp_path,name,row):
    p=tmp_path/name
    p.write_text(json.dumps(row)+'\n',encoding='utf8')
    return p

def test_long_sft_preserves_every_response_token_across_multiple_updates(tmp_path):
    prompt='P'*2590
    completion='r'*3659
    tok=RecordingTokenizer()
    model=RecordingModel()
    path=write(tmp_path,'long.jsonl',{'prompt':prompt,'response':completion})
    mixer=MixedObjectiveTrainer(args(tmp_path,sft_jsonl=path),tok)
    expected=tok.encode(completion,verbose=False)+[tok.eos_token_id]
    actual=[]
    max_steps=5
    for i in range(max_steps):
        loss=mixer.sft_loss(model,torch.device('cpu'))
        assert loss.requires_grad and bool(torch.isfinite(loss))
        loss.backward()
        assert model.emb.weight.grad is not None
        model.zero_grad(set_to_none=True)
        assert len(model.seen[-1][0][0])<=2048
        label=model.seen[-1][1]
        # RecordingModel saw masked labels; all are causal and at least 1 prefix
        assert label is not None and label.shape[1]<=2048
        pos=label[0][label[0]!=-100].tolist()
        assert pos
        actual.extend(pos)
        if mixer._sft_pending is None:
            break
    assert i>0
    assert actual==expected, 'the long response tail must not be lost or repeated'
    assert all(v is False for v in tok.asked_verbose)
    mixer.close()

def test_cap_from_model_and_tokenizer_sentinels(tmp_path):
    model=RecordingModel(cap=96)
    tok=RecordingTokenizer()
    tok.model_max_length=10**30  # Transformers unknown length sentinel
    a=args(tmp_path,max_length=4096)
    assert effective_context_length(a,tok,model)==96
    tok.model_max_length=64
    assert effective_context_length(a,tok,model)==64
    path=write(tmp_path,'tiny_context.jsonl',{'prompt':'a'*90,'response':'b'*150})
    mixer=MixedObjectiveTrainer(args(tmp_path,sft_jsonl=path),tok)
    seen=[]
    for _ in range(5):
        loss=mixer.sft_loss(model,'cpu')
        assert bool(torch.isfinite(loss))
        for token in model.seen[-1][1][0]:
            if token != -100: seen.append(int(token))
        assert model.seen[-1][0].shape[1]<=64
        if mixer._sft_pending is None:break
    assert len(seen)==151
    mixer.close()

def test_ppo_prompt_respects_model_context(tmp_path):
    model=RecordingModel(cap=28)
    tok=RecordingTokenizer();tok.model_max_length=40
    dataset=write(tmp_path,'ppo.jsonl',{'prompt':'q'*105,'reference':'a'})
    a=args(tmp_path,sft_weight=0.,ppo_jsonl=dataset)
    mixer=MixedObjectiveTrainer(a,tok)
    rollout=mixer._new_rollout(model,'cpu')
    assert rollout['ids'].shape[1]<=28
    assert len(rollout['tokens'])>0
    mixer.close()

def test_short_preference_still_works(tmp_path):
    tok=RecordingTokenizer();tok.model_max_length=32
    model=RecordingModel(cap=32)
    dataset=write(tmp_path,'preference.jsonl',{'prompt':'a'*80,'response':'yes', 'rejected':'no'})
    mixer=MixedObjectiveTrainer(args(tmp_path,sft_jsonl=dataset,sft_pref_coef=.4),tok)
    loss=mixer.sft_loss(model,'cpu')
    assert bool(torch.isfinite(loss))
    loss.backward()
    mixer.close()


def test_preference_oversized_fails_without_silent_truncation(tmp_path):
    tok=RecordingTokenizer();tok.model_max_length=32
    model=RecordingModel(cap=32)
    dataset=write(tmp_path,'preference_too_long.jsonl',
                  {'prompt':'hey','response':'z'*150, 'rejected':'no'})
    mixer=MixedObjectiveTrainer(args(tmp_path,sft_jsonl=dataset,sft_pref_coef=.4),tok)
    with pytest.raises(ValueError,match='long pairwise preference'):
        mixer.sft_loss(model,'cpu')
    mixer.close()

def test_long_sft_and_ppo_share_training_steps_without_sequence_overflow(tmp_path):
    """An oversized SFT row must not block the PPO objective in the same step."""
    from patchlib.core import UniversalAdapter
    from patchlib.cli import run_step
    tok=RecordingTokenizer(); tok.model_max_length=48
    model=RecordingModel(cap=48)
    actor=UniversalAdapter(model,'toy','fixed',kind='lora',rank=2,target_scope='linear')
    optim=torch.optim.AdamW(list(actor.trainable_parameters()),lr=.001)
    sft=write(tmp_path,'sft-mixed.jsonl',{'prompt':'p'*100,'response':'c'*140})
    ppo=write(tmp_path,'ppo-mixed.jsonl',{'prompt':'Hi','reference':'x'})
    a=args(tmp_path,sft_jsonl=sft,ppo_jsonl=ppo,grad_accum=1)
    a._v65_mixer=MixedObjectiveTrainer(a,tok)
    for _ in range(4):
        loss=run_step(model,actor,optim,{},a)
        assert torch.isfinite(torch.tensor(loss))
        optim.step();optim.zero_grad(set_to_none=True)
    assert a._v65_mixer.counts['sft']==4
    assert a._v65_mixer.counts['ppo']==4
    assert all(len(ids[0])<=48 for ids,_ in model.seen)
    a._v65_mixer.close()
