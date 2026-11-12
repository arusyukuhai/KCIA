"""Pure/low-state tokenizer, causal-logit and PPO/SFT math helpers.

No dataset/network ownership, optimizer commits or signed protocol mutations.
"""
from __future__ import annotations
import inspect
import math
import torch
from torch import nn
from torch.nn import functional as F
from .mixed_errors_v110 import MixedLearningError
from .model_contract_v64 import invoke_model, _loss_or_logits


def _normalize_advantages(advantages, mode):
    """RMS scaling preserves the sign of a single short rollout's return.

    Zero-centering just 1-8 tokens from ONE trajectory can erase a genuine
    positive policy gradient. Legacy is kept for signed-room compatibility.
    """
    if mode == 'legacy':
        if advantages.numel() > 1 and advantages.std(unbiased=False) > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    elif mode == 'rms':
        advantages = advantages / advantages.square().mean().sqrt().clamp_min(1e-5)
    elif mode != 'none':
        raise MixedLearningError('unsupported advantage normalization mode')
    return advantages.clamp(-10, 10)


def clip_actor_grads(adapter, args):
    """Opt-in global clipping AFTER PT+SFT+PPO gradient accumulation.

    Shared actor weights are updated only when all contributing objectives are
    accumulated, so clipping each loss separately would alter the mixture.
    """
    bound=float(getattr(args,'mixed_actor_grad_norm',0.))
    if bound==0:
        return None
    if not math.isfinite(bound) or not 0 < bound <= 100:
        raise MixedLearningError('mixed_actor_grad_norm must be finite and in (0,100]')
    return torch.nn.utils.clip_grad_norm_(list(adapter.trainable_parameters()),bound,error_if_nonfinite=True)


def _response_mean_logp(logits, ids, prompt_len):
    """Length-normalized log-probability of response tokens only (SimPO)."""
    if logits is None or not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[:2] != ids.shape:
        raise MixedLearningError('SFT preference training requires causal logits [1, seq, vocab]')
    if prompt_len < 1 or prompt_len >= ids.shape[1]:
        raise MixedLearningError('preference completion is empty')
    scores = F.log_softmax(logits[:, prompt_len-1:-1, :].float(), dim=-1)
    targets = ids[:, prompt_len:].unsqueeze(-1)
    return scores.gather(-1, targets).squeeze(-1).mean()


def _tokens(tokenizer, text):
    if not isinstance(text,str) or not text or len(text)>16384:
        raise MixedLearningError('prompt/response/reference must be nonempty UTF-8 text <=16Ki chars')
    # Transformers warns on encoding texts longer than tokenizer.model_max_length
    # *before* KCIA can split them into safe causal windows. Encoding the
    # complete response is intentional; only model forwards are context bound.
    try:
        ids=tokenizer.encode(text, add_special_tokens=False, verbose=False)
    except TypeError as exc:
        if 'verbose' not in str(exc):
            raise
        ids=tokenizer.encode(text, add_special_tokens=False)
    if not ids or not all(isinstance(i,int) and i>=0 for i in ids):
        raise MixedLearningError('tokenizer returned no valid token IDs')
    return list(ids)


def _logits(model, ids, args, *, start=None, length=None):
    batch={'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    outputs=invoke_model(model,batch,input_map=getattr(args,'input_map',None))
    _,logits=_loss_or_logits(outputs)
    if not isinstance(logits,torch.Tensor) or logits.ndim!=3 or logits.shape[:2]!=ids.shape:
        raise MixedLearningError('PPO requires causal logits [1, tokens, vocab]')
    # Slice BEFORE FP32 promotion and finiteness scan. For long contexts,
    # promoting [batch,context,vocab] can temporarily use gigabytes even
    # though PPO needs only the generated token window.
    if start is not None:
        if not isinstance(start,int) or start<0 or not isinstance(length,int) or length<1 or start+length>ids.shape[1]:
            raise MixedLearningError('invalid PPO logits selection')
        logits=logits[:,start:start+length,:]
    if not torch.isfinite(logits).all():
        raise MixedLearningError('nonfinite PPO logits')
    return logits.float()


def _cached_step_logits(model, ids, args, cache_state):
    """Optional causal KV-cache fast path; never changes the PPO update graph.

    A generic PyTorch LM may not implement caching. Cache activation requires
    an explicit forward contract, a real returned cache and no custom input
    remapping. All logits used for actor gradients are still recomputed without
    cache during ppo_loss. We never share caches across policy updates.
    """
    if cache_state['active'] is False:
        return _logits(model, ids, args, start=ids.shape[1]-1, length=1)
    if cache_state['active'] is None:
        if getattr(args,'ppo_kv_cache','auto')=='off' or getattr(args,'input_map',''):
            cache_state['active']=False
            return _logits(model, ids, args, start=ids.shape[1]-1, length=1)
        try:
            params=inspect.signature(model.forward).parameters
        except (ValueError,TypeError):
            params={}
        # Explicit cache ABI, never guess from **kwargs alone.
        if not {'past_key_values','use_cache'}.issubset(params):
            cache_state['active']=False
            return _logits(model, ids, args, start=ids.shape[1]-1, length=1)
        data={'input_ids':ids,'attention_mask':torch.ones_like(ids),'use_cache':True}
    else:
        # Hugging Face causal LMs generally expect the *full prefix* attention
        # mask even though only the newest token is passed as input_ids. A
        # one-token mask can corrupt RoPE/SDPA attention or fail on decode.
        if ids.shape[1] != cache_state.get('last_len', -1)+1:
            raise MixedLearningError('PPO KV cache prefix length changed unexpectedly')
        data={'input_ids':ids[:,-1:],
              'attention_mask':torch.ones_like(ids),
              'past_key_values':cache_state['past'], 'use_cache':True}
    result=model(**data)
    _,logits=_loss_or_logits(result)
    if (not isinstance(logits,torch.Tensor) or logits.ndim!=3 or
            logits.shape[0]!=1 or logits.shape[-1]<2 or logits.shape[1]!=data['input_ids'].shape[1]):
        raise MixedLearningError('PPO cached actor returned invalid logits')
    past=(result.get('past_key_values') if isinstance(result,dict)
          else getattr(result,'past_key_values',None))
    if past is None:
        if cache_state['active']:
            raise MixedLearningError('PPO actor stopped returning a live KV cache')
        cache_state['active']=False
    else:
        cache_state['active']=True
        cache_state['past']=past
        cache_state['last_len']=ids.shape[1]
    last=logits[:,-1:,:]
    if not bool(torch.isfinite(last).all()):
        raise MixedLearningError('nonfinite PPO generation logits')
    return last.float()


def _features(logits, *, offset):
    """Small bounded, detached critic features: no unbounded vocab-size head."""
    detached=logits.detach().float()
    logp=F.log_softmax(detached,dim=-1)
    p=logp.exp()
    entropy=-(p*logp).sum(-1)
    top2=detached.topk(k=min(2,detached.shape[-1]),dim=-1).values
    margin=(top2[...,0]-top2[...,-1]).clamp(0,30)
    normalized=-(logp.max(-1).values).clamp(0,30)
    steps=torch.arange(offset,offset+detached.size(1),device=detached.device,dtype=torch.float32)
    progress=(steps/512).clamp(0,1).unsqueeze(0).expand_as(entropy)
    return torch.stack((entropy.clamp(0,30)/30,margin/30,normalized/30,progress),dim=-1)


class LocalValueHead(nn.Module):
    def __init__(self, kind='mlp', hidden=32):
        super().__init__()
        if kind not in ('mlp','linear') or type(hidden) is not int or not 8 <= hidden <= 512:
            raise MixedLearningError('Critic architecture must be mlp/linear with 8..512 hidden units')
        self.net=(nn.Sequential(nn.Linear(4,hidden),nn.Tanh(),nn.Linear(hidden,1))
                  if kind=='mlp' else nn.Sequential(nn.Linear(4,1)))
    def forward(self,x):
        return self.net(x).squeeze(-1)


def effective_context_length(args, tokenizer, model):
    """Keep mixed-objective token windows within the real model context.

    HF tokenizers may expose a practically infinite sentinel when the limit is
    unknown. Never treat that sentinel as a model context constraint.
    """
    requested=int(getattr(args, 'max_length', 1024))
    limits=[requested]
    token_limit=getattr(tokenizer, 'model_max_length', None)
    if type(token_limit) is int and 2 <= token_limit <= 1_000_000:
        limits.append(token_limit)
    candidates=(model, getattr(model, 'base_model', None), getattr(model, 'model', None))
    for candidate in candidates:
        if candidate is None:
            continue
        config=getattr(candidate, 'config', None)
        if config is None:
            continue
        for key in ('max_position_embeddings', 'n_positions', 'n_ctx'):
            limit=getattr(config,key,None)
            if type(limit) is int and 2 <= limit <= 1_000_000:
                limits.append(limit)
    effective=min(limits)
    if effective < 8:
        raise MixedLearningError('SFT/PPO model context is shorter than 8 tokens')
    return effective


def _sft_window(prompt, response, offset, max_tokens):
    """Return (context, newly supervised tokens, next offset).

    An output token is never dropped or supervised twice. Context is allowed
    to overlap; only the new response suffix contributes to masked causal CE.
    Reserve a prefix token because causal LM predicts token t from token t-1.
    """
    if not 0 <= offset < len(response):
        raise MixedLearningError('invalid SFT response cursor')
    if offset==0 and len(prompt)+len(response)<=max_tokens:
        context=prompt
        chunk=response
    elif offset==0 and len(response)<max_tokens:
        context=prompt[-(max_tokens-len(response)):]
        chunk=response
    else:
        context_len=min(max_tokens-1,max(1,min(256,max_tokens//4)))
        # Build only the bounded causal prefix, even for multi-thousand-token
        # prompts and completions.
        if offset>=context_len:
            context=response[offset-context_len:offset]
        else:
            context=(prompt[-(context_len-offset):]+response[:offset])[-context_len:]
        chunk=response[offset:offset+max_tokens-len(context)]
    if not context or not chunk or len(context)+len(chunk)>max_tokens:
        raise MixedLearningError('SFT window construction failed to preserve supervision')
    return context,chunk,offset+len(chunk)



def move_optimizer_moments_to_device(optimizer):
    """Restore the two AdamW moment tensors to each parameter's device.

    A CPU checkpoint can resume on MPS/CUDA; the optimizer owns its moments,
    never the signed actor protocol. Do not move unrelated state fields.
    """
    for parameter, state in optimizer.state.items():
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state:
                state[key] = state[key].to(parameter.device)
