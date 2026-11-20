"""Opt-in concurrent-objective PT + response-masked SFT + on-policy clipped PPO.

Each local optimization microstep evaluates independent objectives against the
same current adapter.  P2P continues to distribute ONLY adapter updates;
the small auxiliary value estimator is local and never a signed parameter.
This deliberately uses *verifiable reference text* as the PPO reward source,
not the model's own self-judgment.
"""
from __future__ import annotations


import torch
from torch.nn import functional as F

from .model_contract_v64 import invoke_model, training_loss, _loss_or_logits
from .hf_mixed_datasets_v68 import source_selected, source_spec, HFRowCycle
from .ppo_pipeline_v72 import PrefetchedRows
from .offpolicy_critic_v79 import resolved_mode, exact_trajectory_value_surrogate, OffPolicyCorrectionError
from .objective_metrics_v107 import reference_token_match
from .mixed_errors_v110 import MixedLearningError
from .mixed_policy_v110 import V99_DEFAULTS, _v99_options, enabled, spec, validate_args
from .mixed_kernels_v110 import (
    _normalize_advantages, clip_actor_grads, _response_mean_logp,
    _tokens, _logits, _cached_step_logits, _features, LocalValueHead,
    effective_context_length, _sft_window, move_optimizer_moments_to_device,
)
from .mixed_sources_v110 import JsonlCycle



# v99: New controls are strictly opt-in for an existing signed room. A new
# room using them commits the exact values in its immutable mixed policy.


class MixedObjectiveTrainer:
    def __init__(self,args,tokenizer):
        # A HF reward/critic checkpoint may fail to load *after* dataset
        # prefetch workers have started. Release partial construction eagerly.
        self.sft=None
        self.ppo=None
        self.reward_scorer=None
        try:
            self._initialize(args,tokenizer)
        except BaseException:
            for item in (self.sft,self.ppo):
                if item is not None:
                    try:item.close()
                    except Exception:pass
            if self.reward_scorer is not None:
                try:self.reward_scorer.close()
                except Exception:pass
            raise

    def _initialize(self,args,tokenizer):
        validate_args(args)
        self.args=args
        self.tok=tokenizer
        self._closed=False
        self.sft=(PrefetchedRows(HFRowCycle(source_spec(args,'sft'),'sft')) if getattr(args,'sft_hf_repo','') else
                  JsonlCycle(args.sft_jsonl) if getattr(args,'sft_jsonl',None) else None)
        self.ppo=(PrefetchedRows(HFRowCycle(source_spec(args,'ppo'),'ppo')) if getattr(args,'ppo_hf_repo','') else
                  JsonlCycle(args.ppo_jsonl) if getattr(args,'ppo_jsonl',None) else None)
        # v106: verify real Hub columns and the very first complete row BEFORE
        # loading multi-GiB reward/critic checkpoints. Prefetch.peek() retains
        # the validated sample, so the first training example is not lost.
        for role, source in (('sft', self.sft), ('ppo', self.ppo)):
            if source is not None and getattr(args, role+'_hf_repo', '') and hasattr(source,'peek'):
                try:
                    example=source.peek()
                    if (role=='sft' and float(getattr(args,'sft_pref_coef',0.))>0 and
                            not isinstance(example.get('rejected'),str)):
                        raise ValueError('Preference SFT requires a rejected answer column in the pinned HF dataset')
                    if (role=='ppo' and getattr(args,'ppo_reward','token_match')!='hf_model'
                            and not isinstance(example.get('reference'),str)):
                        raise ValueError('PPO dataset must provide a reference for token_match/exact reward; '
                                         'set PPO reference column, use a dataset with answers, or '
                                         'select an external HF Reward Model')
                except Exception as exc:
                    raise MixedLearningError(f'{role.upper()} Hub dataset preflight failed '
                                             f'before Reward/Critic model loading: {exc}') from exc
                print(f'KCIA HF DATASET PREFLIGHT ready role={role}',flush=True)
        # Validate pinned Critic metadata BEFORE loading a potentially large
        # Reward Model. This avoids multi-minute downloads followed by a
        # trivially detectable 2-label critic failure.
        if (self.ppo and float(getattr(args, 'ppo_weight', .2)) > 0 and
                getattr(args, 'critic_kind', 'mlp')=='hf' and
                getattr(args, 'critic_hf_head_init', 'pretrained')=='pretrained'):
            from .hf_value_critic_v69 import preflight_pretrained_scalar_head
            preflight_pretrained_scalar_head(args)
        self.reward_scorer=None
        self._zero_reward_streak=0
        self._ppo_rollout_count=0
        if self.ppo and getattr(args,'ppo_reward','token_match')=='hf_model':
            print('KCIA PPO INIT phase=reward-model repo=' + str(getattr(args,'reward_model_id','')) + ' device=' + str(getattr(args,'reward_model_device','cpu')),flush=True)
            from .reward_model_v67 import HFRewardScorer
            self.reward_scorer=HFRewardScorer(args)
        self.critic=None
        self.critic_optim=None
        self.cached=None
        self.reuses=0
        self.kl_coef=float(getattr(args,'ppo_kl_coef',.02))
        self.counts={'pt':0,'sft':0,'ppo':0}
        self.last_sft_accuracy=None
        self.last_sft_preference_accuracy=None
        self._sft_pending=None  # tokenized long SFT response, resumed across microsteps
        self._sft_window_count=0
        self._context_cap_logged=None
        self.critic_steps=0
        if self.ppo:
            kind=getattr(args,'critic_kind','mlp')
            hidden=int(getattr(args,'critic_hidden',32))
            if kind=='hf':
                print('KCIA PPO INIT phase=value-critic repo=' + str(getattr(args,'critic_hf_repo','')) + ' device=' + str(getattr(args,'critic_hf_device','cpu')) + ' head=' + str(getattr(args,'critic_hf_head_init','pretrained')),flush=True)
                from .hf_value_critic_v69 import HFValueCritic
                self.critic=HFValueCritic(args)
                critic_params=self.critic.trainable_parameters()
            else:
                self.critic=LocalValueHead(kind,hidden)
                critic_params=list(self.critic.parameters())
            self.critic_optim=torch.optim.AdamW(critic_params,lr=float(getattr(args,'critic_lr',1e-3)))
            print('KCIA PPO INIT phase=ready critic=' + str(kind) + ' reward=' + str(getattr(args,'ppo_reward','token_match')) + ' actor_aux=' + str(float(getattr(args,'ppo_reference_aux_coef',0.))) + ' ppo_weight=' + str(float(getattr(args,'ppo_weight',.2))),flush=True)
            self._critic_dir=getattr(args,'critic_state_dir',None)
            self._critic_identity=str(getattr(args,'room_id',None) or getattr(args,'model','local'))
            if kind=='hf':
                from .hf_critic_state_v69 import snapshot_file, load
                if getattr(args,'critic_init',None):
                    raise MixedLearningError('HF Critic init is the pinned Hub model, not a local v66 checkpoint')
                if (not getattr(args,'_v75_skip_legacy_critic',False) and
                        self._critic_dir and snapshot_file(self._critic_dir).is_file()):
                    self.critic_steps=load(snapshot_file(self._critic_dir),self.critic,self.critic_optim,identity=self._critic_identity)
            else:
                from .critic_state_v66 import snapshot_file, load
                if (not getattr(args,'_v75_skip_legacy_critic',False) and
                        self._critic_dir and snapshot_file(self._critic_dir).is_file()):
                    self.critic_steps=load(snapshot_file(self._critic_dir),self.critic,self.critic_optim,
                                           kind=kind,hidden=hidden,identity=self._critic_identity)
                elif getattr(args,'critic_init',None):
                    load(args.critic_init,self.critic,self.critic_optim,kind=kind,hidden=hidden,
                         identity=self._critic_identity,allow_initial=True)

    def sft_loss(self,model,device):
        cap=effective_context_length(self.args,self.tok,model)
        if cap < int(self.args.max_length) and self._context_cap_logged != cap:
            print('KCIA CONTEXT WINDOW: model/tokenizer limit=' + str(cap) +
                  ' (requested max_length=' + str(self.args.max_length) +
                  '); SFT/PPO inputs will be bounded to the smaller limit',flush=True)
            self._context_cap_logged=cap
        pending=self._sft_pending
        if pending is None:
            row=self.sft.next()
            if not isinstance(row.get('prompt'),str) or not isinstance(row.get('response'),str):
                raise MixedLearningError('SFT row requires prompt and response strings')
            prompt=_tokens(self.tok,row['prompt']);response=_tokens(self.tok,row['response'])
            eos=getattr(self.tok,'eos_token_id',None)
            if isinstance(eos,int) and eos>=0:
                response=response+[eos]
            preference_coef=float(getattr(self.args,'sft_pref_coef',0.))
            if preference_coef and (len(response)>=cap or
                                    not isinstance(row.get('rejected'),str) or not row.get('rejected') or
                                    len(_tokens(self.tok,row['rejected']))+int(isinstance(eos,int) and eos>=0)>=cap):
                # Pairwise SimPO is *sequence-level*; silently clipping either
                # answer or comparing independent slices changes its objective.
                raise MixedLearningError('SFT preference pair needs nonempty rejected and '
                                         'chosen response shorter than model context; '
                                         'long pairwise preference training requires '
                                         'sequence-level chunked log probabilities')
            pending={'prompt':prompt,'response':response,'row':row,'offset':0,
                     'cap':cap,'preference':preference_coef}
            self._sft_pending=pending
        elif pending['cap']!=cap:
            # A new model generation can change context size. Start over on the
            # retained row rather than dropping unfinished reference tokens.
            pending['cap']=cap
        prompt,chunk,next_offset=_sft_window(pending['prompt'],pending['response'],
                                               pending['offset'],cap)
        row=pending['row']
        response=pending['response']
        eos=getattr(self.tok,'eos_token_id',None)
        self._sft_window_count+=1
        if next_offset < len(response) and self._sft_window_count<=3:
            print('KCIA SFT WINDOW: response_tokens=' + str(len(response)) +
                  ' context_limit=' + str(cap) + ' supervised=' +
                  str(next_offset) + '/' + str(len(response)),flush=True)
        pending['offset']=next_offset
        # An optimizer step consumes a single window. This keeps peak GPU
        # activation memory bounded by max_tokens and trains ALL response tokens
        # over successive steps, rather than dropping the tail or crashing.
        if next_offset==len(response):
            self._sft_pending=None
        ids=torch.tensor([prompt+chunk],device=device,dtype=torch.long)
        labels=ids.clone();labels[:,:len(prompt)]=-100
        batch={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels}
        outputs=invoke_model(model,batch,input_map=getattr(self.args,'input_map',None))
        smooth=float(getattr(self.args,'sft_label_smoothing',0.))
        if smooth:
            _,logits=_loss_or_logits(outputs)
            if not isinstance(logits,torch.Tensor) or logits.ndim!=3 or logits.shape[:2]!=ids.shape:
                raise MixedLearningError('SFT label smoothing requires causal logits')
            loss=F.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),
                                 labels[:,1:].reshape(-1), ignore_index=-100,
                                 label_smoothing=smooth)
        else:
            loss=training_loss(outputs,batch,task='causal-lm',policy='causal')
        # Accurate supervised-token top-1, response mask only. Sample at the
        # existing ten-step telemetry cadence to avoid a huge extra argmax
        # over the vocabulary on every microstep (especially on Apple MPS).
        if (getattr(self.args,'telemetry_file',None) and
                getattr(self.args,'_metric_step',0)%10==0):
            try:
                from .objective_metrics_v107 import token_top1
                _,sft_logits=_loss_or_logits(outputs)
                self.last_sft_accuracy=token_top1(sft_logits,labels)
            except (RuntimeError,ValueError,TypeError,AttributeError):
                self.last_sft_accuracy=None
        preference_coef=float(getattr(self.args,'sft_pref_coef',0.))
        if preference_coef:
            if not isinstance(row.get('rejected'),str) or not row['rejected']:
                raise MixedLearningError('SFT preference mode requires nonempty rejected in every row')
            rejected=_tokens(self.tok,row['rejected'])
            if isinstance(eos,int) and eos>=0:
                rejected.append(eos)
            # Use identical prompt context for both alternatives; do not let
            # different truncation boundaries artificially favor one answer.
            if len(rejected)>=cap or len(response)>=cap:
                raise MixedLearningError('preference response exceeds max_length')
            common_prompt=pending['prompt'][-(cap-max(len(rejected),len(response))):]
            # The chosen CE above may include more prompt context if rejected
            # is longer. Recompute chosen logits for the common context only.
            if common_prompt!=prompt:
                chosen_ids=torch.tensor([common_prompt+response],device=device,dtype=torch.long)
                chosen_output=invoke_model(model,{'input_ids':chosen_ids,'attention_mask':torch.ones_like(chosen_ids)},
                                           input_map=getattr(self.args,'input_map',None))
                _,chosen_logits=_loss_or_logits(chosen_output)
            else:
                chosen_ids=ids
                _,chosen_logits=_loss_or_logits(outputs)
            rejected_ids=torch.tensor([common_prompt+rejected],device=device,dtype=torch.long)
            rejected_output=invoke_model(model,{'input_ids':rejected_ids,
                                                'attention_mask':torch.ones_like(rejected_ids)},
                                         input_map=getattr(self.args,'input_map',None))
            _,rejected_logits=_loss_or_logits(rejected_output)
            chosen_lp=_response_mean_logp(chosen_logits,chosen_ids,len(common_prompt))
            rejected_lp=_response_mean_logp(rejected_logits,rejected_ids,len(common_prompt))
            beta=float(getattr(self.args,'sft_pref_beta',2.))
            margin=float(getattr(self.args,'sft_pref_margin',0.))
            # SimPO-style reference-free pairwise objective, anchored to SFT CE.
            preference_loss=F.softplus(-(beta*(chosen_lp-rejected_lp)-margin))
            if (getattr(self.args,'telemetry_file',None) and
                    getattr(self.args,'_metric_step',0)%10==0):
                # Observed fraction of preference pairs ranked correctly.
                # Not a held-out win rate.
                self.last_sft_preference_accuracy=float((chosen_lp.detach()>rejected_lp.detach()).float().mean())
            loss=loss+preference_coef*preference_loss
        if not bool(torch.isfinite(loss.detach()).all()):
            raise MixedLearningError('nonfinite SFT/preference objective')
        self.counts['sft']+=1
        return loss

    def _new_rollout(self,model,device):
        # v101: test/CLI callers may supply 'cpu' rather than torch.device.
        # Sampling must not crash on an ordinary device string.
        device=torch.device(device)
        row=self.ppo.next()
        is_hf=self.reward_scorer is not None
        if not isinstance(row.get('prompt'),str) or (not is_hf and not isinstance(row.get('reference'),str)):
            raise MixedLearningError('PPO row requires prompt and, for ground-truth reward, reference strings')
        prompt=_tokens(self.tok,row['prompt'])
        reference=(_tokens(self.tok,row['reference']) if not is_hf else None)
        cap=effective_context_length(self.args,self.tok,model)
        if len(prompt)+2>cap:
            prompt=prompt[-(cap-2):]
        max_new=min(self.args.ppo_max_tokens,cap-len(prompt))
        if max_new<=0:
            raise MixedLearningError('no room for PPO generation')
        # Snapshot the EXACT old Actor before its first sampling forward.
        # run_step holds adapter.lock, so no concurrent P2P/optimizer write
        # can change a tensor between this capture and rollout generation.
        if getattr(self.args,'critic_rollout_gossip',False):
            from .behavior_checkpoint_v82 import capture
            checkpoint=capture(model)
        else:
            checkpoint=None
        was_training=model.training
        model.eval()
        try:
            with torch.no_grad():
                # Avoid reallocating the complete token prefix on every
                # generation step (quadratic tensor copies for long PPO replies).
                ids_buffer=torch.empty((1,len(prompt)+max_new),dtype=torch.long,device=device)
                ids_buffer[0,:len(prompt)]=torch.tensor(prompt,dtype=torch.long,device=device)
                generated_length=len(prompt)
                chosen=[];old_lp=[];features=[]
                cache_state={'active':None,'past':None}
                eos=getattr(self.tok,'eos_token_id',None)
                for _ in range(max_new):
                    final_logits=_cached_step_logits(model,ids_buffer[:,:generated_length],self.args,cache_state)
                    lp=F.log_softmax(final_logits[:,0],dim=-1)
                    from .mps_acceleration_v108 import sample_next
                    sample,sampling_backend=sample_next(lp,backend=device)
                    token=int(sample.item());chosen.append(token)
                    old_lp.append(lp[0,token])
                    if getattr(self.args,'critic_kind','mlp')!='hf':
                        features.append(_features(final_logits,offset=generated_length-1)[0,0])
                    ids_buffer[0,generated_length]=token
                    generated_length+=1
                    if eos is not None and token==int(eos):
                        break
                if not chosen:
                    raise MixedLearningError('PPO generation returned no tokens')
                feat=(torch.stack(features) if features else None)
                if getattr(self.args,'critic_kind','mlp')!='hf':
                    self.critic.to(device)
                # AdamW moments restored from safe CPU checkpoint must follow
                # their parameter tensors when an accelerator is selected.
                move_optimizer_moments_to_device(self.critic_optim)
                self.critic_optim.zero_grad(set_to_none=True)
                prefixes=None
                if getattr(self.args,'critic_kind','mlp')=='hf':
                    # V(s_t) is conditioned on the prompt plus exactly t sampled tokens.
                    # The Critic uses its own tokenizer; token IDs cannot be shared.
                    prefixes=[self.tok.decode(prompt+chosen[:i],skip_special_tokens=False) for i in range(len(chosen))]
                    old_values=self.critic.values(prefixes,actor_device=device).detach()
                else:
                    old_values=self.critic(feat).detach()
                # Observable, deterministic reference-based reward. Never ask
                # this model to grade its own response.
                if is_hf:
                    completion=self.tok.decode(chosen,skip_special_tokens=True)
                    if not completion:
                        completion=self.tok.decode(chosen,skip_special_tokens=False)
                    reward=self.reward_scorer.score(row['prompt'],completion)
                else:
                    exact=int(chosen==reference)
                    token_matches=sum(a==b for a,b in zip(chosen,reference))
                    partial=token_matches/max(len(chosen),len(reference))
                    reward=float(exact if getattr(self.args,'ppo_reward','token_match')=='exact' else partial)
                T=len(chosen)
                # MPS: compute the tiny serial GAE recurrence on CPU in one
                # bounded transfer rather than launching multiple GPU kernels
                # for every token. Actor/Reward/Critic gradients stay on GPU.
                from .mps_acceleration_v108 import terminal_gae
                adv,returns=terminal_gae(reward,old_values,
                    self.args.ppo_gamma,self.args.ppo_gae_lambda)
                adv=_normalize_advantages(adv,str(getattr(self.args,'ppo_adv_norm','legacy')))
                return {'ids':ids_buffer[:,:generated_length].clone().detach(),'start':len(prompt),
                        'tokens':torch.tensor(chosen,device=device),
                        'old_lp':torch.stack(old_lp).detach(), 'old_values':old_values.detach(),
                        'adv':adv.detach(),'returns':returns.detach(), 'reward':reward,
                        'critic_prefixes':prefixes, 'sampling_backend':sampling_backend, 'reference_tokens':reference,
                        'reference_match':reference_token_match(chosen,reference),
                        'behavior_checkpoint_v82':checkpoint, 'behavior_checkpoint_v81':checkpoint}
        finally:
            model.train(was_training)

    def _reference_auxiliary(self,model,rollout,device):
        """Teacher-forced correction for sparse, verifier-grounded rewards.

        With a large vocabulary token_match/exact often gives zero reward for
        every sampled response. An optional, protocol-pinned reference CE
        supplies a nonzero training target without changing the on-policy
        clipped PPO estimator or inventing an external reward score.
        """
        reference=rollout.get('reference_tokens')
        if not reference:
            raise MixedLearningError('PPO reference auxiliary needs nonempty reference tokens')
        start=rollout['start']
        n=min(len(reference),int(self.args.ppo_max_tokens),
              effective_context_length(self.args,self.tok,model)-start)
        if n<=0:
            raise MixedLearningError('PPO reference auxiliary has no available context')
        tokens=torch.tensor(reference[:n],device=device,dtype=torch.long)
        ids=torch.cat((rollout['ids'][:,:start],tokens[None]),dim=1)
        logits=_logits(model,ids,self.args,start=start-1,length=n)[0]
        guidance=F.cross_entropy(logits,tokens)
        if not bool(torch.isfinite(guidance.detach())):
            raise MixedLearningError('nonfinite PPO reference auxiliary')
        return guidance

    def ppo_loss(self,model,device):
        if self.cached is None or self.reuses>=self.args.ppo_reuse:
            self.cached=self._new_rollout(model,device)
            self.reuses=0
        rollout=self.cached
        ids=rollout['ids'];start=rollout['start'];N=rollout['tokens'].numel()
        if ids.device!=device:
            self.cached=None
            raise MixedLearningError('PPO device changed; rollouts must be recollected')
        was_training=model.training
        model.eval()  # avoid dropout mismatch between rollout and update
        try:
            logits=_logits(model,ids,self.args,start=start-1,length=N)
            lp=F.log_softmax(logits,dim=-1)
            chosen=lp.gather(-1,rollout['tokens'].view(1,N,1)).squeeze(0).squeeze(-1)
            # The actor can change through P2P between optimizer steps. Never
            # optimize a stale trajectory against a drastically changed policy.
            # Recollect ONCE, under the existing adapter lock, not in a retry
            # loop; nonfinite fresh samples remain fatal.
            raw_log_ratio=chosen-rollout['old_lp']
            target_kl=float(getattr(self.args,'ppo_target_kl',0.))
            approx_kl=raw_log_ratio.square().mean()/2
            if not bool(torch.isfinite(raw_log_ratio).all()):
                self.cached=None
                raise MixedLearningError('nonfinite PPO importance ratio; discard rollout')
            if bool((raw_log_ratio.detach().abs()>20).any()):
                self.cached=None
                raise MixedLearningError('PPO policy changed too far for a safe replay; discard stale rollout')
            stale=(target_kl>0 and float(approx_kl.detach())>target_kl)
            if stale:
                if self.reuses<=0:
                    self.cached=None
                    raise MixedLearningError('fresh PPO rollout has inconsistent probabilities')
                self.cached=self._new_rollout(model,device)
                self.reuses=0
                rollout=self.cached
                ids=rollout['ids'];start=rollout['start'];N=rollout['tokens'].numel()
                logits=_logits(model,ids,self.args,start=start-1,length=N)
                lp=F.log_softmax(logits,dim=-1)
                chosen=lp.gather(-1,rollout['tokens'].view(1,N,1)).squeeze(0).squeeze(-1)
                raw_log_ratio=chosen-rollout['old_lp']
                approx_kl=raw_log_ratio.square().mean()/2
                if (not bool(torch.isfinite(raw_log_ratio).all()) or
                        bool((raw_log_ratio.detach().abs()>20).any())):
                    self.cached=None
                    raise MixedLearningError('PPO rollout probability mismatch after recollection')
            ratio=raw_log_ratio.exp()
            advantage=rollout['adv']
            clipped=ratio.clamp(1-self.args.ppo_clip,1+self.args.ppo_clip)
            pg_loss=-torch.minimum(ratio*advantage,clipped*advantage).mean()
            entropy=-(lp.exp()*lp).sum(-1).mean()
            entropy_coef=float(getattr(self.args,'ppo_entropy_coef',0.))
            kl_used=self.kl_coef
            actor_loss=pg_loss+kl_used*approx_kl-entropy_coef*entropy
            reference_coef=float(getattr(self.args,'ppo_reference_aux_coef',0.))
            reference_nll=None
            if reference_coef:
                reference_nll=self._reference_auxiliary(model,rollout,device)
                actor_loss=actor_loss+reference_coef*reference_nll
            # A small *local-only* auxiliary value function: trained for GAE,
            # not mixed into signatures, and not a model-as-judge.
            if getattr(self.args,'critic_kind','mlp')=='hf':
                values=self.critic.values(rollout['critic_prefixes'],actor_device=device)
            else:
                features=_features(logits,offset=start-1)[0]
                values=self.critic(features)
            with torch.no_grad():
                clip_frac=((ratio < 1-self.args.ppo_clip) |
                           (ratio > 1+self.args.ppo_clip)).float().mean()
                metrics={'reward':rollout['reward'],
                         'kl':float(approx_kl.detach()),
                         'entropy':float(entropy.detach()),
                         'clipfrac':float(clip_frac),
                         'ratio_mean':float(ratio.detach().mean()),
                         'adv_abs_mean':float(advantage.detach().abs().mean()),
                         'kl_coef':kl_used,
                         'sampling_backend':rollout.get('sampling_backend','device-native'),
                         'reference_nll':(float(reference_nll.detach()) if reference_nll is not None else None),
                         'zero_reward':rollout['reward']==0.0,
                         # Only available for labeled reference PPO. HF
                         # reward scores are never relabeled as accuracy.
                         'reference_match':rollout.get('reference_match')}
            if resolved_mode(self.args)=='trajectory_is':
                # Exact (unclipped) full-trajectory density-ratio correction
                # for the Critic's terminal-reward Monte Carlo objective.
                # It corrects BOTH state visitation and future action
                # distribution, unlike suffix-only weighted regression.
                # The old policy's chosen-action log probabilities are stored
                # at sampling time and are immutable across PPO reuse.
                try:
                    value_loss, diagnostics=exact_trajectory_value_surrogate(
                        values.flatten(),rollout['old_lp'].flatten(),chosen.flatten(),
                        rollout['reward'],float(self.args.ppo_gamma))
                except OffPolicyCorrectionError as exc:
                    self.cached=None
                    raise MixedLearningError(str(exc)) from exc
                metrics.update(diagnostics)
            else:
                previous=rollout['old_values']
                clipped_values=previous+(values-previous).clamp(-self.args.ppo_clip,self.args.ppo_clip)
                value_loss=.5*torch.maximum((values-rollout['returns']).square(),
                                             (clipped_values-rollout['returns']).square()).mean()
            critic_loss=self.args.ppo_value_coef*value_loss
            if not (bool(torch.isfinite(actor_loss).all()) and
                    bool(torch.isfinite(critic_loss).all())):
                self.cached=None
                raise MixedLearningError('nonfinite PPO objective')
            # First use of a newly sampled trajectory has KL~=0 by definition;
            # it is not evidence that the regularizer is too strong.
            if getattr(self.args,'ppo_adaptive_kl',False) and self.reuses>0:
                target=float(getattr(self.args,'ppo_target_kl'))
                error=max(-.2,min(.2,float(approx_kl.detach())/target-1.))
                horizon=int(getattr(self.args,'ppo_kl_horizon',128))
                self.kl_coef=max(1e-6,min(10.,self.kl_coef*(1+error/horizon)))
            self.reuses+=1;self.counts['ppo']+=1
            # Report the empirical learning signal rather than treating a
            # finite PPO loss as proof that the Actor can improve. In ground-
            # truth modes the generated reward may stay exactly zero for a
            # large vocabulary, yielding a long-lived dead learning signal.
            # Count distinct rollouts, not PPO reuse passes.
            if self.reuses==1:
                self._ppo_rollout_count+=1
                self._zero_reward_streak=(self._zero_reward_streak+1 if metrics['zero_reward'] else 0)
                if (self._zero_reward_streak in (10, 50, 100) and
                        not float(getattr(self.args,'ppo_reference_aux_coef',0.)) and
                        getattr(self.args,'ppo_reward','token_match')!='hf_model'):
                    print('KCIA PPO LEARNING WARNING: ' + str(self._zero_reward_streak) +
                          ' consecutive rollouts returned zero reference reward. '
                          'Actor PPO gradient may not improve the reference task; '
                          'consider SFT or --ppo-reference-aux-coef 0.1 in a NEW room.',flush=True)
            metrics['zero_reward_streak']=self._zero_reward_streak
            metrics['rollouts']=self._ppo_rollout_count
            metrics['critic_loss']=float(critic_loss.detach())
            metrics['actor_loss']=float(actor_loss.detach())
            return actor_loss,critic_loss,metrics
        finally:
            model.train(was_training)

    def export_rollout_v80(self):
        """Capture only source-sampled actions (once per PPO rollout).

        Never export activations/Gradients from the sender's Critic. The peer
        must evaluate its OWN Actor and re-derive its own value gradient.
        """
        from .trajectory_gossip_v80 import validate_rollout
        r=self.cached
        if r is None or r.get('reference_tokens') is None:
            raise MixedLearningError('v80 requires ground-truth reference rollout')
        data={'prompt':r['ids'][0,:r['start']].detach().cpu().tolist(),
              'tokens':r['tokens'].detach().cpu().tolist(),
              'behavior_logp':r['old_lp'].detach().cpu().float().tolist(),
              'reward':float(r['reward']),
              'reference':list(r['reference_tokens']),
              'reward_mode':str(self.args.ppo_reward)}
        validate_rollout(data,max_length=int(self.args.max_length),reward_mode=self.args.ppo_reward)
        return data

    def train_imported_rollout(self,model,device,data):
        """One local Critic-only AdamW step, on an authenticated remote sample.

        Actor never receives gradients. It is the TARGET pi, not the source's
        historic behavior mu, so its current action probabilities are used.
        """
        from .trajectory_gossip_v80 import validate_rollout
        validate_rollout(data,max_length=int(self.args.max_length),reward_mode=self.args.ppo_reward)
        prompt=data['prompt'];tokens=data['tokens'];n=len(tokens)
        # Network may try a different generation or vocab: safe input must
        # satisfy the current Actor's actual embedding indices.
        ids=torch.tensor([prompt+tokens],device=device,dtype=torch.long)
        was_training=model.training
        model.eval()
        try:
            with torch.no_grad():
                logits=_logits(model,ids,self.args,start=len(prompt)-1,length=n)
                target=F.log_softmax(logits,dim=-1).gather(
                    -1,torch.tensor(tokens,device=device).reshape(1,n,1)).reshape(n)
                if not bool(torch.isfinite(target).all()):
                    raise MixedLearningError('nonfinite recipient Actor probability')
                if getattr(self.args,'critic_kind','mlp')!='hf':
                    features=_features(logits,offset=len(prompt)-1)[0]
            if getattr(self.args,'critic_kind','mlp')=='hf':
                prefixes=[self.tok.decode(prompt+tokens[:i],skip_special_tokens=False) for i in range(n)]
                values=self.critic.values(prefixes,actor_device=device).flatten()
            else:
                self.critic.to(device)
                values=self.critic(features).flatten()
            move_optimizer_moments_to_device(self.critic_optim)
            self.critic_optim.zero_grad(set_to_none=True)
            loss,diagnostics=exact_trajectory_value_surrogate(
                values,torch.tensor(data['behavior_logp'],device='cpu',dtype=torch.float64),
                target.detach(),float(data['reward']),float(self.args.ppo_gamma))
            (loss*float(self.args.ppo_value_coef)*float(self.args.ppo_weight)).backward()
            params=list(self.critic.trainable_parameters() if getattr(self.args,'critic_kind','mlp')=='hf'
                        else self.critic.parameters())
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                self.critic_optim.zero_grad(set_to_none=True)
                raise MixedLearningError('nonfinite received Critic gradient')
            bound=float(getattr(self.args,'ppo_critic_grad_norm',0.))
            if bound:
                torch.nn.utils.clip_grad_norm_(list(params),bound,error_if_nonfinite=True)
            self.critic_optim.step()
            self.critic_steps+=1
            self.cached=None;self.reuses=0
            return {'loss':float(loss.detach()),**diagnostics}
        finally:
            model.train(was_training)

    def save_critic(self):
        if (self.critic is not None and getattr(self,'_critic_dir',None) and
                not getattr(self,'_v75_paired',False)):
            if getattr(self.args,'critic_kind','mlp')=='hf':
                from .hf_critic_state_v69 import save
                save(self._critic_dir,self.critic,self.critic_optim,
                     identity=self._critic_identity,steps=self.critic_steps)
            else:
                from .critic_state_v66 import save
                save(self._critic_dir,self.critic,self.critic_optim,
                     kind=getattr(self.args,'critic_kind','mlp'),
                     hidden=int(getattr(self.args,'critic_hidden',32)),
                     identity=self._critic_identity,steps=self.critic_steps)

    def close(self):
        if self._closed:
            return
        self._closed=True
        try:
            if self.critic is not None:
                self.save_critic()
        finally:
            # Even a full disk/permission error must not strand open files
            # or background HF network readers during shutdown.
            try:
                if self.reward_scorer is not None:
                    close=getattr(self.reward_scorer,'close',None)
                    if callable(close):close()
            finally:
                for x in (self.sft,self.ppo):
                    if x:x.close()


def build(args, *, tokenizer=None):
    if not enabled(args):
        return None
    validate_args(args)
    if tokenizer is None:
        from .hf_load_v85 import load_hf_tokenizer
        tokenizer=load_hf_tokenizer(args.model,revision=args.revision,
                            trust_remote_code=bool(getattr(args,'trust_remote_code',False)),
                            local_files_only=bool(getattr(args,'trust_remote_code',False)))
    return MixedObjectiveTrainer(args,tokenizer)
