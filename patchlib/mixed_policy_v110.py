"""Signed room-wide mixed-objective policy and argument validation.

Protocol serialization intentionally preserves the historical v65-v99 exact keys:
changing dictionary keys, float defaults or validation order changes room identity.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from .mixed_errors_v110 import MixedLearningError
from .hf_mixed_datasets_v68 import source_selected, source_spec
from .offpolicy_critic_v79 import resolved_mode, OffPolicyCorrectionError


V99_DEFAULTS = {
    'ppo_adv_norm': 'legacy',
    'ppo_target_kl': 0.0,
    'ppo_adaptive_kl': False,
    'ppo_kl_horizon': 128,
    'ppo_entropy_coef': 0.0,
    'ppo_critic_grad_norm': 0.0,
    'ppo_reference_aux_coef': 0.0,
    'sft_label_smoothing': 0.0,
    'sft_pref_coef': 0.0,
    'sft_pref_beta': 2.0,
    'sft_pref_margin': 0.0,
    'mixed_actor_grad_norm': 0.0,
}


def _v99_options(args):
    return {k: getattr(args, k, v) for k, v in V99_DEFAULTS.items()}


def enabled(args):
    return source_selected(args,'sft') or source_selected(args,'ppo')


def spec(args):
    """No local dataset paths in genesis: peers can use distinct data shards."""
    if not enabled(args) and not getattr(args,'mix_policy',None):
        return None
    generated={'version':'pt-sft-clipped-ppo-v65',
            'objectives':[source_selected(args,'sft'),
                          source_selected(args,'ppo')],
            'weights':[float(getattr(args,'pt_weight',1.)),
                       float(getattr(args,'sft_weight',1.)),
                       float(getattr(args,'ppo_weight',.2))],
            'ppo_clip':float(getattr(args,'ppo_clip',.2)),
            'ppo_kl_coef':float(getattr(args,'ppo_kl_coef',.02)),
            'ppo_value_coef':float(getattr(args,'ppo_value_coef',.5)),
            'ppo_gamma':float(getattr(args,'ppo_gamma',.99)),
            'ppo_gae_lambda':float(getattr(args,'ppo_gae_lambda',.95)),
            'ppo_max_tokens':int(getattr(args,'ppo_max_tokens',8)),
            'ppo_reuse':int(getattr(args,'ppo_reuse',2)),
            'ppo_reward':str(getattr(args,'ppo_reward','token_match')),
            'input_map':str(getattr(args,'input_map','')),
            'reward_source':'ground-truth-reference-v1',
            'task':'causal-lm'}
    if getattr(args,'ppo_reward','token_match')=='hf_model':
        from .reward_model_v67 import reward_spec
        generated['version']='pt-sft-clipped-ppo-v67'
        generated['reward_source']='external-hf-sequence-scorer-v1'
        generated['reward_model']=reward_spec(args)
    if (getattr(args,'critic_kind','mlp')!='mlp' or
            int(getattr(args,'critic_hidden',32))!=32 or
            float(getattr(args,'critic_lr',1e-3))!=.001):
        if generated['version']!='pt-sft-clipped-ppo-v67':
            generated['version']='pt-sft-clipped-ppo-v66'
        generated['critic']={'kind':getattr(args,'critic_kind','mlp'),
                             'hidden':int(getattr(args,'critic_hidden',32)),
                             'lr':float(getattr(args,'critic_lr',1e-3))}
        if getattr(args,'critic_kind','mlp')=='hf':
            from .hf_value_critic_v69 import critic_spec
            generated['version']='pt-sft-clipped-ppo-v69'
            generated['critic']['hf']=critic_spec(args)
    if getattr(args,'distributed_critic',False):
        generated['critic_distribution']='signed-root-critic-gossip-v78'
        generated['version']='pt-sft-clipped-ppo-v78'
    if resolved_mode(args)=='trajectory_is':
        generated['critic_offpolicy']='unclipped-trajectory-is-mc-v79'
        generated['version']='pt-sft-clipped-ppo-v79'
    if getattr(args,'critic_rollout_gossip',False):
        generated['critic_distribution']='loha-complete-checkpoint-verified-trajectory-is-v82'
        generated['behavior_policy_proof']='content-addressed-loha-anchors-remotes-live-factors-frozen-base-sha256-v82'
        generated['version']='pt-sft-clipped-ppo-v82'
    options = _v99_options(args)
    ppo_settings = {k: options[k] for k in options if k.startswith('ppo_') and options[k] != V99_DEFAULTS[k]}
    sft_settings = {k: options[k] for k in options if k.startswith('sft_') and options[k] != V99_DEFAULTS[k]}
    if options['mixed_actor_grad_norm']:
        generated['mixed_actor_grad_norm_v99'] = options['mixed_actor_grad_norm']
    if ppo_settings:
        generated['ppo_stability_v99'] = ppo_settings
    if sft_settings:
        generated['sft_ref_free_preference_v99'] = sft_settings
    if not getattr(args,'mix_policy',None):
        return generated
    path=Path(args.mix_policy)
    if not path.is_file() or path.stat().st_size>8192:
        raise MixedLearningError('--mix-policy must be a JSON file <=8KiB')
    try:
        policy=json.loads(path.read_text(encoding='utf8'))
    except (OSError,ValueError) as exc:
        raise MixedLearningError('--mix-policy is not readable JSON') from exc
    if not isinstance(policy,dict) or set(policy)!=set(generated):
        raise MixedLearningError('--mix-policy needs the exact v65 protocol fields')
    # All numerical/scoring settings are immutable room-wide. Per-peer dataset
    # selection is the ONLY permitted difference and is independently checked.
    if any(policy[k]!=generated[k] for k in generated if k!='objectives'):
        raise MixedLearningError('local objective hyperparameters disagree with --mix-policy')
    flags=policy.get('objectives')
    if not isinstance(flags,list) or len(flags)!=2 or any(type(f) is not bool for f in flags):
        raise MixedLearningError('--mix-policy objectives must be [SFT enabled, PPO enabled] booleans')
    if any(local and not agreed for local,agreed in zip(generated['objectives'],flags)):
        raise MixedLearningError('local dataset is not enabled in the shared --mix-policy')
    if not any(flags) or generated['weights'][0]+sum(w for active,w in zip(flags,generated['weights'][1:]) if active)<=0:
        raise MixedLearningError('--mix-policy contains no active mixed learning objectives')
    return policy


def validate_args(args):
    if not enabled(args) and not getattr(args,'mix_policy',None):
        return
    try:
        resolved_mode(args)
    except OffPolicyCorrectionError as exc:
        raise MixedLearningError(str(exc)) from exc
    if getattr(args,'critic_rollout_gossip',False):
        if not getattr(args,'distributed_critic',False):
            raise MixedLearningError('trajectory gossip requires --distributed-critic')
        if getattr(args,'ppo_reward','token_match') not in ('exact','token_match'):
            raise MixedLearningError('v81 trajectory relay requires receiver-verifiable reference rewards; HF reward not supported')
        if resolved_mode(args)!='trajectory_is':
            raise MixedLearningError('v81 distributed trajectory replay requires un-clipped trajectory_is')
        if int(getattr(args,'max_length',1024))>4096:
            raise MixedLearningError('v81 signed trajectory frame requires max_length <= 4096')
    if getattr(args,'distributed_critic',False):
        if not source_selected(args,'ppo') or float(getattr(args,'ppo_weight',.2))<=0:
            raise MixedLearningError('distributed Critic v78 currently needs a PPO dataset on every Critic participant')
        if getattr(args,'critic_init',None):
            raise MixedLearningError('distributed Critic requires a common pinned initial head; --critic-init is local-only')
        if getattr(args,'unsafe_unsigned',False):
            raise MixedLearningError('distributed Critic requires signed authenticated peers')
    options = _v99_options(args)
    if options['ppo_adv_norm'] not in ('legacy','rms','none'):
        raise MixedLearningError('ppo_adv_norm must be legacy, rms, or none')
    for name, lo, hi in (
        ('ppo_target_kl',0.,10.), ('ppo_entropy_coef',0.,1.),
        ('ppo_critic_grad_norm',0.,100.), ('ppo_reference_aux_coef',0.,10.),
        ('sft_label_smoothing',0.,.3),
        ('sft_pref_coef',0.,10.), ('sft_pref_beta',.001,100.),
        ('sft_pref_margin',0.,10.),
        ('mixed_actor_grad_norm',0.,100.),
    ):
        value=options[name]
        if type(value) not in (float,int) or not math.isfinite(value) or not lo <= value <= hi:
            raise MixedLearningError(f'{name} must be finite and in [{lo},{hi}]')
    if type(options['ppo_adaptive_kl']) is not bool:
        raise MixedLearningError('ppo_adaptive_kl must be boolean')
    if type(options['ppo_kl_horizon']) is not int or not 4 <= options['ppo_kl_horizon'] <= 1000000:
        raise MixedLearningError('ppo_kl_horizon must be 4..1000000')
    if options['ppo_adaptive_kl'] and (options['ppo_target_kl'] <= 0 or
                                       float(getattr(args,'ppo_kl_coef',.02)) <= 0):
        raise MixedLearningError('adaptive KL needs positive target and initial coefficient')
    if options['ppo_reference_aux_coef'] > 0 and getattr(args,'ppo_reward','token_match') == 'hf_model':
        raise MixedLearningError('PPO reference auxiliary requires token_match/exact with reference text')
    if options['ppo_reference_aux_coef'] > 0 and (not source_selected(args,'ppo') or
                                                 float(getattr(args,'ppo_weight',.2)) <= 0):
        raise MixedLearningError('PPO reference auxiliary requires an enabled PPO source and positive weight')
    if options['sft_pref_coef'] > 0 and (not source_selected(args,'sft') or
                                         float(getattr(args,'sft_weight',1.)) <= 0):
        raise MixedLearningError('SFT preference objective needs an enabled SFT source and nonzero weight')
    if getattr(args,'ppo_kv_cache','auto') not in ('auto','off'):
        raise MixedLearningError('ppo_kv_cache must be auto or off')
    if getattr(args,'hf_task','causal-lm') != 'causal-lm':
        raise MixedLearningError('PT/SFT/PPO mixture currently requires --hf-task causal-lm')
    if getattr(args,'loss_policy','auto') not in ('auto','causal'):
        raise MixedLearningError('mixed causal-LM requires --loss-policy auto or causal')
    for field in ('pt_weight','sft_weight','ppo_weight'):
        val=float(getattr(args,field,0))
        if not math.isfinite(val) or val<0 or val>100:
            raise MixedLearningError(f'{field} must be finite and in [0,100]')
    if getattr(args,'ppo_reward','token_match') not in ('token_match','exact','hf_model'):
        raise MixedLearningError('PPO reward must be token_match, exact or hf_model')
    if getattr(args,'ppo_reward','token_match')=='hf_model':
        from .reward_model_v67 import reward_spec
        reward_spec(args)
        if getattr(args,'reward_model_device','cpu') not in ('cpu','cuda','mps'):
            raise MixedLearningError('Invalid reward model device')
    if getattr(args,'critic_kind','mlp') not in ('mlp','linear','hf'):
        raise MixedLearningError('critic_kind must be mlp, linear or hf')
    if getattr(args,'critic_kind','mlp')=='hf':
        from .hf_value_critic_v69 import critic_spec
        critic_spec(args)
    if type(getattr(args,'critic_hidden',32)) is not int or not 8<=getattr(args,'critic_hidden',32)<=512:
        raise MixedLearningError('critic_hidden must be 8..512')
    value_lr=float(getattr(args,'critic_lr',1e-3))
    if not math.isfinite(value_lr) or not 1e-7<=value_lr<=.1:
        raise MixedLearningError('critic_lr out of range')
    if not 0<float(args.ppo_clip)<=.5:
        raise MixedLearningError('ppo_clip must be in (0,.5]')
    for name in ('ppo_kl_coef','ppo_value_coef'):
        val=float(getattr(args,name))
        if not math.isfinite(val) or not 0<=val<=10:
            raise MixedLearningError(f'{name} must be in [0,10]')
    for name in ('ppo_gamma','ppo_gae_lambda'):
        val=float(getattr(args,name))
        if not 0<val<=1:
            raise MixedLearningError(f'{name} must be in (0,1]')
    if not 1<=int(args.ppo_max_tokens)<=128 or not 1<=int(args.ppo_reuse)<=8:
        raise MixedLearningError('PPO token cap must be 1..128, reuse 1..8')
    if not 8<=int(getattr(args,'max_length',128))<=131072:
        raise MixedLearningError('max_length must be >= 8 for mixed causal-LM')
    if not getattr(args,'mix_policy',None) and not source_selected(args,'sft') and getattr(args,'sft_weight',1.) not in (0.,1.):
        raise MixedLearningError('SFT weight supplied without --sft-jsonl')
    if not getattr(args,'mix_policy',None) and not source_selected(args,'ppo') and getattr(args,'ppo_weight',.2) not in (0.,.2):
        raise MixedLearningError('PPO weight supplied without --ppo-jsonl')
    if float(args.pt_weight)+ (float(args.sft_weight) if source_selected(args,'sft') else 0) + (float(args.ppo_weight) if source_selected(args,'ppo') else 0) == 0:
        raise MixedLearningError('at least one mixed objective must have positive weight')
    for name in ('sft_jsonl','ppo_jsonl'):
        path=getattr(args,name,None)
        if path and (not Path(path).is_file() or Path(path).stat().st_size>4*1024**3):
            raise MixedLearningError(f'{name} must be a readable local JSONL <= 4GiB')
    source_spec(args,'sft');source_spec(args,'ppo')
    spec(args)  # validate optional shared room policy before any network join

