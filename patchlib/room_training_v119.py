"""Portable SFT/PPO room intent; no private paths or executable content on signaling.

Training intent is part of the immutable room manifest. Local peers may opt out of
SFT/PPO and use the SAME signed objective policy with only PT minibatches.
"""
from __future__ import annotations
from .mix_gui_v66 import DEFAULT, FIELDS, normalize, objective_policy

# Only public HF references and signed objective hyperparameters may be shared.
# Never persist per-device CUDA/CPU settings, local JSONL or private checkpoints.
LOCAL_KEYS={'sft_jsonl','ppo_jsonl','critic_init','critic_hf_device',
            'reward_model_device','reward_model_idle_offload'}

def canonical_training(options, *, task='causal-lm'):
    if options is None or options == {'enabled':False} or (isinstance(options,dict) and options.get('enabled') is False):
        return None
    if not isinstance(options,dict):
        raise ValueError('room training settings must be an object')
    if any(k in LOCAL_KEYS for k in options):
        raise ValueError('room SFT/PPO must use public revision-pinned Hub sources, not private files/device paths')
    if set(options)-set(FIELDS):
        raise ValueError('unknown shared training setting')
    data=normalize({'enabled':True,**options},task=task)
    if data['sft_jsonl'] or data['ppo_jsonl'] or data['critic_init']:
        raise ValueError('private files are forbidden in room training settings')
    if not data['sft_hf_repo'] and not data['ppo_hf_repo']:
        raise ValueError('select a revision-pinned SFT or PPO dataset for shared training')
    compact={k:data[k] for k in sorted(data) if k not in LOCAL_KEYS and data[k]!=DEFAULT[k]}
    compact['enabled']=True
    return {'version':1,'options':compact}

def validate_training(value, *, task='causal-lm'):
    if not isinstance(value,dict) or set(value)!={'version','options'} or type(value['version']) is not int or value['version']!=1:
        raise ValueError('invalid room training descriptor')
    actual=canonical_training(value['options'],task=task)
    if actual!=value:
        raise ValueError('room training descriptor must use canonical immutable options')
    return value

def shared_options(manifest):
    from .room_manifest import validate_manifest
    # Manifest has already been validated at entry points. Avoid recursion.
    training=manifest.get('training')
    if not training:return None
    return {**DEFAULT,**training['options']}

def shared_policy(manifest):
    source=shared_options(manifest)
    if source is None:return None
    return objective_policy(source)

def for_local_peer(manifest, chosen):
    """Retain signed room-wide scoring while permitting peer-private input data.

    Each role must already exist in the room policy. Data repository IDs do
    not enter the signed optimizer/objective genesis; no examples are shared.
    """
    shared=shared_options(manifest)
    if shared is None:
        return chosen
    choice=chosen if isinstance(chosen,dict) else {}
    result=dict(shared)
    local_enabled=choice.get('enabled',True)
    if type(local_enabled) is not bool:
        raise ValueError('mixed enabled must be boolean')
    for key in ('critic_hf_device','reward_model_device','reward_model_idle_offload'):
        if key in choice:result[key]=choice[key]
    for role in ('sft','ppo'):
        if not local_enabled or (choice.get(role+'_hf_repo')=='' and not choice.get(role+'_jsonl')):
            for k in result:
                if k.startswith(role+'_hf_'):result[k]=DEFAULT[k]
            result[role+'_jsonl']=''
        elif choice.get(role+'_jsonl'):
            # Offline local datasets are never uploaded, but can implement a
            # shared objective on this peer, provided the role is permitted.
            if not shared.get(role+'_hf_repo'):
                raise ValueError('local dataset objective not offered by this room')
            for k in result:
                if k.startswith(role+'_hf_'):result[k]=DEFAULT[k]
            result[role+'_jsonl']=choice[role+'_jsonl']
        else:
            if choice.get(role+'_hf_repo') and not shared.get(role+'_hf_repo'):
                raise ValueError('dataset objective not offered by this room')
            # Heterogeneous data is the point of federated learning. Keep the
            # exact signed objective weights/scoring policy, while each peer
            # can select its own revision-pinned HF source for permitted role.
            # This changes ONLY input examples, not shared genesis or scores.
            if choice.get(role+'_hf_repo'):
                if (choice[role+'_hf_repo'] != shared[role+'_hf_repo'] and
                        not choice.get(role+'_hf_revision')):
                    raise ValueError('changing dataset repository requires its own pinned 40-character revision')
                for key in result:
                    if key.startswith(role+'_hf_') and key in choice:
                        result[key]=choice[key]
                from types import SimpleNamespace
                from .hf_mixed_datasets_v68 import source_spec
                source_spec(SimpleNamespace(**result),role)
    result['enabled']=True  # signed policy exists even for a PT-only participant
    return result
