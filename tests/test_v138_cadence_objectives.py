"""v138: WAN model-aware publish policy + pinned multi-objective creation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchlib.auto_policy import estimate_wan_publish_cadence
from patchlib.wan_cadence_v138 import recommend_steps
from patchlib.hf_mixed_datasets_v68 import HFRowCycle, source_spec, HFMixedDatasetError
from patchlib.mix_gui_v66 import DEFAULT, normalize, command_options
from patchlib.room_training_v119 import canonical_training, validate_training, LOCAL_KEYS

SHA='b'*40


def config_model(h, depth, intermediate=None):
    return {'hidden_size':h,'num_hidden_layers':depth,
            'intermediate_size':intermediate or h*4,'vocab_size':32000}


def test_wan_frequency_falls_monotonically_with_model_size_and_patch_bytes():
    configs=[config_model(384,12),config_model(768,24),
             config_model(2048,24),config_model(4096,32),
             config_model(8192,80)]
    plans=[estimate_wan_publish_cadence(config=c) for c in configs]
    steps=[p['local_steps'] for p in plans]
    assert steps==sorted(steps)
    assert steps[0]==32 and steps[-1]>=256 and steps[-1]>steps[1]
    assert all(p['estimated_parameters']>0 for p in plans)
    assert estimate_wan_publish_cadence(config=None)['local_steps']==32
    assert estimate_wan_publish_cadence(config=configs[0],adapter_wire_mib=150)['local_steps']>=32
    slow=estimate_wan_publish_cadence(config=configs[0],adapter_wire_mib=512)
    assert slow['local_steps']>plans[0]['local_steps']
    assert recommend_steps(parameters=500_000_000,baseline=128)>=128
    assert recommend_steps(parameters=500_000_000,observed_patch_bytes=256*1024**2,baseline=128)>=256
    assert recommend_steps(parameters=15_000_000_000,baseline=64)>=recommend_steps(parameters=100_000_000,baseline=64)


def test_sft_only_and_ppo_only_independent_signed_room_objectives():
    sft={'enabled':True,'pt_weight':0,'sft_weight':1,'ppo_weight':0,
         'sft_hf_repo':'org/sft','sft_hf_revision':SHA}
    pinned=canonical_training(sft)
    assert validate_training(pinned)==pinned
    assert pinned['options']['pt_weight']==0
    assert pinned['options']['ppo_weight']==0
    ppo={'enabled':True,'pt_weight':0,'sft_weight':0,'ppo_weight':.2,
         'ppo_hf_repo':'org/ppo','ppo_hf_revision':SHA}
    assert validate_training(canonical_training(ppo))
    with pytest.raises(ValueError):canonical_training({'enabled':True,'pt_weight':0})


def test_pinned_hub_preference_pairs_validate_and_project():
    settings={**DEFAULT,'enabled':True,'pt_weight':0,'ppo_weight':0,
              'sft_hf_repo':'org/pairs','sft_hf_revision':SHA,
              'sft_hf_prompt_column':'query','sft_hf_response_column':'chosen',
              'sft_hf_rejected_column':'loser','sft_pref_coef':.75}
    parsed=normalize(settings,task='causal-lm')
    pinned=canonical_training({k:v for k,v in parsed.items() if k not in LOCAL_KEYS})
    assert validate_training(pinned)==pinned
    assert pinned['options']['sft_pref_coef']==.75
    assert pinned['options']['sft_hf_rejected_column']=='loser'
    flags=command_options(parsed,state_dir=Path('/tmp/kcia-critic-v138'))
    assert flags[flags.index('--sft-hf-rejected-column')+1]=='loser'
    spec=source_spec(SimpleNamespace(**parsed),'sft')
    a=HFRowCycle(spec,'sft',dataset_factory=lambda *a,**kw: iter([{'query':'q','chosen':'yes','loser':'no'}]))
    assert a.peek()=={'prompt':'q','response':'yes','rejected':'no'}
    assert a.next()=={'prompt':'q','response':'yes','rejected':'no'}
    a.close()
    b=HFRowCycle(spec,'sft',dataset_factory=lambda *a,**kw: iter([{'query':'q','chosen':'yes'}]))
    with pytest.raises(HFMixedDatasetError,match='loser'):
        b.peek()  # Missing preference column now rejected before actor/critic load.
    with pytest.raises(ValueError,match='rejected'):
        normalize({**settings,'sft_hf_rejected_column':'query'},task='causal-lm')
    b.close()
    standard=source_spec(SimpleNamespace(**{**parsed,'sft_pref_coef':0}),'sft')
    c=HFRowCycle(standard,'sft',dataset_factory=lambda *a,**kw: iter([{'query':'q','chosen':'yes','loser':{'arbitrary':'metadata'}}]))
    assert c.peek()=={'prompt':'q','response':'yes'}
    c.close()


def test_old_pt_only_policy_remains_legacy_no_mixed_objectives():
    assert canonical_training({'enabled':False}) is None
    assert normalize({'enabled':False},task='causal-lm')['enabled'] is False
