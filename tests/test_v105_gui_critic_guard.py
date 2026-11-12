"""v105: the GUI must not pin a bad HF PPO protocol before cheap head checks."""
from pathlib import Path
import json
from types import SimpleNamespace
import pytest
from patchlib.mix_gui_v66 import DEFAULT, normalize, preflight_gui_critic
from patchlib.hf_value_critic_v69 import HFValueError, preflight_pretrained_scalar_head
from patchlib.roomid import create_room_id
from patchlib.webview_controller import WebviewController

SHA='a'*40

def mixed(tmp_path, **updates):
    p=tmp_path/'ppo.jsonl'; p.write_text('{"prompt":"hello","reference":"world"}\n')
    d={**DEFAULT, 'enabled':True, 'ppo_jsonl':str(p), 'sft_weight':0.0,
       'critic_kind':'hf','critic_hf_repo':'OpenHands/openhands-critic-4b-v1.0',
       'critic_hf_revision':SHA,'critic_hf_head_init':'pretrained'}
    d.update(updates)
    return normalize(d, task='causal-lm')

def create_gui(tmp_path,monkeypatch):
    c=WebviewController(tmp_path)
    room=create_room_id()
    c.save({'room':room,'model':'org/ourmodel','revision':'main',
            'task':'causal-lm','initialization':'pretrained','seed':42,
            'sources':[{'repo':'org/pretraining','revision':'main','config':None,
                        'split':'train','columns':{'text':'text'},'weight':1.0}]})
    started=[]
    monkeypatch.setattr(c.train,'start',lambda cmd,log: started.append((cmd,log)) or 201)
    return c,room,started

def payload(room,mix):
    return {'room':room,'ws':'ws://127.0.0.1:8080/ws',
            'streaming_base':True,'kind':'loha','rank':16,'mixed_training':mix}

def test_reject_multilabel_before_room_policy_written(tmp_path, monkeypatch):
    c,room,started=create_gui(tmp_path,monkeypatch)
    from patchlib import hf_value_critic_v69 as hf
    calls=[]
    def bad(args):
        calls.append((args.critic_hf_repo,args.critic_hf_head_init))
        raise HFValueError('HF Critic PRE-FLIGHT: pinned checkpoint config declares id2label=27')
    monkeypatch.setattr(hf,'preflight_pretrained_scalar_head',bad)
    with pytest.raises(HFValueError,match='id2label=27.*NEW room'):
        c.start(payload(room,mixed(tmp_path)))
    assert calls==[('OpenHands/openhands-critic-4b-v1.0','pretrained')]
    assert not started
    assert not (c._room_path(room).parent/'mixed-objectives-v66.json').exists()
    assert not (c._room_path(room).parent/'mixed-local-default.json').exists()
    c.close()

def test_mlp_and_explicit_new_head_do_not_run_scalar_preflight(tmp_path,monkeypatch):
    c,room,started=create_gui(tmp_path,monkeypatch)
    from patchlib import hf_value_critic_v69 as hf
    monkeypatch.setattr(hf,'preflight_pretrained_scalar_head',lambda a: (_ for _ in ()).throw(AssertionError('unexpected HF download')))
    r=c.start(payload(room,mixed(tmp_path,critic_kind='mlp')))
    assert r['pid']==201
    assert len(started)==1
    c.close()
    # Separate room to respect the immutable signed PPO objective.
    c2,room2,started2=create_gui(tmp_path/'other',monkeypatch)
    r=c2.start(payload(room2,mixed(tmp_path/'other',critic_hf_head_init='new')))
    assert r['pid']==201
    assert len(started2)==1
    c2.close()

def test_preflight_only_for_relevant_training(tmp_path,monkeypatch):
    from patchlib import hf_value_critic_v69 as hf
    monkeypatch.setattr(hf,'preflight_pretrained_scalar_head',lambda a: (_ for _ in ()).throw(AssertionError('should skip')))
    assert preflight_gui_critic(mixed(tmp_path,ppo_weight=0)) is None
    assert preflight_gui_critic({**mixed(tmp_path), 'ppo_jsonl':''}) is None
    assert preflight_gui_critic(mixed(tmp_path,critic_kind='mlp')) is None
    assert preflight_gui_critic(mixed(tmp_path,critic_hf_head_init='new')) is None
