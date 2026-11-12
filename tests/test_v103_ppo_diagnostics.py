"""PPO doctor is a real Actor update, not just a finite-loss assertion."""
import pytest
import torch
from patchlib.ppo_doctor_v103 import main
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer
from tests.test_ppo_correctness_performance_v73 import args_for_ppo, StrictCausalCache, Tok


def test_ppo_doctor_cpu_isolated_weight_update(capsys):
    assert main(['--device','cpu','--steps','2'])==0
    output=capsys.readouterr().out
    assert 'PPO DOCTOR PASS adapter_weight_change_l1=' in output
    assert 'reward=0' in output
    assert 'not certified' in output


def test_ppo_zero_reward_streak_is_reported(tmp_path,capsys):
    path=tmp_path/'samples.jsonl'
    path.write_text('{"prompt":"abcd","reference":"qqq"}\n')
    config=args_for_ppo(path,ppo_reuse=1,ppo_kv_cache='off',ppo_reference_aux_coef=0)
    model=StrictCausalCache()
    trainer=MixedObjectiveTrainer(config,Tok())
    # Force exactly zero terminal reward with a stub score rather than relying
    # on chance token collisions at arbitrary random seeds.
    original=trainer._new_rollout
    def new_rollout(m,d):
        result=original(m,d)
        result['reward']=0.0
        return result
    trainer._new_rollout=new_rollout
    try:
        for _ in range(10):
            _,_,metrics=trainer.ppo_loss(model,torch.device('cpu'))
        assert metrics['zero_reward_streak']==10
        assert metrics['rollouts']==10
        assert 'consecutive rollouts returned zero' in capsys.readouterr().out
    finally:trainer.close()


def test_ppo_doctor_uses_no_user_journal(tmp_path,monkeypatch):
    # A previous journal on the working dir is neither opened nor overwritten.
    monkeypatch.chdir(tmp_path)
    sentinel=tmp_path/'events.sqlite3'
    sentinel.write_bytes(b'no changes to signed journal')
    assert main(['--steps','1'])==0
    assert sentinel.read_bytes()==b'no changes to signed journal'
