import os
from types import SimpleNamespace
import pytest
from patchlib import mps_policy_v92 as policy


def test_safe_auto_cpu_on_macos(monkeypatch):
    monkeypatch.setattr(policy.sys,'platform','darwin')
    monkeypatch.delenv('KCIA_EXPERIMENTAL_MPS',raising=False)
    assert policy.safe_auto_device(mps=True)=='cpu'
    assert policy.safe_auto_device(cuda=True,mps=True)=='cuda'


def test_opt_in_mps(monkeypatch):
    monkeypatch.setattr(policy.sys,'platform','darwin')
    monkeypatch.setenv('KCIA_EXPERIMENTAL_MPS','1')
    assert policy.safe_auto_device(mps=True)=='mps'
    policy.check_device('mps')


def test_explicit_mps_rejected_even_when_actor_is_cpu(monkeypatch):
    monkeypatch.setattr(policy.sys,'platform','darwin')
    monkeypatch.delenv('KCIA_EXPERIMENTAL_MPS',raising=False)
    with pytest.raises(RuntimeError,match='KCIA_EXPERIMENTAL_MPS'):
        policy.guard_training_devices(SimpleNamespace(device='cpu',reward_model_device='mps'))
    with pytest.raises(RuntimeError,match='KCIA_EXPERIMENTAL_MPS'):
        policy.guard_training_devices(SimpleNamespace(device='cpu',critic_hf_device='mps'))


def test_non_macos_unchanged(monkeypatch):
    monkeypatch.setattr(policy.sys,'platform','linux')
    monkeypatch.delenv('KCIA_EXPERIMENTAL_MPS',raising=False)
    assert policy.safe_auto_device(mps=True)=='mps'
    policy.check_device('mps')


def test_training_auto_device_resolves_to_safe_cpu(monkeypatch):
    from patchlib import mps_runtime_v91
    from patchlib import cli
    monkeypatch.setattr(policy.sys,'platform','darwin')
    monkeypatch.delenv('KCIA_EXPERIMENTAL_MPS',raising=False)
    monkeypatch.setattr(cli.torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(cli.torch.backends.mps,'is_available',lambda:True)
    assert str(cli.resolve_device('auto'))=='cpu'
    assert str(mps_runtime_v91.execution_device(SimpleNamespace(device='auto'))) == 'cpu'
    with pytest.raises(RuntimeError,match='disabled'):
        cli.resolve_device('mps')


def test_mps_opt_in_gets_dedicated_worker_for_auto(monkeypatch):
    from patchlib import mps_runtime_v91
    from patchlib import cli
    monkeypatch.setattr(policy.sys,'platform','darwin')
    monkeypatch.setenv('KCIA_EXPERIMENTAL_MPS','1')
    monkeypatch.setattr(cli.torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(cli.torch.backends.mps,'is_available',lambda:True)
    assert str(mps_runtime_v91.execution_device(SimpleNamespace(device='auto'))) == 'mps'


def test_gpu_probe_uses_separate_child_process(monkeypatch,capsys):
    from patchlib import mps_smoke_v92
    from types import SimpleNamespace
    recorded=[]
    def fake_run(cmd,**kwargs):
        recorded.append((cmd,kwargs))
        return SimpleNamespace(returncode=-6,stdout='Metal assertion simulated\\n')
    monkeypatch.setattr(mps_smoke_v92.subprocess,'run',fake_run)
    rc=mps_smoke_v92.main()
    assert rc==-6
    assert recorded[0][0][:2]==[mps_smoke_v92.sys.executable,'-c']
    assert recorded[0][1]['env']['KCIA_EXPERIMENTAL_MPS']=='1'
    assert 'MPS PROBE FAILED' in capsys.readouterr().out
