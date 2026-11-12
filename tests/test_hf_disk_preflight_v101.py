"""v101: prevent late Xet disk exhaustion while preserving model identity."""
from __future__ import annotations
import errno
from types import SimpleNamespace

import pytest

from patchlib import hf_disk_preflight_v101 as guard


def _info(size, cached=False):
    return SimpleNamespace(file_size=size, is_cached=cached)


def test_enospc_detected_in_nested_rust_xet_runtime_error():
    error=RuntimeError('Task error: File reconstruction error: IO Error: No space left on device (os error 28)')
    assert guard.is_no_space_error(error)
    inner=OSError(errno.ENOSPC, 'disk full')
    outer=RuntimeError('Cannot load checkpoint')
    outer.__cause__=inner
    assert guard.is_no_space_error(outer)
    assert not guard.is_no_space_error(ValueError('unsupported scalar head'))


def test_metadata_dry_run_refuses_before_download_when_disk_cannot_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, '_hub_paths', lambda:{'hub':tmp_path, 'xet':tmp_path,'temporary':tmp_path})
    monkeypatch.setattr(guard, '_free', lambda path: 3*guard.GIB)
    called=[]
    def fake(repo, **kw):
        called.append((repo, kw))
        assert kw['dry_run'] is True
        return [_info(2*guard.GIB), _info(1000,True)]
    with pytest.raises(guard.HFCacheSpaceError,match='PHYSICAL|HF CHECKPOINT DISK SPACE'):
        guard.preflight_checkpoint('some/model', 'f'*40, dry_run=fake)
    assert len(called)==1
    assert 'safetensors' in ' '.join(called[0][1]['allow_patterns'])


def test_cached_checkpoint_does_not_reserve_its_full_size(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, '_hub_paths', lambda:{'hub':tmp_path,'xet':tmp_path,'temporary':tmp_path})
    monkeypatch.setattr(guard, '_free', lambda path: 2*guard.GIB)
    result=guard.preflight_checkpoint('some/model', 'e'*40,
        dry_run=lambda *a, **k:[_info(30*guard.GIB,True),_info(123)])
    assert result['missing_bytes']==123
    assert result['unknown_sizes']==0


def test_preview_unavailable_falls_back_to_non_destructive_disk_reserve(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, '_hub_paths', lambda:{'hub':tmp_path,'xet':tmp_path,'temporary':tmp_path})
    monkeypatch.setattr(guard, '_free', lambda path: 4*guard.GIB)
    result=guard.preflight_checkpoint('some/model','f'*40,
          dry_run=lambda *a,**kw:(_ for _ in ()).throw(TimeoutError('offline')))
    assert result is None


def test_selected_models_only_do_not_fetch_when_no_ppo(monkeypatch):
    calls=[]
    monkeypatch.setattr(guard,'preflight_checkpoint',lambda r,v,**kw:calls.append((r,v)))
    a=SimpleNamespace(ppo_jsonl='',ppo_hf_repo='',ppo_weight=.2,
                      ppo_reward='hf_model',reward_model_id='R/M',reward_model_revision='a'*40,
                      critic_kind='hf',critic_hf_repo='C/M',critic_hf_revision='b'*40)
    assert guard.preflight_training_resources(a)==[]
    a.ppo_jsonl='ppo.jsonl'
    guard.preflight_training_resources(a)
    assert calls==[('R/M','a'*40),('C/M','b'*40)]


def test_reward_model_preserves_enospc_reason(monkeypatch):
    import sys
    from patchlib.reward_model_v67 import HFRewardScorer, RewardModelError
    class FakeAuto:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            raise RuntimeError('File reconstruction error: IO Error: No space left on device (os error 28)')
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoModelForSequenceClassification=FakeAuto,AutoModel=FakeAuto))
    args=SimpleNamespace(ppo_reward='hf_model', reward_model_id='Some/model',
                         reward_model_revision='a'*40, reward_model_device='cpu')
    with pytest.raises(RewardModelError,match='PHYSICAL DISK FULL') as exc:
        HFRewardScorer(args)
    assert 'compatibility is NOT the cause' in str(exc.value)


def test_hf_critic_preserves_enospc_reason(monkeypatch):
    import sys
    from patchlib.hf_value_critic_v69 import HFValueCritic, HFValueError
    class FakeAuto:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            raise RuntimeError('Task error: No space left on device (os error 28)')
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoModelForSequenceClassification=FakeAuto,AutoModelForTokenClassification=FakeAuto))
    args=SimpleNamespace(critic_kind='hf',critic_hf_repo='Some/model',
                         critic_hf_revision='a'*40,critic_hf_device='cpu')
    with pytest.raises(HFValueError,match='PHYSICAL DISK FULL'):
        HFValueCritic(args)
