"""v86: actionable model source identity instead of misleading HF fallback."""
import json
import sys
from types import SimpleNamespace

import pytest

from patchlib.hf_config_v83 import MissingModelTypeError, load_hf_config
from patchlib.hf_compat_v63 import diagnose_hf_checkpoint


@pytest.fixture
def mock_transformers(monkeypatch):
    records=[]
    class AutoConfig:
        @staticmethod
        def from_pretrained(*a,**kw):
            records.append(('from_pretrained',a,kw))
            raise ValueError('Unrecognized model in local-folder. Should have a `model_type` key in its config.json')
        @staticmethod
        def for_model(kind, **kwargs):
            records.append(('for_model',kind,kwargs))
            return SimpleNamespace(model_type=kind,**kwargs)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoConfig=AutoConfig))
    return records


def test_config_without_identity_raises_with_exact_source(tmp_path,mock_transformers):
    p=tmp_path/'unknown-source'
    p.mkdir()
    (p/'config.json').write_text(json.dumps({'hidden_size':64,'vocab_size':101}))
    with pytest.raises(MissingModelTypeError) as ei:
        load_hf_config(p,revision='abc',local_files_only=True)
    msg=str(ei.value)
    assert 'unknown-source' in msg and 'abc' in msg and 'model_type' in msg
    assert 'architectures=None' in msg
    assert not mock_transformers  # no HF name heuristic


def test_legacy_known_architecture_auto_restores(tmp_path,mock_transformers):
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['GPTNeoForCausalLM']}))
    model=load_hf_config(tmp_path,local_files_only=True)
    assert model.model_type=='gpt_neo'
    assert mock_transformers[0][0]=='for_model'


def test_local_adapter_cannot_be_incorrectly_loaded_as_full_model(tmp_path,mock_transformers):
    (tmp_path/'adapter_config.json').write_text('{"base_model_name_or_path":"org/base"}')
    with pytest.raises(MissingModelTypeError) as ei:
        load_hf_config(tmp_path,local_files_only=True)
    assert 'PEFT/LoRA adapter' in str(ei.value)
    assert 'base model' in str(ei.value)
    assert not mock_transformers


def test_diagnosis_separates_model_type_from_hf_task(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'hidden_size': 128}))
    result=diagnose_hf_checkpoint(str(tmp_path),local_files_only=True)
    assert result['architecture_identity_verified'] is False
    assert result['raw_model_type'] is None
    assert result['streaming_candidate'] is False
    assert any('task=causal-lm' in n for n in result['notes'])


def test_hf_diagnose_peft_adapter_without_config(tmp_path):
    (tmp_path/'adapter_config.json').write_text('{"base_model_name_or_path":"company/base"}')
    report=diagnose_hf_checkpoint(str(tmp_path),local_files_only=True)
    assert report['source_kind']=='peft-adapter'
    assert report['architecture_identity_verified'] is False
    assert 'base_model_name_or_path' in report['notes'][0]


def test_tokenizer_second_failure_includes_actual_repo(monkeypatch):
    from patchlib.hf_load_v85 import load_hf_tokenizer
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            raise ValueError('Unrecognized model in unrelated. Should have a `model_type` key in its config.json')
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=AutoTokenizer))
    with pytest.raises(MissingModelTypeError) as ei:
        load_hf_tokenizer('model/tokenizer-only',config=SimpleNamespace(model_type='gpt2'))
    assert 'model/tokenizer-only' in str(ei.value)
