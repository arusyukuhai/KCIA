"""Kagami v85: consistently recover *auditable* legacy HF configs on all loads.

Transformers AutoTokenizer.from_pretrained() can call AutoConfig internally when
no tokenizer_class is explicitly provided. The old Kagami workaround only
wrapped *model* config loads; thus data/eval/RM/Critic could still raise the
original ``Should have a model_type key in its config.json`` error.

Do not monkeypatch transformers globally, edit pinned model files, or invent
model_type from a repository name: recovery requires a unique built-in class.
"""
from __future__ import annotations

from .hf_config_v83 import load_hf_config, MissingModelTypeError


def is_missing_model_type_error(exc: BaseException) -> bool:
    """Match only the specific HF missing-config-identity failure."""
    return isinstance(exc, ValueError) and 'model_type' in str(exc).lower() and 'config.json' in str(exc).lower()


def _compat_config(repo, revision, local_files_only=False):
    try:
        return load_hf_config(repo, revision=revision,
                              trust_remote_code=False, local_files_only=local_files_only)
    except MissingModelTypeError as exc:
        raise MissingModelTypeError(
            f'{repo}@{revision}: Hugging Face configuration is missing a verifiable '
            'model_type. Check the source config.json and run '
            '`python -m patchlib.cli hf-diagnose --model <model>`; '
            'if it is a custom architecture, provide a trusted model factory '
            'rather than guessing a built-in model_type. Details: '+str(exc)) from exc


def load_hf_tokenizer(repo, *, revision='main', trust_remote_code=False,
                      local_files_only=False, config=None, **kwargs):
    """Use an auditable config, and always identify the failing tokenizer repo.

    The initial attempt is retained to support tokenizer repos whose own
    tokenizer_config.json already identifies their class. If it recursively
    invokes AutoConfig, use the confirmed config for a single controlled retry.
    """
    from transformers import AutoTokenizer
    opts=dict(revision=revision,trust_remote_code=trust_remote_code,
              local_files_only=local_files_only,**kwargs)
    if config is not None:
        try:
            return AutoTokenizer.from_pretrained(repo, config=config, **opts)
        except ValueError as exc:
            if is_missing_model_type_error(exc):
                raise MissingModelTypeError(
                    f'Tokenizer {repo!r}@{revision!r} still called AutoConfig even when '
                    'an explicit verified config was supplied. Check tokenizer_config.json '
                    'and whether tokenizer repo matches the base model. Original: '
                    +str(exc)) from exc
            raise
    try:
        return AutoTokenizer.from_pretrained(repo, **opts)
    except ValueError as exc:
        if not is_missing_model_type_error(exc) or trust_remote_code:
            raise
        config=_compat_config(repo, revision, local_files_only=local_files_only)
        return load_hf_tokenizer(repo, revision=revision,
                   trust_remote_code=trust_remote_code,
                   local_files_only=local_files_only, config=config, **kwargs)

def load_hf_model(model_class, repo, *, revision='main', trust_remote_code=False,
                  local_files_only=False, **kwargs):
    """Recover a legacy config for a model without guessing its architecture."""
    opts=dict(revision=revision,trust_remote_code=trust_remote_code,
              local_files_only=local_files_only,**kwargs)
    try:
        return model_class.from_pretrained(repo, **opts)
    except ValueError as exc:
        if not is_missing_model_type_error(exc) or trust_remote_code:
            raise
        if 'config' in opts:
            raise MissingModelTypeError(
                f'Model {repo!r}@{revision!r} failed inside from_pretrained '
                'despite an explicit config. This is not an inference problem '
                'that Kagami can safely repair; check the exact base model '
                'directory and its config.json. Original: '+str(exc)) from exc
        opts['config']=_compat_config(repo, revision, local_files_only=local_files_only)
        try:
            return model_class.from_pretrained(repo, **opts)
        except ValueError as second:
            if is_missing_model_type_error(second):
                raise MissingModelTypeError(
                    f'Model {repo!r}@{revision!r} failed after verified config recovery. '
                    'Check the exact model directory, not the adapter directory. Original: '
                    +str(second)) from second
            raise
