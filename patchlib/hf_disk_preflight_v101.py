"""Non-destructive Hugging Face disk checks for PPO's external RM/Critic.

The HF Hub may reconstruct sharded safetensors via Xet using a separate temp
store, then retain the downloaded blob. A model fits in RAM does *not* imply it
fits on disk. Never delete shared HF files or silently change the reward source.
"""
from __future__ import annotations

import errno
import os
import shutil
import sys
import tempfile
from pathlib import Path

GIB = 1024 ** 3
MIN_RESERVE = GIB
WEIGHT_PATTERNS = (
    '*.safetensors', '*.safetensors.index.json', 'config.json',
    'tokenizer*', 'special_tokens_map.json', 'generation_config.json',
    'added_tokens.json', 'vocab.json', 'vocab.txt', 'merges.txt',
    'sentencepiece.bpe.model', '*.tiktoken',
)


class HFCacheSpaceError(RuntimeError):
    """Missing physical free disk for a Hugging Face checkpoint download."""


def is_no_space_error(exc):
    """Recognize native ENOSPC and nested Rust/Xet errors without guessing model incompatibility."""
    visited = set()
    queue = [exc]
    while queue:
        current = queue.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, OSError) and current.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        message = str(current).lower()
        if any(marker in message for marker in (
            'no space left on device', 'os error 28', 'errno 28',
            'disk quota exceeded', 'enospc',
        )):
            return True
        for related in (getattr(current, '__cause__', None),
                        getattr(current, '__context__', None)):
            if related is not None:
                queue.append(related)
    return False


def _hub_paths():
    """Honor resolved Hugging Face Hub defaults, including environment overrides."""
    try:
        from huggingface_hub import constants
        hub = Path(constants.HF_HUB_CACHE).expanduser()
        xet = Path(getattr(constants, 'HF_XET_CACHE', hub.parent / 'xet')).expanduser()
    except ImportError:
        root = Path(os.environ.get('HF_HOME', '~/.cache/huggingface')).expanduser()
        hub = Path(os.environ.get('HF_HUB_CACHE', root / 'hub')).expanduser()
        xet = Path(os.environ.get('HF_XET_CACHE', root / 'xet')).expanduser()
    return {'hub': hub, 'xet': xet, 'temporary': Path(tempfile.gettempdir())}


def _free(path):
    path = Path(path)
    while not path.exists():
        if path == path.parent:
            raise FileNotFoundError(f'No existing parent for cache: {path}')
        path = path.parent
    return shutil.disk_usage(path).free


def _display_bytes(n):
    return f'{n/GIB:.2f} GiB'


def _space_check(label, *, estimated_missing=0, reserve=MIN_RESERVE):
    # Xet reconstruction may need both the completed blob and scratch buffers.
    # 2x is deliberately conservative, not a hard guarantee for every HF version.
    extra = max(0, int(estimated_missing))
    needed = reserve + 2 * extra
    paths = _hub_paths()
    for role, path in paths.items():
        free = _free(path)
        if free < needed:
            raise HFCacheSpaceError(
                f'HF CHECKPOINT DISK SPACE: {label}; {role} volume for {path}: '
                f'free={_display_bytes(free)}, estimated minimum='
                f'{_display_bytes(needed)} (missing checkpoint data '
                f'{_display_bytes(extra)}, Xet/scratch allowance and '
                f'{_display_bytes(reserve)} reserve). '
                'This is physical disk exhaustion, NOT a reward-head/model-type '
                'compatibility error. Stop KCIA; run `df -h ~` and '
                '`hf cache prune --dry-run`; free space or relocate HF_HOME '
                'to a volume with enough capacity before restarting. '
                'KCIA did not delete the shared HF cache or signed journal.'
            )
    return paths


def basic_space_check(label):
    """Offline-safe check also used by direct Reward/Critic constructors."""
    return _space_check(label)


def _missing_size(files):
    total = 0
    unknown = 0
    for item in files:
        if bool(getattr(item, 'is_cached', False)):
            continue
        size = getattr(item, 'file_size', None)
        if type(size) is int and size >= 0:
            total += size
        else:
            unknown += 1
    return total, unknown


def preflight_checkpoint(repo, revision, *, label='HF checkpoint', dry_run=None):
    """Check free space BEFORE loading heavyweight model shards.

    If dry-run metadata is unavailable, check reserve only and let the original
    HF download deliver the real error. This probe never starts a download.
    """
    basic_space_check(f'{label} ({repo}@{revision})')
    if dry_run is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print('KCIA HF PREFLIGHT: huggingface_hub unavailable; size estimate skipped',
                  file=sys.stderr, flush=True)
            return None
        dry_run = snapshot_download
    try:
        files = dry_run(repo, revision=revision, dry_run=True,
                        allow_patterns=list(WEIGHT_PATTERNS), max_workers=1)
    except (Exception,) as exc:
        if is_no_space_error(exc):
            raise HFCacheSpaceError(
                f'HF CHECKPOINT DISK SPACE: {label} {repo}@{revision}: {exc}. '
                'Inspect HF cache and available storage before retrying.') from exc
        # Don't turn a temporary metadata/network failure into a hard offline
        # failure when the full pinned checkpoint could already be cached.
        print(f'KCIA HF PREFLIGHT: metadata unavailable for {repo}@{revision} '
              f'({type(exc).__name__}: {str(exc)[:180]}); using reserve-only check',
              file=sys.stderr, flush=True)
        return None
    if not isinstance(files, (list, tuple)):
        print('KCIA HF PREFLIGHT: dry-run result not a file listing; '
              'using reserve-only check', file=sys.stderr, flush=True)
        return None
    missing, unknown = _missing_size(files)
    _space_check(f'{label} ({repo}@{revision})', estimated_missing=missing)
    print(f'KCIA HF PREFLIGHT: {label} {repo}@{revision}: '
          f'{_display_bytes(missing)} uncached relevant files, '
          f'{unknown} sizes unknown; storage guard passed', flush=True)
    return {'missing_bytes': missing, 'unknown_sizes': unknown}


def preflight_training_resources(args, *, dry_run=None):
    """Explicitly selected secondary models only; never change RLHF objectives."""
    ppo = (bool(getattr(args, 'ppo_jsonl', '')) or
           bool(getattr(args, 'ppo_hf_repo', '')))
    if not ppo or float(getattr(args, 'ppo_weight', 0)) <= 0:
        return []
    selected = []
    if str(getattr(args, 'ppo_reward', 'token_match')) == 'hf_model':
        selected.append(('PPO Reward Model', getattr(args, 'reward_model_id', ''),
                         getattr(args, 'reward_model_revision', '')))
    if str(getattr(args, 'critic_kind', 'mlp')) == 'hf':
        selected.append(('PPO Value Critic', getattr(args, 'critic_hf_repo', ''),
                         getattr(args, 'critic_hf_revision', '')))
    results = []
    seen = set()
    for kind, repo, revision in selected:
        if (repo, revision) in seen:
            continue
        seen.add((repo, revision))
        results.append(preflight_checkpoint(repo, revision, label=kind, dry_run=dry_run))
    return results


def no_space_message(role, exc):
    paths = _hub_paths()
    return (f'{role} Hugging Face download failed: PHYSICAL DISK FULL '
            f'(ENOSPC / Xet file reconstruction): {exc}. '
            f'HF Hub cache: {paths["hub"]}; Xet cache: {paths["xet"]}. '
            'Stop KCIA, run `df -h ~` and `hf cache prune --dry-run`, '
            'or move HF_HOME to a disk with more free space. '
            'Do not delete KCIA signed events/working weights. '
            'Reward/Critic model compatibility is NOT the cause of this exception.')
