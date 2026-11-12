"""Conservative model-side MPS serialization for macOS.

The RTC event loop remains responsive; only torch model/optimizer work uses the
single MPS worker. CPU verification, networking and disk-only writes stay on
normal workers. Synchronize before yielding the worker to a different task,
including when that task raises; PyTorch/Metal errors otherwise appear in a
later unrelated thread. No global multiprocessing/fork state is changed.
"""
from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

# Allocated lazily; the worker never launches a child process.
_mps_executor: ThreadPoolExecutor | None = None


def execution_device(args):
    """Include independently placed MPS reward/critic components in a step.

    An Actor on CPU with a reward scorer or Value Critic on MPS still issues
    Metal commands in the training step; serialize its full GPU transaction.
    """
    for name in ('device', 'reward_model_device', 'critic_hf_device'):
        value = str(getattr(args, name, 'cpu')).lower()
        if value == 'mps' or value.startswith('mps:'):
            return 'mps'
    selected = getattr(args, 'device', 'cpu')
    if str(selected).lower() == 'auto':
        # AUTO must be resolved *before model construction*. Otherwise the
        # first Metal commands run on asyncio.to_thread and training runs on
        # our dedicated worker, recreating the v91 thread-handoff hazard.
        from .cli import resolve_device
        return resolve_device(selected)
    return selected


def uses_mps(*, device: Any = None, model: Any = None) -> bool:
    if device is not None:
        text = str(device).lower()
        if text == 'mps' or text.startswith('mps:'):
            return True
    if model is not None:
        text = str(getattr(model, '_fedpara_compute_device', '')).lower()
        if text == 'mps' or text.startswith('mps:'):
            return True
        # Most models have many CPU-mapped frozen parameters. A single MPS
        # adapter suffices to require the serial worker.
        try:
            for param in model.parameters():
                if getattr(getattr(param, 'device', None), 'type', None) == 'mps':
                    return True
            for buf in model.buffers():
                if getattr(getattr(buf, 'device', None), 'type', None) == 'mps':
                    return True
        except (AttributeError, TypeError):
            pass
    return False


def synchronize_mps():
    import torch
    if not torch.backends.mps.is_available():
        return
    torch.mps.synchronize()


def _run_synced(task):
    try:
        return task()
    finally:
        # The barrier is a boundary between model operations, NOT a substitute
        # for proper application locks around model/optimizer state mutation.
        synchronize_mps()


async def model_task(func, /, *args, device=None, model=None, **kwargs):
    if not uses_mps(device=device, model=model):
        return await asyncio.to_thread(func, *args, **kwargs)
    global _mps_executor
    if _mps_executor is None:
        _mps_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kagami-mps-model')
    context = contextvars.copy_context()
    task = partial(func, *args, **kwargs)
    loop = asyncio.get_running_loop()
    # Execution is FIFO, and GPU completion is fenced before the next model
    # operation can acquire the worker. Data/RTC tasks are not queued here.
    return await loop.run_in_executor(_mps_executor, context.run, _run_synced, task)
