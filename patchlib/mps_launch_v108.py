"""Per-peer GPU opt-in. Not part of any signed model/optimizer policy."""
from __future__ import annotations
import os

_DEVICE_FLAGS = ('--device','--critic-hf-device','--reward-model-device')


def selected_devices(command):
    devices={}
    for index,argument in enumerate(command[:-1]):
        if argument in _DEVICE_FLAGS:
            devices[argument]=command[index+1]
    return devices


def needs_mps(command):
    return any(value=='mps' for value in selected_devices(command).values())


def launch_environment(command,env=None):
    env=dict(os.environ if env is None else env)
    if needs_mps(command):
        # Never enable GPU just because the system has one: the user explicitly
        # chose an MPS component in the local (non-signed) GUI controls.
        env['KAGAMI_EXPERIMENTAL_MPS']='1'
        # Must be set *before* the child imports torch. Unsupported kernels
        # may run on CPU while native MPS matmuls/backward stay GPU-backed.
        # PyTorch cannot use this fallback for float64; v100 CPU FP64 path
        # remains mandatory. Users may explicitly set 0 to forbid slow ops.
        env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
    return env


def validate_devices(device, mixed=None):
    if device not in ('auto','cpu','cuda','mps'):
        raise ValueError('Device must be auto/cpu/cuda/mps')
    mixed=mixed or {}
    required=[device]
    if mixed.get('enabled'):
        if mixed.get('ppo_reward')=='hf_model':
            required.append(mixed.get('reward_model_device','cpu'))
        if mixed.get('critic_kind')=='hf':
            required.append(mixed.get('critic_hf_device','cpu'))
    if 'mps' in required:
        import torch
        if not torch.backends.mps.is_available():
            raise ValueError('Apple MPS selected but unavailable in this PyTorch/macOS build. Run python kagami.py mps-ppo-smoke to diagnose.')
    if 'cuda' in required:
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA selected but no NVIDIA CUDA accelerator is available')
