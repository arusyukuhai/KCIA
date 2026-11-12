"""MPS training hot paths, local runtime only (never a signed objective).

Apple MPS does not have fp64; keep high-precision IS verification on CPU.
Native multinomial avoids copying a whole vocabulary distribution to the host
on every generated token. An explicitly unsupported MPS kernel falls back to
CPU; unrelated errors fail closed instead of being silently swallowed.
"""
from __future__ import annotations

import os
import torch

_UNSUPPORTED = ('not implemented for', 'not implemented on', 'not supported on the mps',
                'mps backend does not support', 'could not run',
                'not currently implemented for the mps')


def native_sampling_enabled() -> bool:
    return os.environ.get('KCIA_MPS_NATIVE_SAMPLING', '1').lower() not in ('0', 'false', 'off')


def sample_next(log_prob, *, backend=None):
    """Return (sample, backend), retaining sampling probability semantics.

    Only the scalar sampled token needs to cross back to the host for EOS
    handling; ordinary MPS success never copies [vocab] probabilities.
    """
    target = str(backend or log_prob.device)
    probs = log_prob.exp().float()
    if target.startswith('mps'):
        if native_sampling_enabled():
            try:
                return torch.multinomial(probs, 1), 'mps-native'
            except (RuntimeError, NotImplementedError) as exc:
                if not any(mark in str(exc).lower() for mark in _UNSUPPORTED):
                    raise
        return torch.multinomial(probs.cpu(), 1), 'cpu-fallback'
    return torch.multinomial(probs, 1), 'device-native'


def terminal_gae(reward, old_values, gamma, gae_lambda):
    """Compute detached per-token GAE with a single host transfer on MPS.

    Recurrence is 2-5 tiny tensor kernels per token on MPS; CPU is faster for
    these serial scalar operations, while the Actor/Critic model stays on GPU.
    No CPU gradient detach is lost: old_values are intentionally baselines.
    """
    device = old_values.device
    n = int(old_values.numel())
    if n < 1:raise ValueError('GAE needs nonempty rollout')
    if device.type == 'mps':
        values = old_values.detach().float().cpu().tolist()
        adv = [0.0] * n
        value = 0.0
        for i in range(n-1, -1, -1):
            next_val = values[i+1] if i+1<n else 0.0
            terminal_reward = float(reward) if i == n-1 else 0.0
            value = terminal_reward + gamma*next_val-values[i] + gamma*gae_lambda*value
            adv[i] = value
        advantages = torch.tensor(adv, dtype=torch.float32, device=device)
        return advantages, advantages + old_values.detach().float()
    rs = torch.zeros(n, device=device, dtype=torch.float32)
    rs[-1] = reward
    advantages = torch.zeros(n, device=device, dtype=torch.float32)
    gae = torch.zeros((), device=device, dtype=torch.float32)
    values=old_values.detach().float()
    for i in range(n-1,-1,-1):
        next_v=values[i+1] if i+1<n else torch.zeros((),device=device)
        delta=rs[i]+gamma*next_v-values[i]
        gae=delta+gamma*gae_lambda*gae
        advantages[i]=gae
    return advantages, advantages + values
