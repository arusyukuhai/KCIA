"""Crash containment policy for macOS Metal/MPS.

There are native Metal/PyTorch assertions that abort the Python interpreter,
not Python exceptions. We cannot recover their in-memory optimizer state.
Never silently select a crash-prone backend for a new training session.
This policy is LOCAL ONLY; it never changes the model's signed identity.
"""
from __future__ import annotations

import os
import sys

_ENV = "KCIA_EXPERIMENTAL_MPS"
_TRUTHY = frozenset(("1", "true", "yes", "on"))
_reported = False


def is_macos() -> bool:
    return sys.platform == "darwin"


def experimental_mps() -> bool:
    return str(os.environ.get(_ENV, "")).strip().lower() in _TRUTHY


def mps_allowed() -> bool:
    return not is_macos() or experimental_mps()


def safe_auto_device(*, cuda: bool = False, mps: bool = False) -> str:
    if cuda:
        return "cuda"
    if mps and mps_allowed():
        return "mps"
    return "cpu"


def check_device(device, *, field: str = "device") -> None:
    name = str(device).lower()
    if (name == "mps" or name.startswith("mps:")) and not mps_allowed():
        raise RuntimeError(
            f"{field}=mps is disabled by KCIA v92 on macOS because native "
            "Metal command-buffer assertions abort the interpreter and may "
            "leave optimizer state unpublished. Set KCIA_EXPERIMENTAL_MPS=1 "
            "to opt in explicitly; or use device=cpu (recommended for stability)."
        )


def guard_training_devices(args) -> None:
    for name in ("device", "reward_model_device", "critic_hf_device"):
        check_device(getattr(args, name, "cpu"), field=name)


def notify_safe_cpu() -> None:
    global _reported
    if is_macos() and not experimental_mps() and not _reported:
        _reported = True
        print("KCIA MPS SAFE MODE: automatic device=cpu on macOS. "
              "Existing signed logs, weights and network protocol are unchanged. "
              "For experimental GPU, set KCIA_EXPERIMENTAL_MPS=1 before launch.",
              file=sys.stderr, flush=True)
