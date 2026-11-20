"""Local-only WAN publication cadence, independent of signed room identity.

The schedule changes only when a new signed event is published. No peer may
force a shorter interval with untrusted network telemetry, and it never
changes ReLoRA event counts, epochs, genesis or the wire encoding.
"""
from __future__ import annotations
import math


def recommend_steps(*, parameters: int, observed_patch_bytes: int = 0,
                    baseline: int = 32, maximum: int = 512) -> int:
    if (any(type(x) is not int for x in (parameters,observed_patch_bytes,baseline,maximum))
            or parameters < 0 or observed_patch_bytes < 0 or baseline < 1
            or maximum < baseline or maximum > 10000):
        raise ValueError('invalid WAN cadence bounds')
    # A model 8x bigger receives roughly 2x the local training allowance.
    # A small model or unknown model retains the WAN floor.
    scale=max(1.0,parameters/150_000_000)
    steps=max(baseline,32*2**max(0,math.ceil(math.log2(scale)/3)))
    # Target at most ~32 MiB per 32 minibatches, before considering overhead.
    # This is an advisory ratio, not a network bandwidth measurement.
    if observed_patch_bytes:
        wire_ratio=observed_patch_bytes/(32*1024**2)
        if wire_ratio>1:
            steps=max(steps,32*2**math.ceil(math.log2(wire_ratio)))
    steps=min(maximum,int(steps))
    # Keep the same 4-minibatch increments used by the GUI.
    return max(baseline,((steps+3)//4)*4)
