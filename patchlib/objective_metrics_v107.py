"""Local *training* metrics for Kagami PT/SFT/PPO. Not validation scores.

Nothing here changes model weights, the signed learning protocol, or the
optimizer. This module deliberately does not infer an accuracy for an HF
reward scorer, because scores without ground-truth targets are not accuracy.
"""
from __future__ import annotations

import math


def token_top1(logits, labels, *, causal: bool = True):
    """Masked top-1 over supervised positions; return None when undefined.

    Causal LM: compare positions t logits to labels at t+1 (ignoring -100).
    Sequence classification: compare logits argmax to class labels.
    Never compute token top-1 on regression / incompatible output heads.
    """
    import torch

    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return None
    with torch.no_grad():
        if causal:
            if not (logits.ndim == 3 and labels.ndim == 2 and
                    logits.shape[:2] == labels.shape and logits.shape[1] > 1):
                return None
            predicted = logits[:, :-1, :].argmax(-1)
            truth = labels[:, 1:]
        else:
            if not (logits.ndim == 2 and labels.ndim == 1 and
                    logits.shape[0] == labels.shape[0]):
                return None
            predicted, truth = logits.argmax(-1), labels
        valid = truth != -100
        total = int(valid.sum().item())
        if total == 0:
            return None
        ratio = float(((predicted == truth) & valid).sum().item()) / total
        return ratio if math.isfinite(ratio) else None


def reference_token_match(tokens, reference):
    """Positional overlap/max(lengths), not validation or semantic accuracy.

    A rollout with a learned HF reward model has no reference and therefore
    no meaningful reference-match score. Different lengths count as errors.
    """
    if reference is None:
        return None
    length = max(len(tokens), len(reference))
    if length == 0:
        return None
    return sum(int(a == b) for a, b in zip(tokens, reference)) / length


def finite_metric(value, *, minimum=None, maximum=None):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number
