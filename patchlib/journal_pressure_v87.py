"""Bounded, proactive generation-export signals for signed update logs.

Never removes signed events and never creates an uncertified epoch. Existing
committee/certificate machinery remains the only authority for event GC.
"""
from __future__ import annotations


def should_propose_for_pressure(used: int, cap: int, last_attempt: int) -> bool:
    """Request a certified generation export well before a finite journal cap.

    Repeat only after substantial growth to bound costly failed attempts. This
    is deliberately subordinate to the existing committee and trainer fences.
    """
    if any(type(x) is not int or x < 0 for x in (used, cap, last_attempt)):
        raise ValueError('invalid signed-journal pressure measurements')
    if cap == 0 or used * 100 < cap * 35:
        return False
    if last_attempt == 0:
        return True
    # A few attempts across the remaining 65% of the cap, even when the
    # normal export cadence is not reached (e.g. only inbound patches grow).
    step = max(1, min(128 * 1024**2, max(1, cap // 16)))
    return used >= last_attempt + step


def should_pause_local_training_for_epoch(used: int, cap: int) -> bool:
    """Reserve headroom for inbound signed events and checkpoint/epoch traffic.

    This is only used for generation-auto rooms; without a live quorum it
    pauses (and reports the blocker) rather than discarding signed history.
    """
    if type(used) is not int or type(cap) is not int or used < 0 or cap < 0:
        raise ValueError('invalid signed-journal pressure measurements')
    return cap > 0 and used * 100 >= cap * 80


def cap_message(used: int, extra: int, cap: int) -> str:
    """Explain an artificial, recoverable quota without claiming physical ENOSPC."""
    gib = 1024**3
    return (
        'durable update log is full '
        f'(signed payload={used/gib:.3f} GiB, incoming={extra/gib:.3f} GiB, '
        f'configured cap={cap/gib:.3f} GiB). '
        'This is KCIA\'s journal quota, not necessarily physical disk exhaustion. '
        'Keep the signed log intact: certified checkpoint/epoch rollover is '
        'required before old P2P history can be safely retired. '
        'For a temporary increase, raise Journal GiB in the GUI or '
        '--max-journal-gb in the CLI ONLY if disk-report shows sufficient free space. '
        'For a disposable DEBUG room, stop all peers and use reset-local-data '
        'then create a fresh room (this loses its learning history). '
        'Run python kcia.py journal-status --journal PATH to see why certified rollover is blocked.'
    )
