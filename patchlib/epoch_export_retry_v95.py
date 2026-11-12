"""Bounded retry scheduling for automatic certified epoch exports.

A stalled journal stops growing after V89 pressure HOLD; retries therefore must
not depend on additional event bytes or local optimizer steps.  Scheduling is
local policy only: no certificate, checkpoint, or journal is ever forged or
removed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class EpochExportRetry:
    failures: int = 0
    retry_at: float = 0.0
    last_error: str = ''
    stage: str = 'idle'

    def can_retry(self, now: float) -> bool:
        return self.failures > 0 and now >= self.retry_at

    def failed(self, now: float, error: BaseException | str) -> float:
        if not math.isfinite(now):
            raise ValueError('nonfinite scheduler clock')
        self.failures += 1
        delay = min(120.0, 5.0 * (2 ** min(self.failures - 1, 5)))
        self.retry_at = now + delay
        self.last_error = str(error)[:600]
        self.stage = 'retry-scheduled'
        return delay

    def started(self) -> None:
        self.stage = 'exporting'

    def succeeded(self) -> None:
        self.failures = 0
        self.retry_at = 0.0
        self.last_error = ''
        self.stage = 'candidate-offered'

    def status(self, now: float) -> str:
        if self.failures:
            return f'{self.stage}; attempt={self.failures}; retry_in={max(0.0, self.retry_at-now):.1f}s; error={self.last_error}'
        return self.stage
