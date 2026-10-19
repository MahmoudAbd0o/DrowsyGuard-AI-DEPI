"""
Timestamp-based PERCLOS (percentage of eye-closure time).

Unlike the original frame-count deque, this meter:
  * uses real timestamps, so a variable frame rate cannot distort the value;
  * ignores invalid measurements instead of counting them as "eyes open";
  * reports measurement coverage so callers can tell UNKNOWN from ALERT.
"""
from __future__ import annotations

from collections import deque

import config


class PerclosMeter:
    """Rolling PERCLOS over a time window.

    ``update(timestamp, eye_closed)``:
        timestamp  - seconds (monotonic clock)
        eye_closed - True (closed), False (open) or None (invalid / no face)
    """

    def __init__(self,
                 window_sec: float | None = None,
                 min_obs: int | None = None,
                 min_coverage: float | None = None):
        self.window_sec = float(config.PERCLOS_WINDOW_SEC
                                if window_sec is None else window_sec)
        self.min_obs = int(config.PERCLOS_MIN_OBS
                           if min_obs is None else min_obs)
        self.min_coverage = float(config.PERCLOS_MIN_COVERAGE
                                  if min_coverage is None else min_coverage)
        self._samples: deque[tuple[float, int]] = deque()
        self.total = 0
        self.valid = 0

    # ------------------------------------------------------------- internal --
    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    # ----------------------------------------------------------------- public --
    def update(self, timestamp: float, eye_closed: bool | None) -> float | None:
        """Record one observation and return the current PERCLOS percentage."""
        self.total += 1
        if eye_closed is not None:
            self.valid += 1
            self._samples.append((float(timestamp), 1 if eye_closed else 0))
        self._prune(timestamp)
        return self.value()

    def value(self) -> float | None:
        """Current PERCLOS in [0, 100], or ``None`` when data is insufficient."""
        if len(self._samples) < self.min_obs:
            return None
        if self.coverage < self.min_coverage:
            return None
        closed = sum(v for _, v in self._samples)
        return 100.0 * closed / len(self._samples)

    @property
    def coverage(self) -> float:
        """Fraction of observations that were valid (0.0 - 1.0)."""
        return (self.valid / self.total) if self.total else 0.0

    @property
    def samples(self) -> int:
        """Number of valid samples currently inside the window."""
        return len(self._samples)

    def reset(self) -> None:
        self._samples.clear()
        self.total = 0
        self.valid = 0
