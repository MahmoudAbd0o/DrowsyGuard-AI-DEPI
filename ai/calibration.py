"""
Personal eye calibration and EAR smoothing.

Why this module exists
----------------------
A single hard-coded ``EAR_CLOSED_THRESH`` cannot fit every driver: eye shape,
glasses, make-up and the camera geometry all shift the "open eye" value. On a
driver whose open EAR is ~0.25 the default 0.21 threshold fires constantly,
which is exactly how a false DROWSY alarm is born.

``EyeCalibrator`` solves that by measuring the driver's own open-eye EAR for a
few seconds and deriving a personal threshold from it.

``EarSmoother`` adds a rolling median so a single bad frame (motion blur, head
snap) can never fake a closed eye on its own.
"""
from __future__ import annotations

import statistics
from collections import deque

import config


class EarSmoother:
    """Rolling median of the last N *valid* EAR samples."""

    def __init__(self, window: int | None = None):
        size = config.EAR_SMOOTH_WINDOW if window is None else window
        self.window = max(1, int(size))
        self._values: deque[float] = deque(maxlen=self.window)

    def update(self, value: float | None) -> float | None:
        """Add one sample and return the smoothed value.

        ``None`` means "this frame is not trustworthy": nothing is added and
        ``None`` is returned, so stale data is never reported as a fresh
        measurement.
        """
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric != numeric:          # NaN guard
            return None
        self._values.append(numeric)
        return float(statistics.median(self._values))

    def reset(self) -> None:
        self._values.clear()

    @property
    def samples(self) -> int:
        return len(self._values)


def eye_gate(face_width_px: float,
             turn: float | None,
             ear: float | None) -> str:
    """Explain whether this frame's EAR can be trusted.

    Returns ``"ok"`` or a short human-readable reason. This is the guard that
    stops a turned head or a far-away face from being read as "eyes closed"
    (the root cause of the false DROWSY alarms).
    """
    if face_width_px and face_width_px < config.FACE_TOO_FAR_WIDTH_PX:
        return "face too small"
    if turn is not None and turn > config.EAR_MAX_TURN_RATIO:
        return f"head turned {turn:.2f}"
    if ear is None:
        return "no eye points"
    return "ok"


class EyeCalibrator:
    """Learn the driver's open-eye EAR and a personal closed-eye threshold.

    Usage::

        calib = EyeCalibrator()          # starts on the first valid sample
        calib.feed(now, smoothed_ear)    # every frame
        if not calib.done:
            ...                          # show "keep your eyes open"
        closed = ear < calib.threshold

    If the driver never looks at the camera, calibration falls back to
    ``config.EAR_CLOSED_THRESH`` instead of blocking the application.
    """

    def __init__(self,
                 duration: float | None = None,
                 min_samples: int | None = None):
        self.duration = float(config.EAR_CALIBRATION_SEC
                              if duration is None else duration)
        self.min_samples = int(config.EAR_CALIBRATION_MIN_SAMPLES
                               if min_samples is None else min_samples)
        self.reset()

    # ------------------------------------------------------------- lifecycle --
    def reset(self, timestamp: float | None = None) -> None:
        """Forget the previous calibration and start over."""
        self.samples: list[float] = []
        self.started_at = float(timestamp) if timestamp is not None else None
        self.last_at: float | None = None
        self.done = False
        self.baseline: float | None = None
        self.threshold = float(config.EAR_CLOSED_THRESH)

    # ----------------------------------------------------------------- public --
    @property
    def active(self) -> bool:
        """True while samples are still being collected."""
        return self.started_at is not None and not self.done

    @property
    def progress(self) -> float:
        """0.0 - 1.0 collection progress (1.0 when finished)."""
        if self.done:
            return 1.0
        if self.started_at is None or self.last_at is None or self.duration <= 0:
            return 0.0
        elapsed = self.last_at - self.started_at
        return max(0.0, min(1.0, elapsed / self.duration))

    def feed(self, timestamp: float, ear: float | None) -> bool:
        """Collect one sample. Returns ``True`` once calibration is finished."""
        if self.done:
            return True
        stamp = float(timestamp)
        usable = ear is not None and ear >= config.EAR_CALIBRATION_MIN_BASELINE
        if not usable:
            # Nothing usable (no face, blink, turned head): do not *start* the
            # clock, otherwise a driver who is not looking would end up
            # "calibrated" on empty data. The clock keeps running once started.
            if self.started_at is None:
                return False
            if stamp - self.started_at >= self.duration:
                self.finish()
                return True
            return False

        if self.started_at is None:
            self.started_at = stamp
        self.last_at = stamp
        self.samples.append(float(ear))

        if stamp - self.started_at >= self.duration:
            self.finish()
            return True
        return False

    def finish(self) -> float:
        """Derive the threshold now and mark calibration as finished."""
        if len(self.samples) >= self.min_samples:
            self.baseline = float(statistics.median(self.samples))
            raw = self.baseline * float(config.EAR_CLOSED_RATIO)
            self.threshold = float(min(config.EAR_CLOSED_MAX,
                                       max(config.EAR_CLOSED_MIN, raw)))
        else:
            # Not enough evidence: keep the conservative default.
            self.baseline = None
            self.threshold = float(config.EAR_CLOSED_THRESH)
        self.done = True
        return self.threshold

    def is_closed(self, ear: float | None) -> bool | None:
        """``None`` for invalid input, otherwise "is this eye closed?"."""
        if ear is None:
            return None
        return float(ear) < self.threshold

    def summary(self) -> str:
        """Short human-readable description for the UI / log."""
        if not self.done:
            return "calibrating..."
        if self.baseline is None:
            return f"default threshold {self.threshold:.2f} (not enough samples)"
        return (f"open EAR {self.baseline:.2f} -> closed below "
                f"{self.threshold:.2f}")
