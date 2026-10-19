"""
Camera capture with automatic reconnection.

Keeps video I/O out of the dashboard: the dashboard asks for a frame and the
camera decides whether to reconnect after a failure streak.
"""
from __future__ import annotations

import cv2
import numpy as np


class Camera:
    """Webcam wrapper with open-with-fallback, warm-up and reconnect."""

    def __init__(self,
                 indices: tuple[int, ...] = (0, 1, 2),
                 reconnect_after: int = 30,
                 retry_every: int = 90,
                 max_failures: int = 150,
                 warmup_reads: int = 10):
        self.indices = tuple(indices)
        self.reconnect_after = int(reconnect_after)
        self.retry_every = int(retry_every)
        self.max_failures = int(max_failures)
        self.warmup_reads = int(warmup_reads)

        self.cap = None
        self.index: int | None = None
        self.failures = 0
        self.reconnects = 0
        self.dead = False
        self._warm = False

    # ------------------------------------------------------------- lifecycle --
    def open(self) -> bool:
        """Open the first working camera. Returns True on success."""
        for idx in self.indices:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            self.cap = cap
            self.index = idx
            self._warm = self._warm_up()
            if self._warm:
                self.failures = 0
                self.dead = False
                return True
            cap.release()
            self.cap = None
        self.index = None
        return False

    def _warm_up(self) -> bool:
        """Read a few frames so the sensor is ready (Zoom/Meet may hold it)."""
        if self.cap is None:
            return False
        got = 0
        for _ in range(self.warmup_reads):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                got += 1
        return got > 0

    def read(self) -> np.ndarray | None:
        """Return the next frame, or ``None`` while the camera is unavailable."""
        if self.cap is None or self.dead:
            return None

        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.failures = 0
            return frame

        self.failures += 1
        if self.failures == self.reconnect_after:
            self._reopen()
            self.reconnects += 1
        elif self.failures > self.reconnect_after and self.failures % self.retry_every == 0:
            self._reopen()
            self.reconnects += 1
        if self.failures >= self.max_failures:
            self.dead = True
        return None

    def _reopen(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        if self.index is None:
            return
        cap = cv2.VideoCapture(self.index)
        if cap.isOpened():
            self.cap = cap
        else:
            cap.release()

    def release(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.cap = None
            self.dead = False

    # ---------------------------------------------------------------- info --
    @property
    def is_open(self) -> bool:
        return self.cap is not None and not self.dead

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
