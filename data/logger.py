"""
Session logging.

``MeasurementLogger`` writes one row per processed frame (measurements +
prediction), buffered so disk I/O never slows the video loop.
``EventLog`` keeps the human-readable alert history.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

import config

MEASUREMENT_FIELDS = [
    "timestamp", "state", "level", "perclos", "ear", "mar",
    "attention", "head_down", "distracted", "yawning",
    "face_present", "coverage",
]


def _ensure_parent(path: str) -> str:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    return folder


class MeasurementLogger:
    """Buffered CSV writer for per-frame measurements."""

    def __init__(self, path: str | None = None, buffer_size: int = 60):
        self.path = path or os.path.join("outputs", "session_log.csv")
        _ensure_parent(self.path)
        self.buffer_size = max(1, int(buffer_size))
        self._buffer: list[dict] = []
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=MEASUREMENT_FIELDS)
        self._writer.writeheader()
        self.rows_written = 0

    def log(self, *, timestamp: float, state: str, level: str,
            perclos: float | None = None, ear: float | None = None,
            mar: float | None = None, attention: float | None = None,
            head_down: bool = False, distracted: bool = False,
            yawning: bool = False, face_present: bool = True,
            coverage: float | None = None) -> None:
        self._buffer.append({
            "timestamp": f"{timestamp:.3f}",
            "state": state,
            "level": level,
            "perclos": "" if perclos is None else f"{perclos:.2f}",
            "ear": "" if ear is None else f"{ear:.4f}",
            "mar": "" if mar is None else f"{mar:.4f}",
            "attention": "" if attention is None else f"{attention:.2f}",
            "head_down": int(bool(head_down)),
            "distracted": int(bool(distracted)),
            "yawning": int(bool(yawning)),
            "face_present": int(bool(face_present)),
            "coverage": "" if coverage is None else f"{coverage:.3f}",
        })
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self._file.flush()
        self.rows_written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        try:
            self._file.close()
        except Exception:
            pass

    def __enter__(self) -> "MeasurementLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class EventLog:
    """Human-readable alert history saved as (time, event) rows."""

    def __init__(self, path: str | None = None):
        self.path = path or config.SESSION_REPORT
        self.rows: list[tuple[str, str]] = []

    def add(self, message: str, timestamp: str | None = None) -> None:
        stamp = timestamp or datetime.now().strftime("%H:%M:%S")
        self.rows.append((stamp, message))

    def save(self) -> bool:
        try:
            _ensure_parent(self.path)
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(("time", "event"))
                writer.writerows(self.rows)
            return True
        except Exception:
            return False
