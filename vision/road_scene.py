"""
Road-scene detection: traffic lights (pretrained YOLOv8) and speed bumps
(custom ``bump_model.pt``).

Design notes
------------
* ``light_state`` is a pure function (BGR crop -> RED/YELLOW/GREEN/UNKNOWN),
  so the colour logic is unit tested without torch or a camera.
* The YOLO detectors import ``ultralytics`` lazily inside ``load()``. Importing
  this module therefore stays cheap and works on machines without the optional
  YOLO extras (the dashboard imports ``vision`` on every start).
* ``RedLightTimer`` isolates the "red light held for N seconds" rule so the
  STOP alert can be tested with fake timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO class id of "traffic light" in the pretrained yolov8n model.
TRAFFIC_CLASS = 9

DEFAULT_CONF = 0.45
MIN_LIGHT_PX = 8          # ignore boxes smaller than this (noise)
MIN_COLOR_PIXELS = 20     # min coloured pixels before we trust the lamp state
RED_ALERT_SEC = 2.0       # red held this long -> STOP alert


@dataclass(frozen=True)
class Detection:
    """One detected object in frame coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


# --------------------------------------------------------------------- colour --
def light_state(crop) -> str:
    """Dominant colour of a traffic-light crop (BGR) -> RED/YELLOW/GREEN/UNKNOWN."""
    import cv2

    if crop is None or getattr(crop, "size", 0) == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255])) | \
        cv2.inRange(hsv, np.array([160, 90, 90]), np.array([180, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 90, 90]), np.array([40, 255, 255]))
    green = cv2.inRange(hsv, np.array([45, 70, 70]), np.array([90, 255, 255]))

    scores = {
        "RED": cv2.countNonZero(red),
        "YELLOW": cv2.countNonZero(yellow),
        "GREEN": cv2.countNonZero(green),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > MIN_COLOR_PIXELS else "UNKNOWN"


def light_colour(state: str) -> tuple[int, int, int]:
    """BGR drawing colour for a lamp state."""
    return {
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
        "GREEN": (0, 200, 0),
    }.get(state, (160, 160, 160))


# ------------------------------------------------------------------ box utils --
def _box_values(box) -> list[float]:
    """``[x1, y1, x2, y2]`` from a torch tensor row or a plain list."""
    return [float(v) for v in box.xyxy[0]]


def clip_box(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Clamp a float box to the frame; ``None`` when it is too small to use."""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(int(width), x2), min(int(height), y2)
    if x2 - x1 < MIN_LIGHT_PX or y2 - y1 < MIN_LIGHT_PX:
        return None
    return x1, y1, x2, y2


def select_traffic_light(detections) -> Detection | None:
    """The most confident traffic light (closest / clearest lamp)."""
    lights = [d for d in detections if d.label == "traffic light"]
    return max(lights, key=lambda d: d.confidence, default=None)


# ------------------------------------------------------------------ detectors --
class _YoloDetector:
    """Shared lazy-loading YOLO wrapper."""

    def __init__(self, weights: str, conf: float = DEFAULT_CONF, model=None):
        self.weights = weights
        self.conf = float(conf)
        self._model = model

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self):
        """Load the weights once. Raises RuntimeError if ultralytics is missing."""
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "Missing ultralytics. Run: pip install -r requirements-yolo.txt"
                ) from exc
            self._model = YOLO(self.weights)
        return self._model

    def _raw_boxes(self, frame):
        model = self.load()
        result = model.predict(frame, conf=self.conf, verbose=False)[0]
        return result.boxes

    def detect(self, frame) -> list[Detection]:
        """Run the model and return every kept box (subclasses filter)."""
        raise NotImplementedError


class TrafficLightDetector(_YoloDetector):
    """Pretrained COCO model; keeps only traffic lights."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = DEFAULT_CONF,
                 model=None):
        super().__init__(weights, conf, model)

    def detect(self, frame) -> list[Detection]:
        height, width = frame.shape[:2]
        found: list[Detection] = []
        for box in self._raw_boxes(frame):
            if int(box.cls[0]) != TRAFFIC_CLASS:
                continue
            clipped = clip_box(_box_values(box), width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            found.append(Detection(x1, y1, x2, y2, float(box.conf[0]),
                                   "traffic light"))
        return found


class BumpDetector(_YoloDetector):
    """Custom speed-bump model (single class, trained by this project)."""

    def __init__(self, weights: str = "bump_model.pt", conf: float = DEFAULT_CONF,
                 model=None):
        super().__init__(weights, conf, model)

    def load(self):
        """Load once; stays unready (instead of raising) if the file is absent."""
        if self._model is None:
            import os
            if not os.path.exists(self.weights):
                return None
            super().load()
        return self._model

    def detect(self, frame) -> list[Detection]:
        if self.load() is None:
            return []
        height, width = frame.shape[:2]
        found: list[Detection] = []
        for box in self._raw_boxes(frame):
            clipped = clip_box(_box_values(box), width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            found.append(Detection(x1, y1, x2, y2, float(box.conf[0]), "speedbump"))
        return found


# --------------------------------------------------------------------- timer --
class RedLightTimer:
    """Tracks how long the light has been red in a row."""

    def __init__(self, hold_sec: float = RED_ALERT_SEC):
        self.hold_sec = float(hold_sec)
        self.since: float | None = None

    def update(self, state: str, now: float) -> float:
        """Return seconds spent on red; resets as soon as the light changes."""
        if state != "RED":
            self.since = None
            return 0.0
        if self.since is None:
            self.since = now
        return max(0.0, now - self.since)

    @property
    def alerting(self) -> float:
        return self.hold_sec

    def should_alert(self, held: float) -> bool:
        return held >= self.hold_sec
