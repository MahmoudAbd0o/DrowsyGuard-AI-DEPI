"""Glue between the road-scene detectors and the dashboard.

YOLO is expensive, so it runs every ``every`` frames on a downscaled copy of
the frame; the boxes are scaled back to full-frame pixels and kept until the
next pass. The scheduling and the red-light rule are pure logic, so this module
can be tested without torch or a camera (fake detectors are injected).

A standalone camera loop is no longer needed: the dashboard calls this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from vision.road_scene import (BumpDetector, Detection, RedLightTimer,
                               TrafficLightDetector, light_state)


@dataclass
class RoadResult:
    """Latest road-scene snapshot (safe to hold between YOLO passes)."""

    light: Detection | None = None
    light_state: str = "NO LIGHT"
    bumps: list[Detection] = field(default_factory=list)
    red_held: float = 0.0
    red_alert: bool = False
    ran: bool = False

    @property
    def summary(self) -> str:
        light = self.light_state if self.light is not None else "no light"
        return f"light: {light} | bumps: {len(self.bumps)}"


class RoadMonitor:
    """Runs bump + traffic-light detection on a schedule."""

    def __init__(self, lights=None, bumps=None,
                 every: int | None = None,
                 downscale: int | None = None,
                 red_hold: float | None = None):
        self.lights = lights
        self.bumps = bumps
        self.every = max(1, int(config.ROAD_YOLO_EVERY if every is None else every))
        self.downscale = max(
            1, int(config.ROAD_YOLO_SCALE if downscale is None else downscale))
        self.red = RedLightTimer(
            config.RED_LIGHT_ALERT_SEC if red_hold is None else red_hold)
        self.frames = 0
        self.result = RoadResult()

    # ------------------------------------------------------------- factory --
    @property
    def ready(self) -> bool:
        return self.lights is not None or self.bumps is not None

    @classmethod
    def load(cls) -> "RoadMonitor":
        """Build with the real YOLO weights; never raises.

        A missing ``ultralytics`` install or a missing weights file simply
        leaves that detector off, so the driver monitor keeps working.
        """
        lights = None
        bumps = None
        try:
            lights = TrafficLightDetector(config.YOLO_LIGHT_WEIGHTS,
                                          config.YOLO_CONF)
            lights.load()
        except Exception:
            lights = None
        try:
            bumps = BumpDetector(config.YOLO_BUMP_WEIGHTS, config.YOLO_CONF)
            if bumps.load() is None:
                bumps = None
        except Exception:
            bumps = None
        return cls(lights=lights, bumps=bumps)

    # --------------------------------------------------------------- update --
    def update(self, frame, now: float) -> RoadResult:
        """Return the current result, running YOLO on schedule."""
        if not self.ready:
            return self.result

        self.frames += 1
        if self.frames % self.every != 0:
            return self.result

        small, scale = self._shrink(frame)

        light: Detection | None = None
        state = "NO LIGHT"
        if self.lights is not None:
            try:
                found = self.lights.detect(small)
            except Exception:
                found = []
            light = max(found, key=lambda d: d.confidence, default=None)
            if light is not None:
                x1, y1, x2, y2 = light.box
                state = light_state(small[y1:y2, x1:x2])
                light = self._rescale(light, scale)

        bumps: list[Detection] = []
        if self.bumps is not None:
            try:
                bumps = [self._rescale(d, scale) for d in self.bumps.detect(small)]
            except Exception:
                bumps = []

        held = self.red.update(state, now)
        self.result = RoadResult(
            light=light,
            light_state=state,
            bumps=bumps,
            red_held=held,
            red_alert=self.red.should_alert(held),
            ran=True,
        )
        return self.result

    def reset(self) -> None:
        self.frames = 0
        self.red = RedLightTimer(self.red.hold_sec)
        self.result = RoadResult()

    # -------------------------------------------------------------- helpers --
    def _shrink(self, frame):
        """Downscaled copy for YOLO + the factor needed to map boxes back."""
        if self.downscale <= 1:
            return frame, 1.0
        import cv2

        height, width = frame.shape[:2]
        small = cv2.resize(
            frame,
            (max(1, width // self.downscale), max(1, height // self.downscale)),
            interpolation=cv2.INTER_AREA,
        )
        return small, width / float(small.shape[1])

    @staticmethod
    def _rescale(det: Detection, factor: float) -> Detection:
        if factor == 1.0:
            return det
        return Detection(
            int(round(det.x1 * factor)), int(round(det.y1 * factor)),
            int(round(det.x2 * factor)), int(round(det.y2 * factor)),
            det.confidence, det.label,
        )
