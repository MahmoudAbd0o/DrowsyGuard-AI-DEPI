"""Tests for the road-monitor glue (no torch, no camera)."""
import unittest

import numpy as np

from vision.road_monitor import RoadMonitor
from vision.road_scene import Detection


class FakeDetector:
    """Stands in for TrafficLightDetector / BumpDetector."""

    def __init__(self, detections=None):
        self.detections = list(detections or [])
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self.detections


def red_frame(width=200, height=100):
    """A frame whose every crop reads as RED."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)
    return frame


def light(x1=10, y1=10, x2=30, y2=30, conf=0.9):
    return Detection(x1, y1, x2, y2, conf, "traffic light")


class TestRoadMonitor(unittest.TestCase):
    def test_without_detectors_it_is_inactive(self):
        monitor = RoadMonitor(lights=None, bumps=None)
        self.assertFalse(monitor.ready)
        result = monitor.update(red_frame(), 0.0)
        self.assertFalse(result.ran)
        self.assertEqual(result.bumps, [])

    def test_runs_every_n_frames(self):
        lights = FakeDetector([light()])
        monitor = RoadMonitor(lights=lights, every=3, downscale=1)
        frame = red_frame()
        for _ in range(2):
            monitor.update(frame, 0.0)
        self.assertEqual(lights.calls, 0)
        monitor.update(frame, 0.0)
        self.assertEqual(lights.calls, 1)
        for _ in range(3):
            monitor.update(frame, 0.0)
        self.assertEqual(lights.calls, 2)

    def test_boxes_are_scaled_back_to_the_full_frame(self):
        bumps = FakeDetector([Detection(10, 10, 20, 20, 0.8, "speedbump")])
        monitor = RoadMonitor(bumps=bumps, every=1, downscale=2)
        result = monitor.update(red_frame(200, 100), 0.0)
        self.assertEqual(len(result.bumps), 1)
        # 200 / (200/2) = 2.0
        self.assertEqual(result.bumps[0].box, (20, 20, 40, 40))

    def test_red_light_alert_needs_time(self):
        lights = FakeDetector([light()])
        monitor = RoadMonitor(lights=lights, every=1, downscale=1, red_hold=2.0)
        frame = red_frame()
        self.assertEqual(monitor.update(frame, 0.0).light_state, "RED")
        self.assertFalse(monitor.update(frame, 1.0).red_alert)
        self.assertTrue(monitor.update(frame, 2.1).red_alert)

    def test_green_cancels_the_red_alert(self):
        lights = FakeDetector([light()])
        monitor = RoadMonitor(lights=lights, every=1, downscale=1, red_hold=1.0)
        monitor.update(red_frame(), 0.0)
        monitor.update(red_frame(), 1.5)
        lights.detections = []
        cleared = monitor.update(red_frame(), 2.0)
        self.assertEqual(cleared.light_state, "NO LIGHT")
        self.assertFalse(cleared.red_alert)

    def test_result_is_cached_between_passes(self):
        bumps = FakeDetector([Detection(10, 10, 20, 20, 0.8, "speedbump")])
        monitor = RoadMonitor(bumps=bumps, every=5, downscale=1)
        frame = red_frame()
        for _ in range(5):
            first = monitor.update(frame, 0.0)      # 5th frame -> pass
        self.assertTrue(first.bumps)
        for _ in range(4):
            cached = monitor.update(frame, 0.0)
            self.assertEqual(cached.bumps[0].box, first.bumps[0].box)

    def test_reset_clears_state(self):
        lights = FakeDetector([light()])
        monitor = RoadMonitor(lights=lights, every=1, downscale=1)
        monitor.update(red_frame(), 0.0)
        monitor.reset()
        self.assertFalse(monitor.result.ran)
        self.assertEqual(monitor.frames, 0)

    def test_a_broken_detector_does_not_crash_the_dashboard(self):
        class Broken:
            def detect(self, frame):
                raise RuntimeError("CUDA out of memory")

        monitor = RoadMonitor(lights=Broken(), bumps=Broken(), every=1,
                              downscale=1)
        result = monitor.update(red_frame(), 0.0)
        self.assertIsNone(result.light)
        self.assertEqual(result.bumps, [])


if __name__ == "__main__":
    unittest.main()
