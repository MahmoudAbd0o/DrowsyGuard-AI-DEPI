"""Unit tests for road-scene detection (no ultralytics / camera required)."""
import unittest
from types import SimpleNamespace

import numpy as np

from vision import road_scene as rs


def solid(bgr, size=40):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


class FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [xyxy]
        self.conf = [conf]
        self.cls = [cls]


class FakeModel:
    """Minimal stand-in for ultralytics.YOLO."""

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, frame, conf=0.45, verbose=False):
        return [SimpleNamespace(boxes=self.boxes)]


class TestLightState(unittest.TestCase):
    def test_red(self):
        self.assertEqual(rs.light_state(solid((0, 0, 255))), "RED")

    def test_green(self):
        self.assertEqual(rs.light_state(solid((0, 255, 0))), "GREEN")

    def test_yellow(self):
        self.assertEqual(rs.light_state(solid((0, 255, 255))), "YELLOW")

    def test_dark_is_unknown(self):
        self.assertEqual(rs.light_state(solid((10, 10, 10))), "UNKNOWN")

    def test_empty_crop_is_unknown(self):
        self.assertEqual(rs.light_state(None), "UNKNOWN")
        self.assertEqual(rs.light_state(np.zeros((0, 0, 3), dtype=np.uint8)),
                         "UNKNOWN")


class TestBoxHelpers(unittest.TestCase):
    def test_clip_box_clamps(self):
        self.assertEqual(rs.clip_box([-5, -5, 50, 60], 40, 40), (0, 0, 40, 40))

    def test_clip_box_rejects_tiny(self):
        self.assertIsNone(rs.clip_box([10, 10, 12, 30], 100, 100))

    def test_select_traffic_light_prefers_confidence(self):
        low = rs.Detection(0, 0, 20, 40, 0.5, "traffic light")
        high = rs.Detection(30, 0, 55, 60, 0.9, "traffic light")
        other = rs.Detection(0, 0, 90, 90, 0.99, "speedbump")
        self.assertEqual(rs.select_traffic_light([low, other, high]), high)

    def test_select_traffic_light_none(self):
        self.assertIsNone(rs.select_traffic_light([]))
        self.assertIsNone(rs.select_traffic_light(
            [rs.Detection(0, 0, 50, 50, 0.9, "speedbump")]))


class TestTrafficLightDetector(unittest.TestCase):
    def test_keeps_only_traffic_lights(self):
        model = FakeModel([
            FakeBox([10, 10, 40, 90], 0.8, rs.TRAFFIC_CLASS),
            FakeBox([0, 0, 30, 30], 0.9, 0),          # person
        ])
        det = rs.TrafficLightDetector(model=model)
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        found = det.detect(frame)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].label, "traffic light")
        self.assertAlmostEqual(found[0].confidence, 0.8)

    def test_drops_boxes_outside_frame(self):
        model = FakeModel([FakeBox([200, 200, 260, 260], 0.9, rs.TRAFFIC_CLASS)])
        det = rs.TrafficLightDetector(model=model)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(det.detect(frame), [])


class TestBumpDetector(unittest.TestCase):
    def test_detects_all_boxes(self):
        model = FakeModel([FakeBox([5, 5, 60, 60], 0.91, 0),
                           FakeBox([20, 70, 90, 95], 0.70, 0)])
        det = rs.BumpDetector(model=model)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        found = det.detect(frame)
        self.assertEqual([d.label for d in found], ["speedbump", "speedbump"])
        self.assertAlmostEqual(found[0].confidence, 0.91)

    def test_missing_weights_is_not_an_error(self):
        det = rs.BumpDetector(weights="definitely_missing.pt")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(det.load())
        self.assertFalse(det.ready)
        self.assertEqual(det.detect(frame), [])


class TestRedLightTimer(unittest.TestCase):
    def test_holds_while_red(self):
        timer = rs.RedLightTimer(hold_sec=2.0)
        self.assertEqual(timer.update("RED", 10.0), 0.0)
        self.assertAlmostEqual(timer.update("RED", 11.5), 1.5)
        self.assertFalse(timer.should_alert(1.5))
        self.assertTrue(timer.should_alert(timer.update("RED", 12.0)))

    def test_resets_when_light_changes(self):
        timer = rs.RedLightTimer(hold_sec=2.0)
        timer.update("RED", 0.0)
        self.assertEqual(timer.update("GREEN", 5.0), 0.0)
        self.assertEqual(timer.update("RED", 9.0), 0.0)


if __name__ == "__main__":
    unittest.main()
