"""Unit tests for geometric feature extraction."""
import unittest

import numpy as np

import config
from ai import features


def empty_landmarks():
    return np.zeros((config.MIN_LANDMARKS, 2), dtype=float)


def put_points(landmarks, indices, points):
    for idx, point in zip(indices, points):
        landmarks[idx] = point


def open_eye_points():
    return [(0, 0), (2, 3), (4, 3), (10, 0), (4, -3), (2, -3)]


def closed_eye_points():
    return [(0, 0), (2, 0), (4, 0), (10, 0), (4, 0), (2, 0)]


class TestEar(unittest.TestCase):
    def test_open_eye(self):
        self.assertAlmostEqual(features.ear(open_eye_points()), 0.6, places=6)

    def test_closed_eye(self):
        self.assertAlmostEqual(features.ear(closed_eye_points()), 0.0, places=6)

    def test_rejects_nan(self):
        pts = np.array(open_eye_points(), dtype=float)
        pts[2, 1] = np.nan
        self.assertIsNone(features.ear(pts))

    def test_rejects_wrong_shape(self):
        self.assertIsNone(features.ear([(0, 0), (1, 1)]))

    def test_zero_width_returns_none(self):
        flat = [(5, 0), (5, 0), (5, 0), (5, 0), (5, 0), (5, 0)]
        self.assertIsNone(features.ear(flat))

    def test_average_ear(self):
        lm = empty_landmarks()
        put_points(lm, config.LEFT_EYE, open_eye_points())
        put_points(lm, config.RIGHT_EYE, open_eye_points())
        self.assertAlmostEqual(features.average_ear(lm), 0.6, places=6)

    def test_average_ear_invalid_landmarks(self):
        self.assertIsNone(features.average_ear(np.zeros((10, 2))))


class TestMar(unittest.TestCase):
    def test_mar_value(self):
        lm = empty_landmarks()
        lm[config.MOUTH_TOP] = (0, 0)
        lm[config.MOUTH_BOTTOM] = (0, 10)
        lm[config.MOUTH_LEFT] = (-10, 5)
        lm[config.MOUTH_RIGHT] = (10, 5)
        self.assertAlmostEqual(features.mar(lm), 0.5, places=6)

    def test_zero_width_returns_none(self):
        lm = empty_landmarks()
        lm[config.MOUTH_TOP] = (0, 0)
        lm[config.MOUTH_BOTTOM] = (0, 10)
        lm[config.MOUTH_LEFT] = (5, 5)
        lm[config.MOUTH_RIGHT] = (5, 5)
        self.assertIsNone(features.mar(lm))


class TestHeadPose(unittest.TestCase):
    def test_head_down_ratio(self):
        lm = empty_landmarks()
        lm[config.LEFT_EYE_CORNER] = (0, 0)
        lm[config.RIGHT_EYE_CORNER] = (100, 0)
        lm[config.NOSE_TIP] = (50, 60)
        lm[config.CHIN] = (50, 100)
        self.assertAlmostEqual(features.head_down_ratio(lm), 0.6, places=6)

    def test_head_down_degenerate_returns_none(self):
        lm = empty_landmarks()
        lm[config.LEFT_EYE_CORNER] = (0, 10)
        lm[config.RIGHT_EYE_CORNER] = (100, 10)
        lm[config.CHIN] = (50, 10)
        self.assertIsNone(features.head_down_ratio(lm))

    def test_head_turn_centred(self):
        lm = empty_landmarks()
        lm[0] = (0, 0)
        lm[config.MIN_LANDMARKS - 1] = (100, 100)
        lm[config.NOSE_TIP] = (50, 50)
        self.assertAlmostEqual(features.head_turn_ratio(lm), 0.0, places=6)

    def test_head_turn_off_centre(self):
        lm = empty_landmarks()
        lm[0] = (0, 0)
        lm[config.MIN_LANDMARKS - 1] = (100, 100)
        lm[config.NOSE_TIP] = (70, 50)
        self.assertAlmostEqual(features.head_turn_ratio(lm), 0.2, places=6)


class TestBoundingBox(unittest.TestCase):
    def test_face_bbox(self):
        lm = empty_landmarks()
        lm[:] = (10, 20)
        lm[config.MIN_LANDMARKS - 1] = (110, 220)
        self.assertEqual(features.face_bbox(lm), (10, 20, 110, 220))

    def test_face_fraction(self):
        lm = empty_landmarks()
        lm[0] = (0, 0)
        lm[config.MIN_LANDMARKS - 1] = (100, 100)
        self.assertAlmostEqual(features.face_fraction(lm, 200, 200), 0.25)


if __name__ == "__main__":
    unittest.main()
