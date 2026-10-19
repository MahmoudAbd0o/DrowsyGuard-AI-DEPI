"""Unit tests for driver selection (no camera / MediaPipe required)."""
import unittest

import numpy as np

from vision.face_detector import select_driver


def box(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)


class TestSelectDriver(unittest.TestCase):
    def test_no_faces(self):
        self.assertIsNone(select_driver([], 200, 200))

    def test_prefers_driver_seat(self):
        left = box(10, 10, 60, 60)
        right = box(120, 10, 170, 60)
        self.assertEqual(select_driver([left, right], 200, 200), 1)

    def test_larger_face_wins_near_same_position(self):
        small = box(120, 10, 150, 40)
        big = box(110, 5, 180, 70)
        self.assertEqual(select_driver([small, big], 200, 200), 1)

    def test_resolution_independent(self):
        left = box(10, 10, 60, 60)
        right = box(120, 10, 170, 60)
        small_frame = select_driver([left, right], 200, 200)
        big_frame = select_driver([left * 2, right * 2], 400, 400)
        self.assertEqual(small_frame, big_frame)

    def test_invalid_frame_size_defaults_to_first(self):
        self.assertEqual(select_driver([box(0, 0, 10, 10)], 0, 0), 0)


if __name__ == "__main__":
    unittest.main()
