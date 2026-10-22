"""Unit tests for the evaluation-report helpers (no matplotlib output needed)."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import make_report as mr


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(mr.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(mr.iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_half_overlap(self):
        # intersection 5x10=50, union 100+100-50=150
        self.assertAlmostEqual(mr.iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)

    def test_zero_area(self):
        self.assertEqual(mr.iou((0, 0, 0, 0), (0, 0, 10, 10)), 0.0)


class TestLabels(unittest.TestCase):
    def test_parse_label_file_to_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text("0 0.5 0.5 0.2 0.4\n1 0.5 0.5 0.5 0.5\n",
                            encoding="utf-8")
            boxes = mr.parse_label_file(path, 100, 200)
            self.assertEqual(len(boxes), 1)  # class 1 ignored
            self.assertAlmostEqual(boxes[0][0], 40)
            self.assertAlmostEqual(boxes[0][1], 60)
            self.assertAlmostEqual(boxes[0][2], 60)
            self.assertAlmostEqual(boxes[0][3], 140)

    def test_missing_label_file_is_empty(self):
        self.assertEqual(mr.parse_label_file(Path("nope.txt"), 10, 10), [])

    def test_parse_skips_bad_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.txt"
            path.write_text("0 0.5 0.5\n0 x y w h\n", encoding="utf-8")
            self.assertEqual(mr.parse_label_file(path, 10, 10), [])


class TestMatching(unittest.TestCase):
    def test_perfect_match(self):
        preds = [(0, 0, 10, 10, 0.9)]
        gts = [(0, 0, 10, 10)]
        self.assertEqual(mr.match_boxes(preds, gts), (1, 0, 0))

    def test_false_positive_and_negative(self):
        preds = [(0, 0, 10, 10, 0.9), (50, 50, 60, 60, 0.8)]
        gts = [(0, 0, 10, 10), (80, 80, 90, 90)]
        self.assertEqual(mr.match_boxes(preds, gts), (1, 1, 1))

    def test_low_iou_counts_as_miss(self):
        preds = [(0, 0, 10, 10, 0.9)]
        gts = [(20, 20, 30, 30)]
        self.assertEqual(mr.match_boxes(preds, gts), (0, 1, 1))

    def test_each_ground_truth_used_once(self):
        preds = [(0, 0, 10, 10, 0.9), (0, 0, 10, 10, 0.8)]
        gts = [(0, 0, 10, 10)]
        self.assertEqual(mr.match_boxes(preds, gts), (1, 1, 0))

    def test_precision_recall_f1(self):
        p, r, f1 = mr.precision_recall_f1(tp=8, fp=2, fn=2)
        self.assertAlmostEqual(p, 0.8)
        self.assertAlmostEqual(r, 0.8)
        self.assertAlmostEqual(f1, 0.8)

    def test_metrics_without_data(self):
        self.assertEqual(mr.precision_recall_f1(0, 0, 0), (0.0, 0.0, 0.0))


class TestSessionSummary(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "timestamp": [0.0, 0.5, 1.0, 1.5],
            "state": ["ALERT", "DROWSY", "DROWSY", "UNKNOWN"],
            "perclos": [5.0, 50.0, None, None],
            "attention": [95.0, 30.0, None, None],
            "face_present": [1, 1, 0, 0],
        })

    def test_summary_numbers(self):
        summary = mr.summarize_session(self.frame())
        self.assertEqual(summary["frames"], 4)
        self.assertAlmostEqual(summary["duration_sec"], 1.5)
        self.assertAlmostEqual(summary["drowsy_pct"], 50.0)
        self.assertAlmostEqual(summary["unknown_pct"], 25.0)
        self.assertAlmostEqual(summary["perclos_mean"], 27.5)
        self.assertAlmostEqual(summary["perclos_max"], 50.0)
        self.assertAlmostEqual(summary["face_coverage"], 0.5)

    def test_empty_session_is_safe(self):
        summary = mr.summarize_session(pd.DataFrame({
            "timestamp": [], "state": [], "perclos": [],
            "attention": [], "face_present": [],
        }))
        self.assertEqual(summary["frames"], 0)
        self.assertEqual(summary["drowsy_pct"], 0.0)
        self.assertIsNone(summary["perclos_mean"])


if __name__ == "__main__":
    unittest.main()
