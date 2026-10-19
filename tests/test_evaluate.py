"""Unit tests for evaluation metrics."""
import unittest

from data import evaluate as ev


class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        self.y_true = ["DROWSY", "DROWSY", "ALERT", "ALERT", "ALERT"]
        self.y_pred = ["DROWSY", "ALERT", "ALERT", "DROWSY", "ALERT"]
        self.cm = ev.confusion_matrix(self.y_true, self.y_pred)

    def test_matrix_counts(self):
        self.assertEqual(self.cm, {"tp": 1, "fp": 1, "tn": 2, "fn": 1})

    def test_precision(self):
        self.assertAlmostEqual(ev.precision(self.cm), 0.5)

    def test_recall(self):
        self.assertAlmostEqual(ev.recall(self.cm), 0.5)

    def test_f1(self):
        self.assertAlmostEqual(ev.f1_score(self.cm), 0.5)

    def test_false_positive_rate(self):
        self.assertAlmostEqual(ev.false_positive_rate(self.cm), 1.0 / 3.0)

    def test_accuracy(self):
        self.assertAlmostEqual(ev.accuracy(self.cm), 3.0 / 5.0)

    def test_evaluate_bundle(self):
        metrics = ev.evaluate(self.y_true, self.y_pred)
        self.assertEqual(metrics["samples"], 5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertIn("false_positive_rate", metrics)


class TestEdgeCases(unittest.TestCase):
    def test_empty_inputs(self):
        cm = ev.confusion_matrix([], [])
        self.assertEqual(cm, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        self.assertEqual(ev.precision(cm), 0.0)
        self.assertEqual(ev.recall(cm), 0.0)
        self.assertEqual(ev.f1_score(cm), 0.0)
        self.assertEqual(ev.accuracy(cm), 0.0)

    def test_perfect_classifier(self):
        cm = ev.confusion_matrix(["DROWSY", "ALERT"], ["DROWSY", "ALERT"])
        self.assertAlmostEqual(ev.precision(cm), 1.0)
        self.assertAlmostEqual(ev.recall(cm), 1.0)
        self.assertAlmostEqual(ev.f1_score(cm), 1.0)


class TestCoverage(unittest.TestCase):
    def test_coverage(self):
        self.assertAlmostEqual(ev.coverage([True, True, False, True]), 0.75)

    def test_empty_coverage(self):
        self.assertEqual(ev.coverage([]), 0.0)


class TestDetectionLatency(unittest.TestCase):
    def test_latency_per_event(self):
        times = [0, 1, 2, 3, 4, 5]
        true = [0, 1, 1, 0, 1, 1]
        pred = [0, 0, 1, 1, 0, 1]
        self.assertEqual(ev.detection_latencies(times, true, pred), [1, 1])

    def test_missed_event_is_skipped(self):
        times = [0, 1, 2, 3]
        true = [0, 1, 1, 0]
        pred = [0, 0, 0, 0]
        self.assertEqual(ev.detection_latencies(times, true, pred), [])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ev.detection_latencies([0, 1], [0, 1], [0])


class TestStateCounts(unittest.TestCase):
    def test_counts(self):
        counts = ev.state_counts(["ALERT", "ALERT", "DROWSY"])
        self.assertEqual(counts, {"ALERT": 2, "DROWSY": 1})


if __name__ == "__main__":
    unittest.main()
