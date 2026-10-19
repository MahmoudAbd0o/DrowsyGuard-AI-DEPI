"""Unit tests for timestamp-based PERCLOS and missing-data handling."""
import unittest

from ai.perclos import PerclosMeter


def meter(**kwargs):
    defaults = dict(window_sec=10.0, min_obs=2, min_coverage=0.0)
    defaults.update(kwargs)
    return PerclosMeter(**defaults)


class TestPerclos(unittest.TestCase):
    def test_insufficient_samples_returns_none(self):
        m = meter(min_obs=5)
        m.update(0.0, True)
        m.update(1.0, True)
        self.assertIsNone(m.value())
        self.assertEqual(m.samples, 2)

    def test_basic_percentage(self):
        m = meter()
        m.update(0.0, True)
        m.update(1.0, True)
        m.update(2.0, False)
        self.assertAlmostEqual(m.value(), 200.0 / 3.0, places=6)

    def test_missing_values_are_not_counted_as_open_eyes(self):
        m = meter()
        m.update(0.0, True)
        m.update(1.0, True)
        m.update(2.0, None)      # invalid / no face
        m.update(3.0, None)
        # value must be 100% (only the two closed samples count)
        self.assertAlmostEqual(m.value(), 100.0, places=6)

    def test_coverage(self):
        m = meter()
        m.update(0.0, True)
        m.update(1.0, None)
        m.update(2.0, None)
        m.update(3.0, False)
        self.assertAlmostEqual(m.coverage, 0.5, places=6)

    def test_low_coverage_returns_none(self):
        m = meter(min_coverage=0.75)
        for i in range(10):
            m.update(float(i), None)
        m.update(10.0, True)
        m.update(11.0, True)
        self.assertIsNone(m.value())

    def test_window_prunes_old_samples(self):
        m = meter(window_sec=10.0, min_obs=1)
        m.update(0.0, True)
        m.update(1.0, True)
        m.update(20.0, False)          # everything older than 10s is pruned
        self.assertEqual(m.samples, 1)
        self.assertAlmostEqual(m.value(), 0.0, places=6)

    def test_reset(self):
        m = meter()
        m.update(0.0, True)
        m.reset()
        self.assertEqual(m.samples, 0)
        self.assertEqual(m.coverage, 0.0)
        self.assertIsNone(m.value())


if __name__ == "__main__":
    unittest.main()
