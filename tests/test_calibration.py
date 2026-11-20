"""Unit tests for eye calibration, EAR smoothing and decision stabilisation."""
import unittest

import numpy as np

import config
from ai import drowsiness as ds
from ai.calibration import EarSmoother, EyeCalibrator, eye_gate
from ai.drowsiness import Decision, DecisionStabilizer
from vision.face_detector import centre_crop, remap_landmarks


def drowsy(reason="PERCLOS 60% >= 40%"):
    return Decision(ds.DROWSY, reason)


def alert():
    return Decision(ds.ALERT, "eyes open, head forward")


class TestEyeCalibrator(unittest.TestCase):
    def test_derives_threshold_from_open_eye(self):
        calib = EyeCalibrator(duration=1.0, min_samples=5)
        for i in range(12):
            calib.feed(i * 0.1, 0.30)
        self.assertTrue(calib.done)
        self.assertAlmostEqual(calib.baseline, 0.30, places=6)
        # 0.30 * 0.60 = 0.18 (inside the clamp)
        self.assertAlmostEqual(calib.threshold, 0.18, places=6)

    def test_threshold_is_clamped(self):
        high = EyeCalibrator(duration=1.0, min_samples=5)
        for i in range(12):
            high.feed(i * 0.1, 0.95)
        self.assertAlmostEqual(high.threshold, config.EAR_CLOSED_MAX, places=6)

        low = EyeCalibrator(duration=1.0, min_samples=5)
        for i in range(12):
            low.feed(i * 0.1, 0.19)
        self.assertAlmostEqual(low.threshold, config.EAR_CLOSED_MIN, places=6)

    def test_falls_back_to_default_without_enough_samples(self):
        calib = EyeCalibrator(duration=1.0, min_samples=8)
        calib.feed(0.0, 0.30)
        calib.feed(1.5, 0.30)
        self.assertTrue(calib.done)
        self.assertIsNone(calib.baseline)
        self.assertAlmostEqual(calib.threshold, config.EAR_CLOSED_THRESH, places=6)

    def test_blinks_do_not_pollute_the_baseline(self):
        calib = EyeCalibrator(duration=1.0, min_samples=4)
        for i in range(8):
            calib.feed(i * 0.1, 0.30)
        for i in range(4):
            calib.feed(1.0 + i * 0.1, 0.05)     # below MIN_BASELINE -> ignored
        self.assertTrue(calib.done)
        self.assertAlmostEqual(calib.baseline, 0.30, places=6)

    def test_does_not_start_without_a_face(self):
        calib = EyeCalibrator(duration=0.5, min_samples=3)
        for i in range(20):
            calib.feed(i * 0.1, None)
        self.assertFalse(calib.active)
        self.assertFalse(calib.done)

    def test_is_closed(self):
        calib = EyeCalibrator(duration=1.0, min_samples=4)
        for i in range(10):
            calib.feed(i * 0.1, 0.30)
        self.assertTrue(calib.is_closed(0.10))
        self.assertFalse(calib.is_closed(0.30))
        self.assertIsNone(calib.is_closed(None))

    def test_progress(self):
        calib = EyeCalibrator(duration=2.0, min_samples=5)
        self.assertEqual(calib.progress, 0.0)
        calib.feed(10.0, 0.30)
        calib.feed(11.0, 0.30)
        self.assertAlmostEqual(calib.progress, 0.5, places=6)
        calib.feed(12.0, 0.30)
        self.assertEqual(calib.progress, 1.0)


class TestEarSmoother(unittest.TestCase):
    def test_single_bad_frame_is_ignored(self):
        smooth = EarSmoother(window=5)
        for value in (0.30, 0.31, 0.29, 0.30):
            smooth.update(value)
        self.assertGreater(smooth.update(0.05), 0.25)

    def test_none_is_not_smeared(self):
        smooth = EarSmoother(window=5)
        smooth.update(0.30)
        self.assertIsNone(smooth.update(None))
        self.assertIsNone(smooth.update(None))

    def test_reset(self):
        smooth = EarSmoother(window=3)
        smooth.update(0.30)
        smooth.reset()
        self.assertEqual(smooth.samples, 0)


class TestDecisionStabilizer(unittest.TestCase):
    def setUp(self):
        self.stabilizer = DecisionStabilizer(confirm_sec=1.0, hold_sec=0.5,
                                            exit_perclos=25.0)

    def test_drowsy_needs_confirmation(self):
        self.assertEqual(self.stabilizer.apply(drowsy(), 60.0, 0.0).state,
                         ds.LOW_RISK)
        self.assertEqual(self.stabilizer.apply(drowsy(), 60.0, 0.6).state,
                         ds.LOW_RISK)
        self.assertEqual(self.stabilizer.apply(drowsy(), 60.0, 1.1).state,
                         ds.DROWSY)

    def test_hysteresis_keeps_drowsy(self):
        for t in (0.0, 1.1):
            self.stabilizer.apply(drowsy(), 60.0, t)
        self.assertEqual(self.stabilizer.apply(alert(), 20.0, 1.2).state, ds.DROWSY)
        self.assertEqual(self.stabilizer.apply(alert(), 20.0, 1.4).state, ds.DROWSY)
        self.assertEqual(self.stabilizer.apply(alert(), 20.0, 1.9).state, ds.ALERT)

    def test_high_perclos_does_not_release(self):
        for t in (0.0, 1.1):
            self.stabilizer.apply(drowsy(), 60.0, t)
        # 30% is below PERCLOS_HIGH but still above the exit threshold
        self.assertEqual(self.stabilizer.apply(alert(), 30.0, 1.5).state, ds.DROWSY)
        self.assertEqual(self.stabilizer.apply(alert(), 30.0, 5.0).state, ds.DROWSY)

    def test_losing_the_face_clears_everything(self):
        for t in (0.0, 1.1):
            self.stabilizer.apply(drowsy(), 60.0, t)
        unknown = Decision(ds.UNKNOWN, "no face detected")
        self.assertEqual(self.stabilizer.apply(unknown, None, 1.2).state, ds.UNKNOWN)
        # a fresh episode starts from scratch
        self.assertEqual(self.stabilizer.apply(drowsy(), 60.0, 1.3).state,
                         ds.LOW_RISK)

    def test_alert_is_passed_through_when_not_drowsy(self):
        self.assertEqual(self.stabilizer.apply(alert(), 5.0, 0.0).state, ds.ALERT)


class TestZoomGeometry(unittest.TestCase):
    def test_centre_crop(self):
        self.assertEqual(centre_crop(1280, 720, 2.0), (320, 180, 640, 360))

    def test_no_zoom_is_full_frame(self):
        self.assertEqual(centre_crop(640, 480, 1.0), (0, 0, 640, 480))

    def test_remap_round_trip(self):
        face = np.array([[100.0, 50.0], [600.0, 350.0]])
        mapped = remap_landmarks([face], 320, 180, 640, 360, 1280, 720)
        self.assertEqual(len(mapped), 1)
        self.assertAlmostEqual(mapped[0][0, 0], 370.0, places=6)
        self.assertAlmostEqual(mapped[0][0, 1], 205.0, places=6)
        self.assertAlmostEqual(mapped[0][1, 0], 620.0, places=6)
        self.assertAlmostEqual(mapped[0][1, 1], 355.0, places=6)

    def test_remap_does_not_mutate_input(self):
        face = np.array([[10.0, 10.0]])
        remap_landmarks([face], 100, 100, 50, 50, 100, 100)
        self.assertAlmostEqual(face[0, 0], 10.0, places=6)


class TestFalseAlarmRegression(unittest.TestCase):
    """End-to-end pipeline checks for the two reported field problems."""

    def _run(self, samples, gate=False):
        from ai.perclos import PerclosMeter
        from ai.drowsiness import DecisionEngine

        calib = EyeCalibrator(duration=2.0, min_samples=10)
        smooth = EarSmoother()
        meter = PerclosMeter()
        engine = DecisionEngine()
        stabilizer = DecisionStabilizer()
        states = []
        for timestamp, ear, turn, width in samples:
            reason = eye_gate(width, turn, ear) if gate else "ok"
            smoothed = smooth.update(ear if reason == "ok" else None)
            if not calib.done:
                calib.feed(timestamp, smoothed)
            closed = calib.is_closed(smoothed) if smoothed is not None else None
            perclos = meter.update(timestamp, closed)
            decision = engine.evaluate(perclos=perclos, face_present=True,
                                       too_far=reason == "face too small")
            states.append(stabilizer.apply(decision, perclos, timestamp).state)
        return states, calib

    def test_looking_away_never_alarms(self):
        fps, rows = 30.0, []
        for i in range(int(11 * fps)):
            t = i / fps
            if t < 7.0:
                rows.append((t, 0.55, 0.05, 300))   # eyes open, facing forward
            else:
                rows.append((t, 0.10, 0.40, 300))   # glancing at the screen
        states, calib = self._run(rows, gate=True)
        self.assertTrue(calib.done)
        during = states[int(7 * fps):]
        self.assertNotIn(ds.DROWSY, during)
        self.assertNotIn(ds.LOW_RISK, during)

    def test_real_closure_still_alarms(self):
        fps, rows = 30.0, []
        for i in range(int(11 * fps)):
            t = i / fps
            rows.append((t, 0.55 if t < 7.0 else 0.08, 0.05, 300))
        states, _ = self._run(rows, gate=True)
        self.assertIn(ds.DROWSY, states[int(7 * fps):])

    def test_far_face_is_unknown_not_drowsy(self):
        fps, rows = 30.0, []
        for i in range(int(11 * fps)):
            t = i / fps
            rows.append((t, 0.55 if t < 7.0 else 0.05, 0.05, 60))
        states, _ = self._run(rows, gate=True)
        during = states[int(7 * fps):]
        self.assertNotIn(ds.DROWSY, during)
        self.assertIn(ds.UNKNOWN, during)


if __name__ == "__main__":
    unittest.main()
