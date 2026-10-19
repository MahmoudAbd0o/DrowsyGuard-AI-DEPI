"""Unit tests for driver-state decision logic."""
import unittest

from ai import drowsiness as ds


def evaluate(**kwargs):
    defaults = dict(perclos=5.0, head_down=False, distracted=False,
                    recent_yawns=0, yawning_now=False,
                    face_present=True, too_far=False, camera_error=False)
    defaults.update(kwargs)
    return ds.DecisionEngine().evaluate(**defaults)


class TestDecisionLogic(unittest.TestCase):
    def test_alert(self):
        self.assertEqual(evaluate().state, ds.ALERT)

    def test_no_face_is_unknown_not_alert(self):
        self.assertEqual(evaluate(face_present=False).state, ds.UNKNOWN)

    def test_missing_perclos_is_unknown(self):
        self.assertEqual(evaluate(perclos=None).state, ds.UNKNOWN)

    def test_face_too_small_is_unknown(self):
        self.assertEqual(evaluate(too_far=True).state, ds.UNKNOWN)

    def test_camera_error(self):
        self.assertEqual(evaluate(camera_error=True).state, ds.CAMERA_ERROR)

    def test_drowsy_by_perclos(self):
        self.assertEqual(evaluate(perclos=50.0).state, ds.DROWSY)

    def test_drowsy_by_head_down_with_closure(self):
        self.assertEqual(evaluate(perclos=25.0, head_down=True).state, ds.DROWSY)

    def test_head_down_without_closure_is_not_drowsy(self):
        self.assertEqual(evaluate(perclos=5.0, head_down=True).state, ds.ALERT)

    def test_drowsy_by_yawns(self):
        self.assertEqual(evaluate(perclos=5.0, recent_yawns=2).state, ds.DROWSY)

    def test_distracted(self):
        self.assertEqual(evaluate(perclos=5.0, distracted=True).state, ds.DISTRACTED)

    def test_low_risk_by_perclos(self):
        self.assertEqual(evaluate(perclos=20.0).state, ds.LOW_RISK)

    def test_low_risk_by_yawning(self):
        self.assertEqual(evaluate(perclos=5.0, yawning_now=True).state, ds.LOW_RISK)

    def test_priority_drowsy_over_distracted(self):
        self.assertEqual(evaluate(perclos=60.0, distracted=True).state, ds.DROWSY)

    def test_unknown_is_not_reported_as_safe(self):
        decision = evaluate(face_present=False)
        self.assertNotEqual(decision.state, ds.ALERT)


if __name__ == "__main__":
    unittest.main()
