"""Unit tests for the Python <-> Arduino serial protocol."""
import unittest

from ai import drowsiness as ds
from hardware import protocol


class TestProtocol(unittest.TestCase):
    def test_drowsy_maps_to_2(self):
        # Regression: phase0 used to send '1' for DROWSY, but the firmware
        # treats '2' as maximum danger. This must never regress.
        self.assertEqual(protocol.command_for_state(ds.DROWSY), "2")

    def test_all_states_map_to_valid_commands(self):
        for state in ds.ALL_STATES:
            command = protocol.command_for_state(state)
            self.assertIn(command, protocol.VALID_COMMANDS, msg=state)

    def test_expected_mapping(self):
        self.assertEqual(protocol.command_for_state(ds.ALERT), "0")
        self.assertEqual(protocol.command_for_state(ds.LOW_RISK), "1")
        self.assertEqual(protocol.command_for_state(ds.DROWSY), "2")
        self.assertEqual(protocol.command_for_state(ds.DISTRACTED), "3")
        self.assertEqual(protocol.command_for_state(ds.UNKNOWN), "N")
        self.assertEqual(protocol.command_for_state(ds.CAMERA_ERROR), "N")

    def test_unknown_state_defaults_to_n(self):
        self.assertEqual(protocol.command_for_state("SOMETHING ELSE"), "N")

    def test_valid_commands(self):
        for ch in "0123N":
            self.assertTrue(protocol.is_valid_command(ch))
        for ch in ("4", "n", "", "X"):
            self.assertFalse(protocol.is_valid_command(ch))


class TestLevelDebouncer(unittest.TestCase):
    def test_requires_hold_frames(self):
        deb = protocol.LevelDebouncer(frames=3)
        self.assertEqual(deb.update("2"), "0")
        self.assertEqual(deb.update("2"), "0")
        self.assertEqual(deb.update("2"), "2")

    def test_flicker_is_ignored(self):
        deb = protocol.LevelDebouncer(frames=3)
        deb.update("2")
        deb.update("2")
        deb.update("0")          # single flicker resets the candidate
        self.assertEqual(deb.update("2"), "0")

    def test_reset(self):
        deb = protocol.LevelDebouncer(frames=2)
        deb.update("2")
        deb.update("2")
        self.assertEqual(deb.stable, "2")
        deb.reset()
        self.assertEqual(deb.stable, "0")


if __name__ == "__main__":
    unittest.main()
