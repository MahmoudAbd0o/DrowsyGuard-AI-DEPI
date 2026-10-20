"""
Serial protocol between the Python application and the Arduino.

The Arduino firmware (arduino_bridge.ino) accepts exactly five commands:
    '0'  safe        -> green LED, motor running, silent
    '1'  low risk    -> yellow LED, soft beep, vibration 30%
    '2'  DROWSY      -> red LED, loud alarm, vibration 100%, motor STOP
    '3'  distracted  -> yellow blink, double beep, vibration 60%
    'N'  no face     -> yellow slow blink, silent, motor running

This module is the single place where driver states become wire commands,
so the mapping can never drift between the dashboard and the firmware.
"""
from __future__ import annotations

import config
from ai.drowsiness import (ALERT, CAMERA_ERROR, DISTRACTED, DROWSY,
                           LOW_RISK, UNKNOWN, level_for_state)

VALID_COMMANDS = frozenset({"0", "1", "2", "3", "N"})

# Human-readable description of each command (documentation + tests).
COMMAND_MEANING = {
    "0": "safe",
    "1": "low risk",
    "2": "drowsy",
    "3": "distracted",
    "N": "no face / unknown",
}


def is_valid_command(command: str) -> bool:
    """True when the Arduino firmware would accept this command."""
    return command in VALID_COMMANDS


def command_for_state(state: str) -> str:
    """Map a driver state to a wire command, always returning a valid one."""
    command = level_for_state(state)
    return command if command in VALID_COMMANDS else "N"


class LevelDebouncer:
    """Require a level to hold for N frames before it is published.

    Prevents alarm flicker when the raw level oscillates frame to frame.
    """

    def __init__(self, frames: int | None = None):
        self.frames = int(config.LEVEL_DEBOUNCE_FRAMES if frames is None else frames)
        self._candidate: str | None = None
        self._count = 0
        self.stable = "0"

    def update(self, level: str) -> str:
        if level == self._candidate:
            self._count += 1
        else:
            self._candidate = level
            self._count = 1
        if self._count >= self.frames:
            self.stable = level
        return self.stable

    def reset(self) -> None:
        self._candidate = None
        self._count = 0
        self.stable = "0"
