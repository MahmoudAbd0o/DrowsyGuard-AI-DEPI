"""
Driver-state classification (rule-based decision logic).

The engine is deliberately conservative:
  * a missing face is UNKNOWN, never ALERT;
  * an invalid/insufficient eye measurement is UNKNOWN, never "eyes open";
  * UNKNOWN is kept distinct from ALERT and is mapped to a safe hardware
    level (no alarm), while still being visible to the user.
"""
from __future__ import annotations

from dataclasses import dataclass

import config

# --------------------------------------------------------------- states --
ALERT = "ALERT"
LOW_RISK = "LOW RISK"
DISTRACTED = "DISTRACTED"
DROWSY = "DROWSY"
UNKNOWN = "UNKNOWN"
CAMERA_ERROR = "CAMERA ERROR"

ALL_STATES = (ALERT, LOW_RISK, DISTRACTED, DROWSY, UNKNOWN, CAMERA_ERROR)

# Hardware command sent to the Arduino for each state.
# NOTE: DROWSY is '2' (the Arduino firmware treats '2' as maximum danger).
STATE_LEVEL = {
    ALERT: "0",
    LOW_RISK: "1",
    DROWSY: "2",
    DISTRACTED: "3",
    UNKNOWN: "N",
    CAMERA_ERROR: "N",
}


def level_for_state(state: str) -> str:
    """Serial command character for a driver state (defaults to 'N')."""
    return STATE_LEVEL.get(state, "N")


@dataclass(frozen=True)
class Decision:
    """Outcome of one evaluation."""
    state: str
    reason: str

    @property
    def level(self) -> str:
        return level_for_state(self.state)


class DecisionEngine:
    """Stateless rule evaluation for a single frame."""

    def evaluate(self,
                 *,
                 perclos: float | None,
                 head_down: bool = False,
                 distracted: bool = False,
                 recent_yawns: int = 0,
                 yawning_now: bool = False,
                 face_present: bool = True,
                 too_far: bool = False,
                 camera_error: bool = False) -> Decision:
        if camera_error:
            return Decision(CAMERA_ERROR, "camera unavailable")

        if not face_present:
            return Decision(UNKNOWN, "no face detected")

        if too_far:
            return Decision(UNKNOWN, "face too small - move closer")

        if perclos is None:
            return Decision(UNKNOWN, "insufficient eye measurements")

        drowsy_eyes = perclos >= config.PERCLOS_HIGH
        drowsy_head = head_down and perclos > 20.0
        drowsy_yawns = recent_yawns >= config.YAWN_ALERT_COUNT

        if drowsy_eyes or drowsy_head or drowsy_yawns:
            if drowsy_eyes:
                reason = f"PERCLOS {perclos:.0f}% >= {config.PERCLOS_HIGH:.0f}%"
            elif drowsy_head:
                reason = "head down with elevated eye closure"
            else:
                reason = f"{recent_yawns} yawns in window"
            return Decision(DROWSY, reason)

        if distracted:
            return Decision(DISTRACTED, "head turned away from the road")

        if perclos > config.PERCLOS_LOW or yawning_now:
            reason = "blinking / yawning" if yawning_now else f"PERCLOS {perclos:.0f}%"
            return Decision(LOW_RISK, reason)

        return Decision(ALERT, "eyes open, head forward")
