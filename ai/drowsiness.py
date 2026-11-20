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


class DecisionStabilizer:
    """Turn a per-frame decision into a stable, alarm-safe state.

    The raw engine is fast but twitchy: PERCLOS crossing 40% for a single
    frame used to mean an instant red alarm. This wrapper adds the time
    dimension the physics needs:

    * **Confirmation** - DROWSY must persist ``DROWSY_CONFIRM_SEC`` before it
      is published. In the meantime the driver sees LOW RISK (yellow, soft
      beep), which is the graded warning.
    * **Hysteresis** - once confirmed, DROWSY is only released after PERCLOS
      stays below ``DROWSY_EXIT_PERCLOS`` for ``DROWSY_HOLD_SEC``, so the state
      cannot flicker SAFE/DROWSY/SAFE.
    * **No evidence, no alarm** - losing the face clears everything.
    """

    def __init__(self,
                 confirm_sec: float | None = None,
                 hold_sec: float | None = None,
                 exit_perclos: float | None = None):
        self.confirm_sec = float(config.DROWSY_CONFIRM_SEC
                                 if confirm_sec is None else confirm_sec)
        self.hold_sec = float(config.DROWSY_HOLD_SEC
                              if hold_sec is None else hold_sec)
        self.exit_perclos = float(config.DROWSY_EXIT_PERCLOS
                                  if exit_perclos is None else exit_perclos)
        self.reset()

    def reset(self) -> None:
        self._pending_since: float | None = None
        self._drowsy = False
        self._release_since: float | None = None

    @property
    def confirming(self) -> bool:
        """True while a drowsy candidate is waiting for confirmation."""
        return self._pending_since is not None and not self._drowsy

    @property
    def holding(self) -> bool:
        """True while a confirmed DROWSY state is being held."""
        return self._drowsy

    def apply(self, decision: Decision, perclos: float | None, now: float) -> Decision:
        """Return the decision that should actually be shown / sent."""
        if decision.state in (UNKNOWN, CAMERA_ERROR):
            # Without a trustworthy face there is no evidence for an alarm.
            self.reset()
            return decision

        if decision.state == DROWSY:
            self._release_since = None
            if not self._drowsy:
                if self._pending_since is None:
                    self._pending_since = now
                waited = now - self._pending_since
                if waited < self.confirm_sec:
                    return Decision(
                        LOW_RISK,
                        f"confirming drowsiness {waited:.1f}/{self.confirm_sec:.1f}s "
                        f"({decision.reason})",
                    )
                self._drowsy = True
            return decision

        # --- the raw decision is no longer DROWSY ---
        self._pending_since = None
        if not self._drowsy:
            return decision

        if perclos is not None and perclos < self.exit_perclos:
            if self._release_since is None:
                self._release_since = now
            if now - self._release_since >= self.hold_sec:
                self.reset()
                return decision
        else:
            self._release_since = None

        pct = "--" if perclos is None else f"{perclos:.0f}%"
        return Decision(DROWSY,
                        f"holding drowsy (PERCLOS {pct}, exits below "
                        f"{self.exit_perclos:.0f}%)")


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
