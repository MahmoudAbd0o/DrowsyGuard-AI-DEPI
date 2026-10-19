"""Attention score: an indicative 0-100 meter derived from the indicators."""
from __future__ import annotations

import config


def attention_score(perclos: float | None,
                    head_down: bool,
                    distracted: bool,
                    recent_yawns: int,
                    yawning_now: bool,
                    state: str) -> float | None:
    """Return 0-100 attention, or ``None`` when PERCLOS is unknown.

    Eye closure is the main factor; head-down, head-turn and yawns add
    penalties. Caps keep the score consistent with the reported state.
    """
    if perclos is None:
        return None

    score = 100.0 - float(perclos)
    if head_down:
        score -= config.ATT_PENALTY_HEAD_DOWN
    if distracted:
        score -= config.ATT_PENALTY_DISTRACTED
    score -= config.ATT_PENALTY_YAWN * min(max(recent_yawns, 0), 2)
    if yawning_now:
        score -= config.ATT_PENALTY_YAWN_NOW

    if state.startswith("DROWSY"):
        score = min(score, config.ATT_CAP_DROWSY)
    elif state.startswith("DISTRACTED"):
        score = min(score, config.ATT_CAP_DISTRACTED)
    elif state.startswith("LOW"):
        score = min(score, config.ATT_CAP_LOW)

    return max(0.0, min(100.0, score))


def attention_level(score: float | None) -> tuple[str, str]:
    """Map a score to a (label, colour-name) pair."""
    if score is None:
        return ("--", "gray")
    if score >= config.ATT_HIGH:
        return ("High", "green")
    if score >= config.ATT_MED:
        return ("Medium", "orange")
    return ("Low", "red")
