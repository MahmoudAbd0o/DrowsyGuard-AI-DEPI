"""
Generate a reproducible demo session log for the evaluation report.

The camera input is *scripted*, but everything else is real: the same
``PerclosMeter``, ``DecisionEngine`` and ``attention_score`` used by the
dashboard produce the states. Use this when no recording with drowsy events is
available, and label it clearly as a controlled simulation in the report.

Usage:
    python tools/simulate_session.py
    python tools/simulate_session.py --seconds 120 --out outputs/simulation_log.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ai.attention import attention_score
from ai.drowsiness import DecisionEngine
from ai.perclos import PerclosMeter
from data.logger import MeasurementLogger

FPS = 20.0

# (start, end, phase)
PHASES = [
    (0.0, 16.0, "alert"),
    (16.0, 28.0, "distract"),
    (28.0, 38.0, "alert"),
    (38.0, 50.0, "drowsy"),
    (50.0, 58.0, "alert"),
    (58.0, 66.0, "noface"),
    (66.0, 74.0, "alert"),
    (74.0, 84.0, "yawn"),
    (84.0, 95.0, "alert"),
]

YAWN_BURSTS = [(76.0, 79.0)]


def phase_at(t: float) -> str:
    for start, end, name in PHASES:
        if start <= t < end:
            return name
    return "alert"


def eyes_closed(t: float, phase: str) -> bool:
    """Blink schedule: short blinks normally, long closures while drowsy."""
    if phase == "drowsy":
        return (t % 2.0) < 1.4          # ~70% closure -> high PERCLOS
    return (t % 4.0) < 0.15             # ~4% closure -> normal blinking


def yawning(t: float) -> bool:
    return any(start <= t < end for start, end in YAWN_BURSTS)


def simulate(seconds: float = 90.0, seed: int = 7):
    rng = random.Random(seed)
    meter = PerclosMeter()
    engine = DecisionEngine()

    frames: list[dict] = []
    yawn_times: list[float] = []
    yawn_open_frames = 0
    mouth_was_open = False

    steps = int(seconds * FPS)
    for i in range(steps):
        t = i / FPS
        phase = phase_at(t)
        now_yawning = yawning(t) and phase == "yawn"
        face = phase != "noface"

        # --- yawn counter: same rule as the dashboard (hold + count) ---
        if now_yawning:
            yawn_open_frames += 1
            mouth_was_open = True
        elif mouth_was_open:
            if yawn_open_frames >= config.YAWN_MIN_FRAMES:
                yawn_times.append(t)
            yawn_open_frames = 0
            mouth_was_open = False
        yawn_times = [ts for ts in yawn_times if t - ts <= config.YAWN_WINDOW_SEC]

        closed = eyes_closed(t, phase) if face else None
        perclos = meter.update(t, closed)

        distracted = face and phase == "distract"
        head_down = face and phase == "drowsy" and t >= 46.0
        decision = engine.evaluate(
            perclos=perclos,
            head_down=head_down,
            distracted=distracted,
            recent_yawns=len(yawn_times),
            yawning_now=now_yawning,
            face_present=face,
        )
        attention = attention_score(perclos, head_down, distracted,
                                    len(yawn_times), now_yawning, decision.state)

        if face:
            ear = (0.13 if closed else 0.30) + rng.uniform(-0.012, 0.012)
        else:
            ear = None
        mar = (0.72 if now_yawning else 0.08) + rng.uniform(-0.02, 0.02)
        coverage = 1.0 if face else 0.0

        frames.append(dict(
            timestamp=t,
            state=decision.state,
            level=decision.level,
            perclos=perclos,
            ear=ear,
            mar=mar if face else None,
            attention=attention,
            head_down=head_down,
            distracted=distracted,
            yawning=now_yawning,
            face_present=face,
            coverage=coverage,
        ))
    return frames


def write_log(frames, path: Path) -> None:
    logger = MeasurementLogger(str(path))
    for row in frames:
        logger.log(**row)
    logger.close()


def write_phases(path: Path, seconds: float) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("start", "end", "phase"))
        for start, end, name in PHASES:
            if start < seconds:
                writer.writerow((start, min(end, seconds), name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="outputs/simulation_log.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = simulate(args.seconds, args.seed)
    write_log(frames, out)
    write_phases(out.with_name("simulation_phases.csv"), args.seconds)

    counts: dict[str, int] = {}
    for row in frames:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    print(f"wrote {len(frames)} frames -> {out}")
    for state, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {state:<12} {count:5d}  ({100 * count / len(frames):.1f}%)")


if __name__ == "__main__":
    main()
