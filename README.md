# DrowsyGuard AI — DEPI Graduation Project

Smart driver monitoring + ADAS prototype by ENG. Mahmoud Abdo (CS Year 3, AI — Digital Egypt Pioneers).

**What it does**
- Face box + eye-closure % (PERCLOS via EAR) + HEAD DOWN detection
- Driver states: `ALERT` / `LOW RISK` / `DISTRACTED` / `DROWSY` / `UNKNOWN`
- Yawning detection (MAR) + distraction alert (`Look Ahead!`)
- Attention score (0–100) with a live meter
- Arduino bridge: buzzer + LEDs + motor brake + HC-SR04 auto-brake
- Timestamp-based PERCLOS + missing-data handling + session logging

## Quick start
Easiest (Windows, no terminal): install Python 3.10, double-click `setup.bat`, then double-click `run.bat`.
```bash
conda create -n drowsiness python=3.10 -y
conda activate drowsiness
pip install -r requirements.txt
python test_camera.py        # camera check, Q to quit
python main.py               # full dashboard (or: python dashboard.py)
python drowsiness_phase0.py  # minimal: face + eye % + risk
python drowsiness_phase1.py  # + yawning + distraction
```
Keys: `Q` quit (phase scripts), `M` mute beep, `R` start/stop recording (dashboard).

## Project structure
```text
DrowsyGuard-AI-DEPI-main/
├── main.py                  # entry point -> dashboard
├── config.py                # every threshold in one place
├── dashboard.py             # CustomTkinter app (live video + stats + graph)
├── vision/
│   ├── camera.py            # capture, warm-up, auto-reconnect
│   └── face_detector.py     # MediaPipe Face Mesh + driver selection
├── ai/
│   ├── features.py          # EAR / MAR / head pose (pure functions)
│   ├── perclos.py           # timestamp-based PERCLOS + coverage
│   ├── attention.py         # 0–100 attention score
│   └── drowsiness.py        # driver states + rule-based decision engine
├── hardware/
│   └── protocol.py          # Python <-> Arduino commands + debounce
├── data/
│   ├── logger.py            # session measurements + event log
│   └── evaluate.py          # precision / recall / F1 / FPR / latency
├── tests/                   # 68 unit tests (unittest)
├── outputs/                 # generated logs & reports (gitignored)
├── arduino_bridge/arduino_bridge.ino
├── drowsiness_phase0.py
├── drowsiness_phase1.py
├── yolo_phase2.py
├── TESTING.md
└── PROJECT_PLAN.md
```

## Driver states → Arduino commands
| State | Meaning | Serial |
|---|---|---|
| `ALERT` | eyes open, head forward | `0` |
| `LOW RISK` | blinking / yawning | `1` |
| `DROWSY` | PERCLOS high, head down, or repeated yawns | `2` |
| `DISTRACTED` | head turned away from the road | `3` |
| `UNKNOWN` | no face / face too small / not enough data | `N` |

`UNKNOWN` is deliberately **not** reported as `ALERT`: a missing face never
means the driver is safe.

## Missing-data policy
- Invalid measurements are `None`, never `0`.
- Missing eye measurements are **not** counted as open eyes.
- PERCLOS uses timestamps, not frame counts.
- The dashboard shows `--` until enough valid samples exist.
- Coverage (valid / total) is logged with every frame.

## Tests
```bash
python -m unittest discover -s tests -t . -v
```
Covers: features, PERCLOS + missing data, decision logic, serial protocol,
driver selection, logging and evaluation metrics.

## Evaluation (Track 3)
```python
from data.evaluate import evaluate_csv
metrics = evaluate_csv("labeled_session.csv",
                       true_col="ground_truth", pred_col="prediction")
print(metrics)   # precision, recall, f1, false_positive_rate, accuracy
```
`detection_latencies(times, true_flags, pred_flags)` returns the delay between
each event onset and its first correct detection.

## Files
| File | Purpose |
|------|---------|
| `dashboard.py` | Live app: video, states, EAR/MAR/PERCLOS, attention, graph, log |
| `drowsiness_phase0.py` | Face box, EAR/PERCLOS %, head-down, HIGH/LOW, Stay Alert |
| `drowsiness_phase1.py` | + yawning (MAR) + distraction (head-turn) |
| `yolo_phase2.py` | YOLO traffic light + optional speed-bump model |
| `test_camera.py` | Camera sanity check |
| `arduino_bridge/arduino_bridge.ino` | Buzzer/LED/motor + ultrasonic auto-brake |
| `TESTING.md` | 13 team test cases + bug template |
| `PROJECT_PLAN.md` | Roadmap + pro shopping list (~1300 EGP) |

## Team testing
See `TESTING.md`: 13 cases (eyes, yawn x2, look-away, no-face, 2 people, glasses, low light, unplug camera/Arduino, passenger). Report Pass/Fail + EAR/MAR values.

## Roadmap
- [x] Phase 0: face + eye % + risk + alert
- [x] Phase 1: yawning + distraction
- [x] Dashboard (live % + graph + status + graduated alarm)
- [x] Modular architecture + 68 automated tests
- [ ] Arduino hardware test (buzzer/LED/brake)
- [ ] YOLO: traffic light + speed bump
- [ ] Telegram alert + final evaluation report + demo video

## Demo
_(Coming soon: screenshot/GIF of live detection + demo video link)_
