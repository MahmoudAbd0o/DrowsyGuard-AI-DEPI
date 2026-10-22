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
│   ├── face_detector.py     # MediaPipe Face Mesh + driver selection
│   └── road_scene.py        # traffic light + speed-bump detection (YOLO)
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
├── tools/
│   ├── prepare_bump_dataset.py  # filter bump class + split + data.yaml
│   ├── simulate_session.py      # scripted session log for the report
│   └── make_report.py           # charts + report/report.md
├── report/                  # generated figures + Evaluation Report
├── tests/                   # 99 unit tests (unittest)
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
driver selection, road-scene detection, logging, evaluation metrics and
report helpers.

## Evaluation (Track 3)
```python
from data.evaluate import evaluate_csv
metrics = evaluate_csv("labeled_session.csv",
                       true_col="ground_truth", pred_col="prediction")
print(metrics)   # precision, recall, f1, false_positive_rate, accuracy
```
`detection_latencies(times, true_flags, pred_flags)` returns the delay between
each event onset and its first correct detection.

## Speed-bump model (`bump_model.pt`)
`bump_model.pt` (6.35 MB, YOLOv8n) is committed to the repo and loaded by
`vision/road_scene.py`; `yolo_phase2.py` runs it live next to the traffic-light
model. Validation on 277 images / 305 instances: **P 98.7%, R 97.4%,
mAP@50 99.4%, mAP@50-95 71.6%**.

To retrain it:

**Option A — Kaggle (free GPU, recommended)**
1. Open the [Urban Road Hazard dataset](https://www.kaggle.com/datasets/shuaifanpi/urban-road-hazard-datasetyolo-format) → **New Notebook**
2. Settings → Accelerator `GPU T4` · Internet `On`
3. `File → Import Notebook → Link` with
   `https://raw.githubusercontent.com/MahmoudAbd0o/DrowsyGuard-AI-DEPI/main/notebooks/train_bump_kaggle.ipynb`
4. **Run All** (~45 min) → download `bump_model.zip` → extract `bump_model.pt`

**Option B — local**
```bash
python tools/prepare_bump_dataset.py "path/to/Urban Road Hazard Dataset" --out datasets/bump
python train_bump.py --data datasets/bump/data.yaml --epochs 50 --batch 8
```
Local GPU training needs a CUDA build of PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu121`).

## Evaluation report
`tools/make_report.py` writes `report/report.md` plus charts
(`report/figures/`): training metrics, PERCLOS/EAR timeline, state
distribution, and - when test images are supplied - bump precision/recall.
```bash
python tools/simulate_session.py       # controlled session log (scripted input,
                                       # real PerclosMeter + DecisionEngine)
python tools/make_report.py            # rebuild report/report.md

# optional: field test the bump model on your own images (+ YOLO labels)
python tools/make_report.py --images data/bump_test/images --labels data/bump_test/labels
```

## Files
| File | Purpose |
|------|---------|
| `dashboard.py` | Live app: video, states, EAR/MAR/PERCLOS, attention, graph, log |
| `drowsiness_phase0.py` | Face box, EAR/PERCLOS %, head-down, HIGH/LOW, Stay Alert |
| `drowsiness_phase1.py` | + yawning (MAR) + distraction (head-turn) |
| `yolo_phase2.py` | YOLO traffic light + speed-bump model (logic in `vision/road_scene.py`) |
| `test_camera.py` | Camera sanity check |
| `tools/make_report.py` | Evaluation charts + `report/report.md` |
| `arduino_bridge/arduino_bridge.ino` | Buzzer/LED/motor + ultrasonic auto-brake |
| `TESTING.md` | 13 team test cases + bug template |
| `PROJECT_PLAN.md` | Roadmap + pro shopping list (~1300 EGP) |

## Team testing
See `TESTING.md`: 13 cases (eyes, yawn x2, look-away, no-face, 2 people, glasses, low light, unplug camera/Arduino, passenger). Report Pass/Fail + EAR/MAR values.

## Roadmap
- [x] Phase 0: face + eye % + risk + alert
- [x] Phase 1: yawning + distraction
- [x] Dashboard (live % + graph + status + graduated alarm)
- [x] Modular architecture + 99 automated tests
- [ ] Arduino hardware test (buzzer/LED/brake)
- [x] YOLO: traffic light + speed bump (`bump_model.pt`, mAP@50 99.4%)
- [x] Evaluation report (`report/report.md`)
- [ ] Telegram alert + demo video

## Demo
_(Coming soon: screenshot/GIF of live detection + demo video link)_
