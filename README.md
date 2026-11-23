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
python dashboard.py          # full dashboard (or: python main.py)
```
Keys: `Q` quit, `M` mute beep, `R` start/stop recording,
`C` re-run eye calibration.

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
│   ├── calibration.py       # personal eye calibration + EAR median filter
│   ├── perclos.py           # timestamp-based PERCLOS + coverage
│   ├── attention.py         # 0–100 attention score
│   └── drowsiness.py        # driver states + decision engine + stabiliser
├── hardware/
│   └── protocol.py          # Python <-> Arduino commands + debounce
├── data/
│   ├── logger.py            # session measurements + event log
│   └── evaluate.py          # precision / recall / F1 / FPR / latency
├── tools/
│   ├── prepare_bump_dataset.py  # filter bump class + split + data.yaml
│   ├── simulate_session.py      # scripted session log for the report
│   ├── make_report.py           # charts + report/report.md
│   └── test_arduino.py          # sends 0/1/2/3/N to the car, no camera
├── report/                  # generated figures + Evaluation Report
├── tests/                   # 121 unit tests (unittest)
├── outputs/                 # generated logs & reports (gitignored)
├── arduino_bridge/arduino_bridge.ino
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

## Arduino / car firmware
`arduino_bridge/arduino_bridge.ino` is a drop-in for the team's Bluetooth car:
L298N motors (D2-D5), steering servo (D9), HC-SR04 (D6/D7), MPU6050 (I2C),
green/white/red LEDs (A0-A2), buzzer (A3), HC-06 on D10/D11.

- Camera commands (`0/1/2/3/N`) are read from **Bluetooth and USB** at the same time.
- Driving commands `S` (arm/stop) and `F/B/L/R` still work from the phone app.
- `2` brakes immediately and cancels an automatic turn; no camera byte for
  3 s falls back to `N` (the dashboard sends a heartbeat every second).
- Bench test without the camera: `python tools/test_arduino.py COM7`
  (the car brakes during the `2` step, so lift the wheels).

## Missing-data policy
- Invalid measurements are `None`, never `0`.
- Missing eye measurements are **not** counted as open eyes.
- PERCLOS uses timestamps, not frame counts.
- The dashboard shows `--` until enough valid samples exist.
- Coverage (valid / total) is logged with every frame.

## False alarms and distance (v2 tuning)
Two field problems were fixed without adding hardware:

1. **Personal eye calibration** (`ai/calibration.py`) - the first 3 seconds
   measure the driver's own open-eye EAR and derive the closed threshold
   (`baseline x 0.60`, clamped to 0.13-0.26). Press `C` to redo it.
2. **EAR validity gate** - a turned head (`head_turn_ratio > 0.22`) or a face
   narrower than 110 px makes the eye measurement `None` instead of "closed".
   Looking at the screen can no longer fake a closure.
3. **Median filter** over the last 5 EAR samples - one bad frame is ignored.
4. **Time-based confirmation** (`ai.drowsiness.DecisionStabilizer`) - DROWSY
   must hold 1.2 s before the red alarm (the driver sees a yellow warning
   first) and is only released below 25% PERCLOS + 0.8 s hold.
5. **1280x720 capture + centre-zoom retry** - when no face is found, the middle
   of the frame is cropped and re-detected once (`ZOOM_FACTOR = 2`), and the
   screen shows `MOVE CLOSER TO THE CAMERA` instead of failing silently.

Every decision now writes its reason to the screen and to the `reason` column
of the session CSV, so a false alarm can be diagnosed after the fact.

## Tests
```bash
python -m unittest discover -s tests -t . -v
```
Covers: features, PERCLOS + missing data, decision logic + stabiliser,
eye calibration, EAR smoothing, zoom geometry, serial protocol,
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
| `yolo_phase2.py` | YOLO traffic light + speed-bump model (logic in `vision/road_scene.py`) |
| `tools/make_report.py` | Evaluation charts + `report/report.md` |
| `arduino_bridge/arduino_bridge.ino` | Car firmware: buzzer/LED/motor/servo + ultrasonic auto-brake + camera alerts |
| `TESTING.md` | 13 team test cases + bug template |
| `PROJECT_PLAN.md` | Roadmap + pro shopping list (~1300 EGP) |

## Team testing
See `TESTING.md`: 13 cases (eyes, yawn x2, look-away, no-face, 2 people, glasses, low light, unplug camera/Arduino, passenger). Report Pass/Fail + EAR/MAR values.

## Roadmap
- [x] Phase 0: face + eye % + risk + alert
- [x] Phase 1: yawning + distraction
- [x] Dashboard (live % + graph + status + graduated alarm)
- [x] Modular architecture + 121 automated tests
- [ ] Arduino hardware test (buzzer/LED/brake)
- [x] YOLO: traffic light + speed bump (`bump_model.pt`, mAP@50 99.4%)
- [x] Evaluation report (`report/report.md`)
- [ ] Telegram alert + demo video

## Demo
_(Coming soon: screenshot/GIF of live detection + demo video link)_
