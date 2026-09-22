# DrowsyGuard AI — DEPI Graduation Project

Smart driver monitoring + ADAS prototype by ENG. Mahmoud Abdo (CS Year 3, AI — Digital Egypt Pioneers).

**What it does**
- Face box + eye-closure % (PERCLOS via EAR) + HEAD DOWN detection
- HIGH / LOW risk + on-screen alert `Drowsiness Detected Stay Alert`
- Yawning detection (MAR) + distraction alert (`Look Ahead!`)
- Arduino bridge: buzzer + LEDs + motor brake + HC-SR04 auto-brake

## Quick start
Easiest (Windows, no terminal): install Python 3.10, double-click `setup.bat`, then double-click `run.bat`.
```bash
conda create -n drowsiness python=3.10 -y
conda activate drowsiness
pip install -r requirements.txt
python test_camera.py        # camera check, Q to quit
python drowsiness_phase0.py  # core: face + eye % + risk
python drowsiness_phase1.py  # + yawning + distraction
```
Keys: `Q` quit, `M` mute beep.

## Files
| File | Purpose |
|------|---------|
| `drowsiness_phase0.py` | Face box, EAR/PERCLOS %, head-down, HIGH/LOW, Stay Alert |
| `drowsiness_phase1.py` | + yawning (MAR) + distraction (head-turn) |
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
- [ ] Arduino hardware test (buzzer/LED/brake)
- [ ] YOLO: traffic light + speed bump
- [ ] Telegram alert + session report + demo video

## Demo
_(Coming soon: screenshot/GIF of live detection + demo video link)_
