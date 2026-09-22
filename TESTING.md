# DrowsyGuard AI — Team Testing Guide (Phase 0+1)

## Setup (5 min)
```bash
# with Anaconda
conda create -n drowsiness python=3.10 -y
conda activate drowsiness
pip install -r requirements.txt
# without Anaconda
pip install -r requirements.txt
```

## Run
```bash
python test_camera.py        # Q to quit, expect: your face on screen
python drowsiness_phase0.py  # face box + eye % + HIGH/LOW + Stay Alert
python drowsiness_phase1.py  # + yawning + distraction
```

## Test cases (report Pass/Fail + notes)
| # | Test | Expected |
|---|------|----------|
| 1 | Open eyes, look ahead | ALERT - Safe, eye % near 0 |
| 2 | Close eyes 3 sec | DROWSY - Stay Alert!, % rises to 60+ |
| 3 | Head down | HEAD DOWN label |
| 4 | Big yawn (mouth wide) | YAWNING + yawns counter +1 |
| 5 | 2 yawns in 90s | DROWSY even with eyes open |
| 6 | Look left/right 3s | DISTRACTED - Look Ahead! |
| 7 | No face (cover camera) | No Face, no false alarm |
| 8 | 2 people in frame | Tracks largest face only |
| 9 | Glasses on/off | Still works, note any drop |
| 10 | Low light | Note accuracy drop |
| 11 | Unplug camera mid-run 30s | "Camera lost! Reconnecting..." in log, auto-recovers, no crash |
| 12 | Unplug Arduino mid-run | Vision keeps running + warning in log, no crash |
| 13 | Passenger next to driver | Driver = green box, passenger = gray "ignored", alarm only for driver |

## Bug report template
- Name / laptop / camera:
- Test # + what you did:
- What happened vs expected:
- Screenshot (optional) + EAR/MAR values on screen:
