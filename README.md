# DrowsyGuard AI Project — DEPI

Smart driver monitoring + ADAS prototype:
- Drowsiness detection (MediaPipe FaceMesh + EAR/PERCLOS + head-down)
- Safe-distance auto-brake (HC-SR04)
- Alcohol check (MQ-3), bump detection (MPU6050), in-car OLED status
- Python AI brain + Arduino Uno actuators + Smart-car chassis

## Quick start
```bash
conda create -n drowsiness python=3.10 -y
conda activate drowsiness
pip install -r requirements.txt
python test_camera.py
python drowsiness_phase0.py   # Q quit, M mute
```

## Structure
- `drowsiness_phase0.py` — Phase 0 vision (face box, eye %, HIGH/LOW, Stay Alert)
- `test_camera.py` — camera check
- `arduino_bridge/arduino_bridge.ino` — buzzer/LED/motor + ultrasonic brake
- `PROJECT_PLAN.md` — roadmap + shopping list

## Team
Mahmoud — CS Year 3, AI — Digital Egypt Pioneers (Rowad Misr الرقمية)
