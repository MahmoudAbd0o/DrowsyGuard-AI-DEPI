# Driver Drowsiness + ADAS — Professional Plan (Mahmoud)

## Status (tonight)
- [x] Env `drowsiness` (Python 3.10) created
- [x] Installed: opencv, mediapipe, numpy, pyserial, customtkinter
- [ ] Tomorrow AM: `pip install ultralytics` (torch ~800MB, needs good net/DNS 8.8.8.8)

## Run tomorrow
1. Camera test:
   `C:\Users\EL-Mostwred\anaconda3\envs\drowsiness\python.exe test_camera.py`
2. Phase 0 AI (face box + eye % + HEAD DOWN + HIGH/LOW + alert):
   `C:\Users\EL-Mostwred\anaconda3\envs\drowsiness\python.exe drowsiness_phase0.py`
   - Q quit, M mute. Tune in file: EAR_CLOSED_THRESH, PERCLOS_HIGH, HEAD_DOWN_THRESH, SERIAL_PORT.

## Architecture
Camera -> Laptop Python (MediaPipe + EAR/PERCLOS + YOLO later)
  -> Serial (D1/D0/N) -> Arduino Uno -> Buzzer + LEDs + L298N motor (brake)
  + HC-SR04 distance auto-brake + MQ-3 alcohol + MPU6050 bump + OLED status

## Shopping (professional ~1300-1500 EGP)
1. Smart car chassis 4WD + L298N (~700) — the look
2. HC-SR04 x2 (~120) — safe distance
3. MQ-3 alcohol (~250) — car won't start if drunk (wins points)
4. MPU6050 (~150) — bump/pothole from vibration
5. OLED 0.96 (~250) — in-car display
6. Buzzer + LEDs (~20) — alarm
7. ESP32-CAM last (~450, optional) — on-car camera instead of laptop

Buy order: 1+2+6 first, then 3+4+5.

## Phase 1 (vision only, no hardware)
- Tune EAR/PERCLOS on your face, glasses, low light
- Add yawning (mouth ratio) + distraction (gaze left/right)

## Phase 2 (Arduino)
- Upload arduino_bridge.ino, set SERIAL_PORT (check Device Manager -> COMx)
- Test: cover eyes -> buzzer + red LED + motor stop

## Exceptions to implement (judges love this)
- No face -> "No Face Detected", no false alarm
- 2 people -> track largest face (driver) only
- Camera fail -> clean error, no crash
- Arduino unplugged -> vision keeps running, warning on screen
- Dark / sunglasses -> note + test; night preprocessing later

## Notes
- DNS was failing tonight (ISP). Fixed with 8.8.8.8. If pip fails: retry morning.
- Quadro T2000 + i9 + 16GB is more than enough for nano models realtime.
