# DrowsyGuard AI — Session Handoff (Mahmoud resumes here)

## Who / What
- Owner: ENG. Mahmoud Abdo, CS Year 3 AI, DEPI (Rowad Misr الرقمية)
- Repo: https://github.com/MahmoudAbd0o/DrowsyGuard-AI-DEPI (main, clean)
- Stack: Python 3.10 + conda env `drowsiness`, Arduino Uno (later), smart-car chassis (to buy)

## Locked decisions (verified by testing, do NOT change casually)
- mediapipe==0.10.9 (0.10.33/0.10.35/1.x removed `solutions` → code breaks; tested 22/09)
- Team members on Python 3.11 must use a 3.10 conda env, NOT bump requirements
- requirements.txt = light (no torch). YOLO extras only in requirements-yolo.txt
- Easiest team run: setup.bat + run.bat (one-click, Windows)

## Thresholds (tuned live 22/09)
- EAR_CLOSED_THRESH=0.21, PERCLOS_WINDOW=60, PERCLOS_HIGH=40
- MAR_YAWN_THRESH=0.60, YAWN_MIN_FRAMES=12, YAWN_ALERT_COUNT=2 (90s)
- Distraction: smooth>0.22 enter (45 frames), <0.12 exit (15 frames), 3s cooldown
- HEAD_DOWN_THRESH=0.55, driver seat anchor left (Egypt) DRIVER_ANCHOR=0.30, SIDE_BIAS=0.5
- Alarm debounce: level stable 5 frames; beeps LOW 800Hz/2s, DISTRACTED 1000Hz double/1.2s, DROWSY 1500Hz/0.8s
- Serial levels to Arduino: 0 safe / 1 low / 3 distracted / 2 drowsy / N no-face; COM5 default
- Arduino: buzzer 8, ledR 7, ledG 6, ledY 5, trig 9, echo 10, ENA 3, IN1 4, IN2 2, VIB 11, SAFE_CM 30

## Done (all pushed)
Phase 0+1 vision, Dashboard (video+graph+log+autosave 60s+report CSV), graduated alarm+serial,
driver-seat lock + passenger-ignore, guards (camera fallback 0/1/2, night CLAHE<40, too-far<0.02,
vision try/except, serial whitelist), TESTING.md (13 cases), PROJECT_PLAN.md, setup/run .bat

## Next (Mahmoud picks on return)
1. YOLO traffic-light/speed-bump (requirements-yolo.txt ready, torch ~800MB)
2. Arduino hardware test (buy: chassis+L298N, HC-SR04 x2, MQ-3, MPU6050, OLED, vibration motor ~40EGP, buzzer/LEDs)
3. Telegram alert + demo video + final report

## After Windows reinstall
1. Install Python 3.10 (PATH checked) + Git
2. git clone repo → setup.bat → run.bat
3. First push re-login GitHub in browser (Sign in with browser)
4. Paste this file into a new chat to resume with full context.
