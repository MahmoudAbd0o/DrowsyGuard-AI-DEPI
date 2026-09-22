"""
Driver Drowsiness Detection - Phase 0 (no hardware needed)
- Face box via MediaPipe FaceMesh (468 landmarks)
- Eye closure % (PERCLOS) via EAR
- Head-down detection (nose vs eye line)
- Risk HIGH/LOW + on-screen alert
- Optional serial to Arduino: sends 'D1' drowsy / 'D0' alert / 'N' no-face

Run:
  C:\\Users\\EL-Mostwred\\anaconda3\\envs\\drowsiness\\python.exe drowsiness_phase0.py
Keys: Q quit, M mute buzzer-beep (console only in Phase 0)
"""
from collections import deque
import math
import time

import cv2
import mediapipe as mp
import numpy as np

try:
    import serial
except ImportError:
    serial = None

# ---- Config ----
EAR_CLOSED_THRESH = 0.21      # below this => eye considered closed
PERCLOS_WINDOW = 60           # frames (~2 sec at 30fps)
PERCLOS_HIGH = 40.0           # % closed -> HIGH risk
HEAD_DOWN_THRESH = 0.55       # tuned ratio, increase if too sensitive
SERIAL_PORT = "COM5"          # change to your Arduino port later
SERIAL_BAUD = 9600

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils


def ear(pts):
    # pts: 6x2 array. EAR = (|p2-p6|+|p3-p5|) / (2*|p1-p4|)
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def open_serial():
    if serial is None:
        return None
    try:
        s = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)  # Arduino reset delay
        print(f"[serial] connected {SERIAL_PORT}")
        return s
    except Exception as e:
        print(f"[serial] not connected ({e}). Running vision-only.")
        return None


def main():
    ser = open_serial()
    ear_hist = deque(maxlen=PERCLOS_WINDOW)
    muted = False
    last_beep = 0.0

    face_mesh = mp_face.FaceMesh(
        max_num_faces=2, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found.")
        return

    print("Running. Q=quit, M=mute")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARN: frame drop")
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        status, color = "No Face", (160, 160, 160)
        perclos = 0.0
        avg_ear = 0.0
        head_down = False

        if res.multi_face_landmarks:
            # pick largest face = driver (exception: multiple people)
            def area(fl):
                xs = [p.x for p in fl.landmark]
                ys = [p.y for p in fl.landmark]
                return (max(xs) - min(xs)) * (max(ys) - min(ys))
            fl = max(res.multi_face_landmarks, key=area)
            lm = np.array([(p.x * w, p.y * h) for p in fl.landmark])

            xs, ys = lm[:, 0], lm[:, 1]
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            le = ear(lm[LEFT_EYE])
            re = ear(lm[RIGHT_EYE])
            avg_ear = (le + re) / 2.0
            ear_hist.append(1 if avg_ear < EAR_CLOSED_THRESH else 0)
            closed = sum(ear_hist)
            perclos = 100.0 * closed / max(1, len(ear_hist))

            # head-down: nose tip (1) drops far below eye line
            eye_y = (lm[33][1] + lm[263][1]) / 2.0
            nose_y = lm[1][1]
            chin_y = lm[152][1]
            ratio = (nose_y - eye_y) / max(1.0, (chin_y - eye_y))
            head_down = ratio > HEAD_DOWN_THRESH

            if perclos >= PERCLOS_HIGH or (head_down and perclos > 20):
                status, color = "DROWSY - Stay Alert!", (0, 0, 255)
            elif perclos > 15:
                status, color = "LOW RISK - Blinking", (0, 255, 255)
            else:
                status, color = "ALERT - Safe", (0, 200, 0)

            # eye dots (like the demo video)
            for i in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, (int(lm[i][0]), int(lm[i][1])), 2, (255, 0, 0), -1)
        else:
            ear_hist.append(0)

        # HUD
        cv2.putText(frame, f"Eye closed: {perclos:.0f}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, status, (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if head_down and res.multi_face_landmarks:
            cv2.putText(frame, "HEAD DOWN", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if status.startswith("DROWSY"):
            cv2.putText(frame, "Drowsiness Detected Stay Alert", (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if not muted and time.time() - last_beep > 1.0:
                print("\a[ALERT] DROWSY", flush=True)  # console beep
                last_beep = time.time()

        # serial to Arduino (exception-safe: never crash if unplugged)
        if ser is not None:
            try:
                if not res.multi_face_landmarks:
                    ser.write(b"N")
                elif status.startswith("DROWSY"):
                    ser.write(b"1")
                else:
                    ser.write(b"0")
            except Exception as e:
                print(f"[serial] write failed: {e}")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None

        cv2.imshow("Driver Drowsiness - Phase 0", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q')):
            break
        if k in (ord('m'), ord('M')):
            muted = not muted

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
