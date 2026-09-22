"""
DrowsyGuard AI - Phase 1: drowsiness + yawning + distraction
Run:
  C:\\Users\\EL-Mostwred\\anaconda3\\envs\\drowsiness\\python.exe drowsiness_phase1.py
Keys: Q quit, M mute
"""
from collections import deque
import time

import cv2
import mediapipe as mp
import numpy as np

EAR_CLOSED_THRESH = 0.21
PERCLOS_WINDOW = 60
PERCLOS_HIGH = 40.0
HEAD_DOWN_THRESH = 0.55
MAR_YAWN_THRESH = 0.60   # mouth open ratio -> yawning
YAWN_MIN_FRAMES = 12     # must stay open this long = real yawn
YAWN_ALERT_COUNT = 2     # yawns in 90s -> drowsy contributor
TURN_THRESH = 0.18       # head-turn ratio -> distracted
TURN_MIN_FRAMES = 30

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
DRIVER_ANCHOR = 0.30
SIDE_BIAS = 0.5

mp_face = mp.solutions.face_mesh


def ear(pts):
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def main():
    ear_hist = deque(maxlen=PERCLOS_WINDOW)
    muted = False
    last_beep = 0.0
    yawn_open = 0
    yawns = deque(maxlen=16)  # timestamps
    turn_frames = 0
    turn_hist = deque(maxlen=10)
    turn_clear = 0
    distract_ready_at = 0.0

    mesh = mp_face.FaceMesh(max_num_faces=2, refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found.")
        return
    print("Phase 1 running. Q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        status, color = "No Face", (160, 160, 160)
        perclos, avg_ear, mar = 0.0, 0.0, 0.0
        head_down, yawning_now, distracted = False, False, False

        if res.multi_face_landmarks:
            def _score(f):
                xs = [p.x for p in f.landmark]
                ys = [p.y for p in f.landmark]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                return area - SIDE_BIAS * abs((max(xs) + min(xs)) / 2.0 - DRIVER_ANCHOR)
            fl = max(res.multi_face_landmarks, key=_score)
            for _o in res.multi_face_landmarks:
                if _o is fl:
                    continue
                _ox = [p.x * w for p in _o.landmark]
                _oy = [p.y * h for p in _o.landmark]
                cv2.rectangle(frame, (int(min(_ox)), int(min(_oy))),
                              (int(max(_ox)), int(max(_oy))), (160, 160, 160), 1)
                cv2.putText(frame, "Passenger - ignored", (int(min(_ox)), int(min(_oy)) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)
            lm = np.array([(p.x * w, p.y * h) for p in fl.landmark])
            x1, y1, x2, y2 = int(lm[:, 0].min()), int(lm[:, 1].min()), \
                int(lm[:, 0].max()), int(lm[:, 1].max())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            avg_ear = (ear(lm[LEFT_EYE]) + ear(lm[RIGHT_EYE])) / 2.0
            ear_hist.append(1 if avg_ear < EAR_CLOSED_THRESH else 0)
            perclos = 100.0 * sum(ear_hist) / max(1, len(ear_hist))

            # MAR: mouth height |13-14| / width |78-308|
            mh = np.linalg.norm(lm[13] - lm[14])
            mw = np.linalg.norm(lm[78] - lm[308])
            mar = mh / (mw + 1e-6)
            if mar > MAR_YAWN_THRESH:
                yawn_open += 1
            else:
                if yawn_open >= YAWN_MIN_FRAMES:
                    yawns.append(time.time())
                    print(f"[yawn] #{len([t for t in yawns if time.time()-t < 90])} MAR={mar:.2f}")
                yawn_open = 0
            yawning_now = yawn_open >= YAWN_MIN_FRAMES
            recent_yawns = len([t for t in yawns if time.time() - t < 90])

            # head down
            eye_y = (lm[33][1] + lm[263][1]) / 2.0
            ratio = (lm[1][1] - eye_y) / max(1.0, (lm[152][1] - eye_y))
            head_down = ratio > HEAD_DOWN_THRESH

            # head turn (distraction): nose x offset vs face center
            face_cx = (x1 + x2) / 2.0
            face_w = max(1.0, x2 - x1)
            turn = abs(lm[1][0] - face_cx) / face_w
            turn_hist.append(turn)
            smooth = sum(turn_hist) / len(turn_hist)
            if smooth > 0.22:
                turn_frames += 1
                turn_clear = 0
            elif smooth < 0.12:
                turn_clear += 1
                if turn_clear >= 15:
                    turn_frames = 0
                    if distracted:
                        distract_ready_at = time.time() + 3.0
            distracted = turn_frames >= 45 and time.time() >= distract_ready_at

            drowsy_eye = perclos >= PERCLOS_HIGH or (head_down and perclos > 20)
            drowsy_yawn = recent_yawns >= YAWN_ALERT_COUNT
            if drowsy_eye or drowsy_yawn:
                status, color = "DROWSY - Stay Alert!", (0, 0, 255)
            elif distracted:
                status, color = "DISTRACTED - Look Ahead!", (0, 165, 255)
            elif perclos > 15 or yawning_now:
                status, color = "LOW RISK - Blinking/Yawn", (0, 255, 255)
            else:
                status, color = "ALERT - Safe", (0, 200, 0)

            for i in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, (int(lm[i][0]), int(lm[i][1])), 2, (255, 0, 0), -1)
        else:
            ear_hist.append(0)

        cv2.putText(frame, f"Eye closed: {perclos:.0f}%  EAR:{avg_ear:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR:{mar:.2f} Yawns(90s):{len([t for t in yawns if time.time()-t < 90])}", (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status, (20, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if head_down and res.multi_face_landmarks:
            cv2.putText(frame, "HEAD DOWN", (20, 148),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if yawning_now:
            cv2.putText(frame, "YAWNING...", (20, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        if distracted:
            cv2.putText(frame, "LOOK AHEAD!", (20, 214),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        if status.startswith("DROWSY") and not muted and time.time() - last_beep > 1.0:
            print("\a[ALERT] DROWSY", flush=True)
            last_beep = time.time()

        cv2.imshow("DrowsyGuard AI - Phase 1", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q')):
            break
        if k in (ord('m'), ord('M')):
            muted = not muted

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
