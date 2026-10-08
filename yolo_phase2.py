"""
DrowsyGuard AI - Phase 2 (YOLO): traffic light + speed bump detection
Vision-only, no hardware needed.

Run:
  pip install -r requirements-yolo.txt   # one time, heavy (~800MB torch)
  python yolo_phase2.py
Keys: Q quit

- Traffic light: pretrained yolov8n (COCO class 9) + color state (red/yellow/green)
- Speed bump: custom model file `bump_model.pt` (Roboflow) if present, else skipped
"""
import sys
import time

try:
    import cv2
except ImportError:
    raise SystemExit("Missing opencv-python. Run: pip install -r requirements.txt")

try:
    import numpy as np
except ImportError:
    raise SystemExit("Missing numpy. Run: pip install -r requirements.txt")

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Missing ultralytics. Run: pip install -r requirements-yolo.txt")

YOLO_MODEL = "yolov8n.pt"   # COCO pretrained (traffic light = class 9)
BUMP_MODEL_FILE = "bump_model.pt"  # custom Roboflow model, optional
TRAFFIC_CLASS = 9
CONF_THRESH = 0.45
RED_ALERT_SEC = 2.0  # red light held this long -> STOP alert


def light_state(crop):
    # Dominant color inside the traffic-light box (BGR frame crop).
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255])) | \
        cv2.inRange(hsv, np.array([160, 90, 90]), np.array([180, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 90, 90]), np.array([40, 255, 255]))
    green = cv2.inRange(hsv, np.array([45, 70, 70]), np.array([90, 255, 255]))
    scores = {"RED": cv2.countNonZero(red),
              "YELLOW": cv2.countNonZero(yellow),
              "GREEN": cv2.countNonZero(green)}
    best = max(scores, key=scores.get)
    return best if scores[best] > 20 else "UNKNOWN"


def main():
    print(f"Loading {YOLO_MODEL} ...")
    model = YOLO(YOLO_MODEL)
    bump_model = None
    try:
        bump_model = YOLO(BUMP_MODEL_FILE)
        print(f"Bump model loaded ({BUMP_MODEL_FILE}).")
    except Exception:
        print(f"No {BUMP_MODEL_FILE} - bump detection skipped "
              f"(traffic lights only).")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found.")
        return
    print("Phase 2 running. Q=quit")

    red_since = None
    last_beep = 0.0
    fails = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails == 30 or (fails > 30 and fails % 30 == 0):
                print("Camera lost (sleep?) - reconnecting...")
                try:
                    cap.release()
                    cap = cv2.VideoCapture(0)
                except Exception:
                    pass
            if fails >= 150:
                print("ERROR: camera did not come back. Rerun the script.")
                break
            continue
        fails = 0
        h, w = frame.shape[:2]
        state, sconf = "NO LIGHT", 0.0
        bumps = []

        res = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]
        for b in res.boxes:
            if int(b.cls[0]) != TRAFFIC_CLASS:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            sconf = float(b.conf[0])
            state = light_state(frame[y1:y2, x1:x2])
            color = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255),
                     "GREEN": (0, 200, 0)}.get(state, (160, 160, 160))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{state} {sconf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            break  # nearest light only

        if bump_model is not None:
            bres = bump_model.predict(frame, conf=CONF_THRESH, verbose=False)[0]
            for b in bres.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                bumps.append(float(b.conf[0]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Bump {float(b.conf[0]):.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # STOP logic: red light held -> alert (brake hook for Arduino later)
        if state == "RED":
            if red_since is None:
                red_since = time.time()
            held = time.time() - red_since
            if held >= RED_ALERT_SEC:
                cv2.putText(frame, "STOP - Red Light!", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if time.time() - last_beep > 1.0:
                    print("\a[ALERT] RED LIGHT - STOP", flush=True)
                    last_beep = time.time()
        else:
            red_since = None

        cv2.putText(frame, f"Light: {state}  Bumps: {len(bumps)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("DrowsyGuard AI - Phase 2 (YOLO)", frame)
        if cv2.getWindowProperty("DrowsyGuard AI - Phase 2 (YOLO)",
                                 cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed via X
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 10):
        print(f"WARNING: Python {sys.version_info[0]}.{sys.version_info[1]} - "
              f"verified on 3.10.")
    main()
