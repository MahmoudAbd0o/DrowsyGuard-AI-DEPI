"""
DrowsyGuard AI - Phase 2 (YOLO): traffic light + speed bump detection.
Vision-only, no hardware needed.

Run:
  pip install -r requirements-yolo.txt   # one time, heavy (~800MB torch)
  python yolo_phase2.py
Keys: Q quit

Detection logic lives in ``vision.road_scene`` so it can be unit tested and
reused by the dashboard; this file is only the camera loop / drawing layer.
"""
import sys
import time

try:
    import cv2
except ImportError:
    raise SystemExit("Missing opencv-python. Run: pip install -r requirements.txt")

import config
from vision.road_scene import (BumpDetector, RedLightTimer, TrafficLightDetector,
                               light_colour, light_state)

CONF_THRESH = 0.45
RED_ALERT_SEC = 2.0


def draw_light(frame, detection, state):
    x1, y1, x2, y2 = detection.box
    color = light_colour(state)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{state} {detection.confidence:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_bump(frame, detection):
    x1, y1, x2, y2 = detection.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), config.BGR_GREEN, 2)
    cv2.putText(frame, f"Bump {detection.confidence:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.BGR_GREEN, 2)


def main():
    print("Loading yolov8n.pt ...")
    lights = TrafficLightDetector("yolov8n.pt", CONF_THRESH)
    lights.load()
    print("Traffic-light model loaded (yolov8n.pt).")

    bumps = BumpDetector("bump_model.pt", CONF_THRESH)
    if bumps.load() is not None:
        print("Bump model loaded (bump_model.pt).")
    else:
        print("No bump_model.pt - bump detection skipped (traffic lights only).")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found.")
        return
    print("Phase 2 running. Q=quit")

    red = RedLightTimer(RED_ALERT_SEC)
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

        state, sconf = "NO LIGHT", 0.0
        light = max(lights.detect(frame), key=lambda d: d.confidence, default=None)
        if light is not None:
            x1, y1, x2, y2 = light.box
            state = light_state(frame[y1:y2, x1:x2])
            sconf = light.confidence
            draw_light(frame, light, state)

        bump_list = bumps.detect(frame)
        for bump in bump_list:
            draw_bump(frame, bump)

        held = red.update(state, time.time())
        if red.should_alert(held):
            cv2.putText(frame, "STOP - Red Light!", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, config.BGR_RED, 2)
            if time.time() - last_beep > 1.0:
                print("\a[ALERT] RED LIGHT - STOP", flush=True)
                last_beep = time.time()

        cv2.putText(frame, f"Light: {state}  Bumps: {len(bump_list)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.BGR_WHITE, 2)
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
