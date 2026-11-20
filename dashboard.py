"""
DrowsyGuard AI - Dashboard (CustomTkinter)

Run:
  python dashboard.py

Architecture:
  vision.camera.Camera          -> video capture + reconnect
  vision.face_detector          -> MediaPipe landmarks + driver selection
  ai.features / ai.perclos      -> measurements
  ai.attention / ai.drowsiness  -> score + state
  hardware.protocol             -> serial commands + debounce
  data.logger                   -> session log + event log
"""
import sys
import time
from collections import deque
from datetime import datetime

try:
    import cv2
except ImportError:
    raise SystemExit("Missing opencv-python. Run: pip install -r requirements.txt")
try:
    import customtkinter as ctk
except ImportError:
    raise SystemExit("Missing customtkinter. Run: pip install -r requirements.txt")
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    raise SystemExit("Missing matplotlib. Run: pip install -r requirements.txt")
try:
    import mediapipe as mp  # noqa: F401  (checked here for a friendly error)
except ImportError:
    raise SystemExit("Missing mediapipe. Run: pip install -r requirements.txt")
try:
    import numpy as np  # noqa: F401
except ImportError:
    raise SystemExit("Missing numpy. Run: pip install -r requirements.txt")
try:
    from PIL import Image
except ImportError:
    raise SystemExit("Missing pillow. Run: pip install -r requirements.txt")

import config
from ai import drowsiness as ds
from ai import features
from ai.attention import attention_level, attention_score
from ai.calibration import EarSmoother, EyeCalibrator, eye_gate
from ai.drowsiness import DecisionEngine, DecisionStabilizer
from ai.perclos import PerclosMeter
from data.logger import EventLog, MeasurementLogger
from hardware.protocol import LevelDebouncer, command_for_state
from vision.camera import Camera
from vision.face_detector import (FaceDetector, centre_crop, remap_landmarks,
                                  select_driver)

try:
    import serial
except ImportError:
    serial = None

if sys.version_info[:2] != (3, 10):
    print(f"WARNING: Python {sys.version_info[0]}.{sys.version_info[1]} detected. "
          f"This project is verified on Python 3.10 (mediapipe==0.10.9). "
          f"Create env: conda create -n drowsiness python=3.10 -y")

# State -> (label shown on screen, colour name)
STATE_DISPLAY = {
    ds.ALERT: ("ALERT - Safe", "green"),
    ds.LOW_RISK: ("LOW RISK - Blinking/Yawn", "yellow"),
    ds.DISTRACTED: ("DISTRACTED - Look Ahead!", "orange"),
    ds.DROWSY: ("DROWSY - Stay Alert!", "red"),
    ds.UNKNOWN: ("UNKNOWN - No face / low data", "gray"),
    ds.CAMERA_ERROR: ("CAMERA ERROR", "gray"),
}

ctk.set_appearance_mode("dark")


class Dashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DrowsyGuard AI | ENG. Mahmoud Abdo")
        self.geometry("1220x800")

        # --- state / measurement objects ---
        self.perclos_meter = PerclosMeter()
        self.engine = DecisionEngine()
        self.stabilizer = DecisionStabilizer()
        self.ear_smoother = EarSmoother()
        self.calibrator = EyeCalibrator()
        self.debouncer = LevelDebouncer()
        self.graph_hist = deque(maxlen=config.GRAPH_LEN)
        self.yawns = deque(maxlen=16)
        self.yawn_open = 0
        self.turn_frames = 0
        self.turn_hist = deque(maxlen=10)
        self.turn_clear = 0
        self.distract_ready_at = 0.0
        self.drowsy_count = 0
        self.was_drowsy = False
        self.was_low_att = False
        self.muted = False
        self.running = True
        self.last_level = ""
        self.last_beep = 0.0
        self.ser = None
        self.vision_errors = 0
        self._last_reconnects = 0
        self.zoom = 1.0
        self.eye_gate = "no face"
        self.eye_valid = False
        self.too_far_hint = False

        # --- logging ---
        self.meas_log = MeasurementLogger()
        self.event_log = EventLog(config.SESSION_REPORT)

        # --- serial ---
        if serial is not None:
            try:
                self.ser = serial.Serial(config.SERIAL_PORT, config.SERIAL_BAUD, timeout=1)
                time.sleep(2)
                self._pending_log = f"Arduino linked on {config.SERIAL_PORT}."
            except Exception as e:
                self.ser = None
                self._pending_log = f"Arduino not found ({e}) - vision only."

        # --- vision ---
        self.detector = FaceDetector()
        self.cam = Camera()
        if not self.cam.open():
            raise SystemExit("No camera found (tried 0,1,2). Check cable/privacy settings.")
        self._pending_cam = f"Camera {self.cam.index} opened."

        self.last_autosave = time.time()
        self.recording = False
        self.writer = None
        self.bind("<KeyPress-r>", lambda e: self.toggle_rec())
        self.bind("<KeyPress-R>", lambda e: self.toggle_rec())
        self.bind("<KeyPress-c>", lambda e: self.recalibrate())
        self.bind("<KeyPress-C>", lambda e: self.recalibrate())
        self.last_dark_warn = 0.0

        # --- layout: video left, stats right ---
        self.video_lbl = ctk.CTkLabel(self, text="")
        self.video_lbl.grid(row=0, column=0, rowspan=8, padx=12, pady=12, sticky="nsew")

        self.status_lbl = ctk.CTkLabel(self, text="Starting...", font=("Arial", 24, "bold"))
        self.status_lbl.grid(row=0, column=1, padx=12, pady=(12, 4), sticky="ew")

        self.perc_lbl = ctk.CTkLabel(self, text="Eye closed: 0%", font=("Arial", 18))
        self.perc_lbl.grid(row=1, column=1, padx=12, sticky="ew")
        self.bar = ctk.CTkProgressBar(self)
        self.bar.set(0)
        self.bar.grid(row=2, column=1, padx=12, sticky="ew")

        self.ear_lbl = ctk.CTkLabel(self, text="EAR: -  MAR: -", font=("Arial", 16))
        self.ear_lbl.grid(row=3, column=1, padx=12, sticky="ew")
        self.count_lbl = ctk.CTkLabel(self, text="Drowsy: 0 | Yawns(90s): 0", font=("Arial", 16))
        self.count_lbl.grid(row=4, column=1, padx=12, sticky="ew")

        self.att_lbl = ctk.CTkLabel(self, text="Attention: --", font=("Arial", 18, "bold"))
        self.att_lbl.grid(row=5, column=1, padx=12, sticky="ew")
        self.att_bar = ctk.CTkProgressBar(self)
        self.att_bar.set(0)
        self.att_bar.grid(row=6, column=1, padx=12, sticky="ew")

        self.mute_btn = ctk.CTkButton(self, text="Mute: OFF", command=self.toggle_mute)
        self.mute_btn.grid(row=7, column=1, padx=12, pady=4, sticky="ew")

        # --- graph bottom-left ---
        self.fig = Figure(figsize=(6.6, 2.2), dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("Eye closure % (live)")
        self.line, = self.ax.plot([], [])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=8, column=0, padx=12, pady=8, sticky="nsew")

        # --- alert log bottom-right ---
        self.log = ctk.CTkTextbox(self, height=180)
        self.log.grid(row=8, column=1, padx=12, pady=8, sticky="nsew")
        self.log.insert("end", "Alert log ready.\n")
        self.log.insert("end", f"Python {sys.version_info[0]}.{sys.version_info[1]} (verified: 3.10).\n")
        if getattr(self, "_pending_cam", ""):
            self.log.insert("end", self._pending_cam + "\n")
        if getattr(self, "_pending_log", ""):
            self.log.insert("end", self._pending_log + "\n")

        self.reason_lbl = ctk.CTkLabel(
            self, text="Reason: --", font=("Consolas", 13),
            text_color="#9AA3AE", anchor="w", justify="left",
        )
        self.reason_lbl.grid(row=9, column=0, columnspan=2,
                             padx=14, pady=(0, 8), sticky="ew")

        self.grid_columnconfigure(0, weight=0, minsize=config.DISPLAY_WIDTH + 24)
        self.grid_columnconfigure(1, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, lambda: (self.lift(), self.focus_force()))
        self.after(30, self.loop)

    # ------------------------------------------------------------- controls --
    def toggle_mute(self):
        self.muted = not self.muted
        self.mute_btn.configure(text=f"Mute: {'ON' if self.muted else 'OFF'}")

    def recalibrate(self):
        """Restart eye calibration (press C)."""
        self.calibrator.reset()
        self.perclos_meter.reset()
        self.ear_smoother.reset()
        self.stabilizer.reset()
        self.debouncer.reset()
        self.add_log("Eye calibration started - look at the camera, eyes open.")

    def _detect(self, frame, min_face_width=None):
        """Detect faces, retrying once with a centre zoom when needed.

        A second pass is triggered when nobody is found **or** when the driver's
        box is narrower than ``min_face_width`` (the driver is far away and the
        landmarks are too coarse to trust). The pass that sees the driver
        bigger wins.

        Returns ``(faces, zoom)`` where ``zoom`` is 1.0 for a normal pass or
        ``config.ZOOM_FACTOR`` when the crop retry was used.
        """
        height, width = frame.shape[:2]
        faces = self.detector.process(frame)

        def driver_width(candidates):
            if not candidates:
                return 0
            index = select_driver(candidates, width, height)
            box = features.face_bbox(candidates[index])
            return 0 if box is None else (box[2] - box[0])

        need_zoom = not faces
        if faces and min_face_width:
            need_zoom = driver_width(faces) < int(min_face_width)
        if not need_zoom:
            return faces, 1.0

        x, y, crop_w, crop_h = centre_crop(width, height, config.ZOOM_FACTOR)
        if crop_w >= width or crop_h >= height:
            return faces, 1.0
        crop = frame[y:y + crop_h, x:x + crop_w]
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        zoomed = self.detector.process(crop)
        if not zoomed:
            return faces, 1.0
        mapped = remap_landmarks(zoomed, x, y, crop_w, crop_h, width, height)
        if faces and driver_width(mapped) < driver_width(faces):
            return faces, 1.0
        return mapped, config.ZOOM_FACTOR

    @staticmethod
    def _draw_banner(frame, text, colour, scale=1.0):
        """Draw a centred, black-backed banner on top of the video frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.9 * scale
        thickness = max(2, int(round(2 * scale)))
        (text_w, text_h), _ = cv2.getTextSize(text, font, size, thickness)
        height, width = frame.shape[:2]
        x = max(0, (width - text_w) // 2)
        y = int(62 * scale)
        pad_x, pad_y = int(12 * scale), int(10 * scale)
        cv2.rectangle(frame,
                      (max(0, x - pad_x), max(0, y - text_h - pad_y)),
                      (min(width - 1, x + text_w + pad_x), y + pad_y),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, size, colour, thickness)

    def add_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")
        self.event_log.add(msg, timestamp=ts)

    def toggle_rec(self):
        if not self.recording:
            try:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.writer = cv2.VideoWriter("demo_session.avi", fourcc,
                                              config.RECORD_FPS,
                                              (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                self.recording = True
                self.add_log("Recording started (demo_session.avi). Press R to stop.")
            except Exception as e:
                self.add_log(f"Recording failed: {e}")
        else:
            self.recording = False
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None
            self.add_log("Recording stopped.")

    def _save_report(self):
        return self.event_log.save()

    def on_close(self):
        self.running = False
        self.cam.release()
        self.detector.close()
        try:
            self.meas_log.close()
        except Exception:
            pass
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        if self.ser is not None:
            try:
                self.ser.write(b"0")
                self.ser.close()
            except Exception:
                pass
        if not self._save_report():
            print("WARNING: could not save session_report.csv (folder read-only?).")
        self.destroy()

    # ----------------------------------------------------------------- loop --
    def loop(self):
        if not self.running:
            return

        frame = self.cam.read()
        if frame is None:
            if self.cam.reconnects != self._last_reconnects:
                self._last_reconnects = self.cam.reconnects
                self.add_log("Camera lost! Reconnecting...")
            if self.cam.dead:
                self.add_log("Camera did not come back. Rerun the app.")
                self.running = False
            self.after(100, self.loop)
            return

        if config.MIRROR:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        _gs = cv2.cvtColor(cv2.resize(frame, (80, 60)), cv2.COLOR_BGR2GRAY)
        brightness = float(_gs.mean())
        if brightness < 40.0:  # night/low-light: warn + auto-enhance
            if time.time() - self.last_dark_warn > 10:
                self.last_dark_warn = time.time()
                self.add_log(f"Low light ({brightness:.0f}) - accuracy reduced.")
            _lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            _l, _a, _b = cv2.split(_lab)
            _l = cv2.createCLAHE(2.0, (8, 8)).apply(_l)
            frame = cv2.cvtColor(cv2.merge((_l, _a, _b)), cv2.COLOR_LAB2BGR)

        try:
            faces, self.zoom = self._detect(frame, config.FACE_TOO_FAR_WIDTH_PX)
        except Exception as e:
            self.vision_errors += 1
            if self.vision_errors <= 3:
                self.add_log(f"Vision hiccup ignored ({e})")
            self.after(30, self.loop)
            return

        now = time.monotonic()
        face_present = bool(faces)
        perclos = None
        raw_ear = None
        avg_ear = None
        mar = None
        head_down = False
        yawning_now = False
        distracted = False
        recent = 0
        too_far = False
        eye_valid = False
        lm = None
        turn = None

        if face_present:
            driver_idx = select_driver(faces, w, h)
            for i, other in enumerate(faces):
                if i == driver_idx:
                    continue
                ox, oy = other[:, 0], other[:, 1]
                cv2.rectangle(frame, (int(ox.min()), int(oy.min())),
                              (int(ox.max()), int(oy.max())), config.BGR_GRAY, 1)
                cv2.putText(frame, "Passenger - ignored", (int(ox.min()), int(oy.min()) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.BGR_GRAY, 2)

            lm = faces[driver_idx]
            box = features.face_bbox(lm)
            face_w_px = 0 if box is None else (box[2] - box[0])
            # Landmarks from a zoomed pass are ~ZOOM_FACTOR larger; convert back
            # so "is the driver really far?" is measured on the real frame.
            real_w_px = face_w_px / max(1.0, self.zoom)
            if box is not None:
                box_colour = config.BGR_ORANGE if self.zoom > 1.0 else config.BGR_GREEN
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_colour, 2)
            too_far = face_w_px < config.FACE_TOO_FAR_WIDTH_PX
            # Still ask the driver to come closer even when the zoom pass made
            # the landmarks usable.
            self.too_far_hint = real_w_px < config.FACE_TOO_FAR_WIDTH_PX

            turn = features.head_turn_ratio(lm)

            raw_ear = features.average_ear(lm)
            self.eye_gate = eye_gate(face_w_px, turn, raw_ear)

            if self.eye_gate == "ok":
                avg_ear = self.ear_smoother.update(raw_ear)
            else:
                # A turned head / tiny face deforms the eye corners: feeding it
                # to the meter would fake a closed eye and trigger DROWSY.
                self.ear_smoother.update(None)
                avg_ear = None
            eye_valid = avg_ear is not None

            if not self.calibrator.done:
                if self.calibrator.feed(now, avg_ear if eye_valid else None):
                    self.add_log("Calibration done - " + self.calibrator.summary())
                    self.perclos_meter.reset()

            eye_closed = self.calibrator.is_closed(avg_ear) if eye_valid else None
            perclos = self.perclos_meter.update(now, eye_closed)

            mar = features.mar(lm)
            if mar is not None:
                if mar > config.MAR_YAWN_THRESH:
                    self.yawn_open += 1
                else:
                    if self.yawn_open >= config.YAWN_MIN_FRAMES:
                        self.yawns.append(time.time())
                        self.add_log(f"Yawn detected (MAR={mar:.2f})")
                    self.yawn_open = 0
                yawning_now = self.yawn_open >= config.YAWN_MIN_FRAMES
            recent = len([t for t in self.yawns if time.time() - t < config.YAWN_WINDOW_SEC])

            hd = features.head_down_ratio(lm)
            head_down = hd is not None and hd > config.HEAD_DOWN_THRESH

            if turn is not None:
                self.turn_hist.append(turn)
                smooth = sum(self.turn_hist) / len(self.turn_hist)
                if smooth > config.TURN_ENTER:      # enter: clearly turned
                    self.turn_frames += 1
                    self.turn_clear = 0
                elif smooth < config.TURN_EXIT:     # exit: back to center
                    self.turn_clear += 1
                    if self.turn_clear >= config.TURN_CLEAR_FRAMES:
                        self.turn_frames = 0
                        if distracted:
                            self.distract_ready_at = now + config.DISTRACT_COOLDOWN_SEC
                # dead zone between TURN_EXIT and TURN_ENTER: hold counter (no jitter)
                distracted = (self.turn_frames >= config.TURN_MIN_FRAMES
                              and now >= self.distract_ready_at)
        else:
            perclos = self.perclos_meter.update(now, None)
            self.ear_smoother.update(None)
            self.eye_gate = "no face"
            self.too_far_hint = False

        # ------------------------------------------------------- decision --
        if self.calibrator.active:
            # Do not judge the driver while we are still learning their eyes.
            decision = ds.Decision(
                ds.UNKNOWN,
                f"calibrating eyes {self.calibrator.progress * 100:.0f}% - keep them open",
            )
        else:
            raw_decision = self.engine.evaluate(
                perclos=perclos,
                head_down=head_down,
                distracted=distracted,
                recent_yawns=recent,
                yawning_now=yawning_now,
                face_present=face_present,
                too_far=too_far,
            )
            decision = self.stabilizer.apply(raw_decision, perclos, now)
        state = decision.state
        level = command_for_state(state)
        stable_level = self.debouncer.update(level)

        attention = None
        if perclos is not None and state not in (ds.UNKNOWN, ds.CAMERA_ERROR):
            attention = attention_score(perclos, head_down, distracted,
                                        recent, yawning_now, state)
        att_text, att_color = attention_level(attention)

        # --------------------------------------------------------- serial --
        if self.ser is not None and stable_level != self.last_level:
            try:
                self.ser.write(stable_level.encode())
            except Exception:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
        self.last_level = stable_level

        # ----------------------------------------------------------- beep --
        if not self.muted and stable_level in ("1", "2", "3"):
            gap = {"1": 2.0, "3": 1.2, "2": 0.8}[stable_level]
            if now - self.last_beep > gap:
                self.last_beep = now
                try:
                    import winsound
                    if stable_level == "2":
                        winsound.Beep(1500, 400)          # urgent high
                    elif stable_level == "3":
                        winsound.Beep(1000, 150)          # double-beep
                        winsound.Beep(1000, 150)
                    else:
                        winsound.Beep(800, 150)           # soft tick
                except Exception:
                    print("\a", flush=True)

        # ---------------------------------------------------------- logs --
        if state == ds.DROWSY:
            if not self.was_drowsy:
                self.drowsy_count += 1
                pct = f"{perclos:.0f}%" if perclos is not None else "--"
                self.add_log(f"DROWSINESS ALERT #{self.drowsy_count} ({pct}) - {decision.reason}")
            self.was_drowsy = True
        else:
            self.was_drowsy = False

        if att_text == "Low" and face_present:
            if not self.was_low_att:
                self.add_log(f"Attention LOW ({attention:.0f}%) - {state}")
            self.was_low_att = True
        else:
            self.was_low_att = False

        # -------------------------------------------------------------- UI --
        if self.calibrator.active:
            self.status_lbl.configure(
                text=f"CALIBRATING {self.calibrator.progress * 100:.0f}% - keep eyes open",
                text_color="orange")
        else:
            label, color = STATE_DISPLAY.get(state, (state, "gray"))
            self.status_lbl.configure(text=label, text_color=color)

        if perclos is None:
            self.perc_lbl.configure(text="Eye closed: --")
            self.bar.set(0)
        else:
            self.perc_lbl.configure(text=f"Eye closed: {perclos:.0f}%")
            self.bar.set(perclos / 100.0)

        ear_txt = "EAR: --" if avg_ear is None else f"EAR: {avg_ear:.2f}"
        mar_txt = "MAR: --" if mar is None else f"MAR: {mar:.2f}"
        self.ear_lbl.configure(text=f"{ear_txt}  {mar_txt}")
        self.count_lbl.configure(
            text=f"Drowsy: {self.drowsy_count} | "
                 f"Yawns(90s): {len([t for t in self.yawns if time.time() - t < 90])}")

        if attention is None:
            self.att_lbl.configure(text="Attention: --", text_color="gray")
            self.att_bar.set(0)
        else:
            self.att_lbl.configure(text=f"Attention: {attention:.0f}% ({att_text})",
                                   text_color=att_color)
            self.att_bar.set(attention / 100.0)
            try:
                self.att_bar.configure(progress_color=att_color)
            except Exception:
                pass

        # ------------------------------------------------------- reason line --
        calib_note = ""
        if self.calibrator.done and self.calibrator.baseline is not None:
            calib_note = " (calibrated)"
        self.reason_lbl.configure(
            text=(f"Reason: {decision.reason}"
                  f"   |   eye gate: {self.eye_gate}"
                  f"   |   EAR closed < {self.calibrator.threshold:.2f}{calib_note}"
                  f"   |   zoom {self.zoom:.1f}x"
                  f"   |   press C to recalibrate")
        )

        # --------------------------------------------------- frame drawing --
        s = max(0.5, w / 640.0)          # every size below was tuned at 640 px
        font = cv2.FONT_HERSHEY_SIMPLEX
        thin = max(1, int(round(1 * s)))
        thick = max(1, int(round(2 * s)))

        if lm is not None and face_present:
            for i in config.LEFT_EYE + config.RIGHT_EYE:
                cv2.circle(frame, (int(lm[i][0]), int(lm[i][1])),
                           max(1, int(2 * s)), (255, 0, 0), -1)

        extra = []
        if head_down and face_present:
            extra.append("HEAD DOWN")
        if yawning_now:
            extra.append("YAWNING...")
        if extra:
            cv2.putText(frame, " ".join(extra), (int(20 * s), int(36 * s)),
                        font, 0.8 * s, config.BGR_RED, thick)

        if self.calibrator.active:
            hint = (f"CALIBRATING {self.calibrator.progress * 100:.0f}%"
                    " - KEEP YOUR EYES OPEN")
        elif not face_present:
            hint = "NO FACE - LOOK AT THE CAMERA"
        elif too_far or self.too_far_hint:
            hint = "MOVE CLOSER TO THE CAMERA"
        elif self.eye_gate.startswith("head turned"):
            hint = "HEAD TURNED - FACE THE ROAD"
        else:
            hint = ""
        if hint:
            self._draw_banner(frame, hint, config.BGR_RED, s)

        if face_present:
            if perclos is None:
                eyes_state = "--"
            elif perclos >= config.PERCLOS_HIGH:
                eyes_state = "Closed"
            elif perclos > config.PERCLOS_LOW:
                eyes_state = "Closing"
            else:
                eyes_state = "Open"
            if not eye_valid:
                eyes_state = f"n/a ({self.eye_gate})"
            hp_state = "Down" if head_down else "Normal"
            for _j, (_txt, _c) in enumerate([
                (f"Eyes: {eyes_state}", config.BGR_WHITE),
                (f"Head Pose: {hp_state}", config.BGR_WHITE),
                (f"Attention: {att_text}", config.ATT_COLORS.get(att_color, config.BGR_WHITE)),
            ]):
                cv2.putText(frame, _txt,
                            (max(int(10 * s), w - int(290 * s)), int(26 * s + _j * 26 * s)),
                            font, 0.55 * s, _c, thin)

        reason_txt = f"Reason: {decision.reason}"
        if len(reason_txt) > 78:
            reason_txt = reason_txt[:75] + "..."
        cv2.putText(frame, reason_txt, (int(20 * s), h - int(18 * s)),
                    font, 0.5 * s, config.BGR_WHITE, thin)

        if self.recording:
            cv2.circle(frame, (w - int(40 * s), int(35 * s)), int(12 * s),
                       config.BGR_RED, -1)
            cv2.putText(frame, "REC", (w - int(115 * s), int(45 * s)),
                        font, 0.8 * s, config.BGR_RED, thick)
            try:
                self.writer.write(cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT)))
            except Exception:
                pass

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
            (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT), Image.LANCZOS)
        self.video_lbl.configure(image=ctk.CTkImage(light_image=img,
                                                    size=(config.DISPLAY_WIDTH,
                                                          config.DISPLAY_HEIGHT)),
                                 text="")

        self.graph_hist.append(float("nan") if perclos is None else perclos)
        self.line.set_data(range(len(self.graph_hist)), list(self.graph_hist))
        self.ax.set_xlim(0, max(config.GRAPH_LEN, len(self.graph_hist)))
        self.canvas.draw_idle()

        # ------------------------------------------------------ measurement --
        try:
            self.meas_log.log(
                timestamp=now, state=state, level=stable_level,
                reason=decision.reason,
                perclos=perclos, ear=avg_ear, mar=mar, attention=attention,
                head_down=head_down, distracted=distracted, yawning=yawning_now,
                face_present=face_present, eye_valid=eye_valid,
                coverage=self.perclos_meter.coverage,
            )
        except Exception:
            pass

        if time.time() - self.last_autosave >= config.AUTOSAVE_SEC:  # autosave every minute
            self.last_autosave = time.time()
            if self._save_report():
                self.add_log("Session autosaved.")
        self.after(30, self.loop)


if __name__ == "__main__":
    Dashboard().mainloop()
