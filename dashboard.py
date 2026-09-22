"""
DrowsyGuard AI - Dashboard (CustomTkinter)
Run:
  C:\\Users\\EL-Mostwred\\anaconda3\\envs\\drowsiness\\python.exe dashboard.py
"""
from collections import deque
import csv
import time
from datetime import datetime

import cv2
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import mediapipe as mp
import numpy as np
from PIL import Image

try:
    import serial
except ImportError:
    serial = None

SERIAL_PORT = "COM5"   # change to your Arduino port (Device Manager)
SERIAL_BAUD = 9600

EAR_CLOSED_THRESH = 0.21
PERCLOS_WINDOW = 60
PERCLOS_HIGH = 40.0
HEAD_DOWN_THRESH = 0.55
MAR_YAWN_THRESH = 0.60
YAWN_MIN_FRAMES = 12
YAWN_ALERT_COUNT = 2
TURN_THRESH = 0.18
TURN_MIN_FRAMES = 30
GRAPH_LEN = 120

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

mp_face = mp.solutions.face_mesh
ctk.set_appearance_mode("dark")


def ear(pts):
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


class Dashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DrowsyGuard AI - DEPI Dashboard")
        self.geometry("1180x720")

        self.ear_hist = deque(maxlen=PERCLOS_WINDOW)
        self.graph_hist = deque(maxlen=GRAPH_LEN)
        self.yawns = deque(maxlen=16)
        self.yawn_open = 0
        self.turn_frames = 0
        self.turn_hist = deque(maxlen=10)
        self.turn_clear = 0
        self.distract_ready_at = 0.0
        self.drowsy_count = 0
        self.was_drowsy = False
        self.muted = False
        self.running = True
        self.alerts = []
        self.last_level = ""
        self.last_beep = 0.0
        self.ser = None
        if serial is not None:
            try:
                import time as _t
                self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
                _t.sleep(2)
                self._pending_log = f"Arduino linked on {SERIAL_PORT}."
            except Exception as e:
                self.ser = None
                self._pending_log = f"Arduino not found ({e}) - vision only."

        self.mesh = mp_face.FaceMesh(max_num_faces=2, refine_landmarks=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise SystemExit("Camera not found.")

        # --- layout: video left, stats right ---
        self.video_lbl = ctk.CTkLabel(self, text="")
        self.video_lbl.grid(row=0, column=0, rowspan=6, padx=12, pady=12, sticky="nsew")

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

        self.mute_btn = ctk.CTkButton(self, text="Mute: OFF", command=self.toggle_mute)
        self.mute_btn.grid(row=5, column=1, padx=12, pady=4, sticky="ew")

        # --- graph bottom-left ---
        self.fig = Figure(figsize=(5.4, 2.4), dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("Eye closure % (live)")
        self.line, = self.ax.plot([], [])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=6, column=0, padx=12, pady=8, sticky="nsew")

        # --- alert log bottom-right ---
        self.log = ctk.CTkTextbox(self, height=180)
        self.log.grid(row=6, column=1, padx=12, pady=8, sticky="nsew")
        self.log.insert("end", "Alert log ready.\n")
        if getattr(self, "_pending_log", ""):
            self.log.insert("end", self._pending_log + "\n")

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, lambda: (self.lift(), self.focus_force()))
        self.after(30, self.loop)

    def toggle_mute(self):
        self.muted = not self.muted
        self.mute_btn.configure(text=f"Mute: {'ON' if self.muted else 'OFF'}")

    def add_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")
        self.alerts.append((ts, msg))

    def on_close(self):
        self.running = False
        self.cap.release()
        if self.ser is not None:
            try:
                self.ser.write(b"0")
                self.ser.close()
            except Exception:
                pass
        try:
            with open("session_report.csv", "w", newline="") as f:
                csv.writer(f).writerows([("time", "event")] + self.alerts)
        except Exception:
            pass
        self.destroy()

    def loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            h, w = frame.shape[:2]
            res = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            perclos, avg_ear, mar = 0.0, 0.0, 0.0
            status, color = "No Face", "gray"
            head_down, yawning_now, distracted = False, False, False

            if res.multi_face_landmarks:
                fl = max(res.multi_face_landmarks, key=lambda f: (
                    (max(p.x for p in f.landmark) - min(p.x for p in f.landmark)) *
                    (max(p.y for p in f.landmark) - min(p.y for p in f.landmark))))
                lm = np.array([(p.x * w, p.y * h) for p in fl.landmark])
                cv2.rectangle(frame, (int(lm[:, 0].min()), int(lm[:, 1].min())),
                              (int(lm[:, 0].max()), int(lm[:, 1].max())), (0, 255, 0), 2)
                avg_ear = (ear(lm[LEFT_EYE]) + ear(lm[RIGHT_EYE])) / 2.0
                self.ear_hist.append(1 if avg_ear < EAR_CLOSED_THRESH else 0)
                perclos = 100.0 * sum(self.ear_hist) / max(1, len(self.ear_hist))

                mh = np.linalg.norm(lm[13] - lm[14])
                mw = np.linalg.norm(lm[78] - lm[308])
                mar = mh / (mw + 1e-6)
                if mar > MAR_YAWN_THRESH:
                    self.yawn_open += 1
                else:
                    if self.yawn_open >= YAWN_MIN_FRAMES:
                        self.yawns.append(time.time())
                        self.add_log(f"Yawn detected (MAR={mar:.2f})")
                    self.yawn_open = 0
                yawning_now = self.yawn_open >= YAWN_MIN_FRAMES
                recent = len([t for t in self.yawns if time.time() - t < 90])

                eye_y = (lm[33][1] + lm[263][1]) / 2.0
                head_down = (lm[1][1] - eye_y) / max(1.0, (lm[152][1] - eye_y)) > HEAD_DOWN_THRESH

                face_w = max(1.0, lm[:, 0].max() - lm[:, 0].min())
                face_cx = (lm[:, 0].max() + lm[:, 0].min()) / 2.0
                turn = abs(lm[1][0] - face_cx) / face_w
                self.turn_hist.append(turn)
                smooth = sum(self.turn_hist) / len(self.turn_hist)
                if smooth > 0.22:            # enter: clearly turned
                    self.turn_frames += 1
                    self.turn_clear = 0
                elif smooth < 0.12:          # exit: back to center
                    self.turn_clear += 1
                    if self.turn_clear >= 15:
                        self.turn_frames = 0
                        if distracted:
                            self.distract_ready_at = time.time() + 3.0  # cooldown
                # dead zone 0.12-0.22: hold counter (no jitter)
                distracted = self.turn_frames >= 45 and time.time() >= self.distract_ready_at

                if perclos >= PERCLOS_HIGH or (head_down and perclos > 20) or recent >= YAWN_ALERT_COUNT:
                    status, color = "DROWSY - Stay Alert!", "red"
                elif distracted:
                    status, color = "DISTRACTED - Look Ahead!", "orange"
                elif perclos > 15 or yawning_now:
                    status, color = "LOW RISK", "yellow"
                else:
                    status, color = "ALERT - Safe", "green"

                for i in LEFT_EYE + RIGHT_EYE:
                    cv2.circle(frame, (int(lm[i][0]), int(lm[i][1])), 2, (255, 0, 0), -1)
            else:
                self.ear_hist.append(0)
                recent = 0

            self.graph_hist.append(perclos)
            # graduated alarm: level char -> Arduino, distinct PC beep per level
            if status.startswith("DROWSY"):
                level = "2"
            elif status.startswith("DISTRACTED"):
                level = "3"
            elif status.startswith("LOW"):
                level = "1"
            elif status == "No Face":
                level = "N"
            else:
                level = "0"

            if self.ser is not None and level != self.last_level:
                try:
                    self.ser.write(level.encode())
                except Exception:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    self.ser = None
            self.last_level = level

            if not self.muted and level in ("1", "2", "3"):
                now_t = time.time()
                gap = {"1": 2.0, "3": 1.2, "2": 0.8}[level]
                if now_t - self.last_beep > gap:
                    self.last_beep = now_t
                    try:
                        import winsound
                        if level == "2":
                            winsound.Beep(1500, 400)          # urgent high
                        elif level == "3":
                            winsound.Beep(1000, 150)          # double-beep
                            winsound.Beep(1000, 150)
                        else:
                            winsound.Beep(800, 150)           # soft tick
                    except Exception:
                        print("\a", flush=True)

            if status.startswith("DROWSY"):
                if not self.was_drowsy:
                    self.drowsy_count += 1
                    self.add_log(f"DROWSINESS ALERT #{self.drowsy_count} ({perclos:.0f}%)")
                self.was_drowsy = True
            else:
                self.was_drowsy = False

            self.status_lbl.configure(text=status, text_color=color)
            self.perc_lbl.configure(text=f"Eye closed: {perclos:.0f}%")
            self.bar.set(perclos / 100.0)
            self.ear_lbl.configure(text=f"EAR: {avg_ear:.2f}  MAR: {mar:.2f}")
            self.count_lbl.configure(
                text=f"Drowsy: {self.drowsy_count} | Yawns(90s): {len([t for t in self.yawns if time.time()-t < 90])}")
            extra = "HEAD DOWN" if head_down and res.multi_face_landmarks else ""
            if yawning_now:
                extra += " YAWNING..."
            if extra:
                cv2.putText(frame, extra.strip(), (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 480))
            self.video_lbl.configure(image=ctk.CTkImage(light_image=img, size=(640, 480)), text="")
            self.line.set_data(range(len(self.graph_hist)), list(self.graph_hist))
            self.ax.set_xlim(0, max(GRAPH_LEN, len(self.graph_hist)))
            self.canvas.draw_idle()

        self.after(30, self.loop)


if __name__ == "__main__":
    Dashboard().mainloop()
