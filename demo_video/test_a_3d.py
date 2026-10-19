"""
Test A - 3D explainer video rendered in Python (matplotlib 3D + OpenCV).

This is a SMALL SAMPLE (5 seconds) to judge the look & feel of approach A:
a real .mp4 file, works offline, no browser and no internet needed.

Run:
  python test_a_3d.py

Output:
  test_A_3d.mp4  (next to this file)
"""
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_A_3d.mp4")

W, H = 960, 540
FPS = 24
SECONDS = 5
BG = "#0a0f1e"
ROAD = "#2b3245"
LANE = "#e8eefc"
CAR = "#38bdf8"
CAR2 = "#0ea5e9"


def draw_scene(ax, phase, title, subtitle):
    ax.clear()
    ax.set_facecolor(BG)

    # ground / road
    ax.bar3d(-7, -4, -0.6, 14, 70, 0.6, color=ROAD, shade=True)
    # shoulders
    ax.bar3d(-9, -4, -0.6, 2, 70, 0.6, color="#1b2233", shade=True)
    ax.bar3d(7, -4, -0.6, 2, 70, 0.6, color="#1b2233", shade=True)

    # lane dashes scrolling towards the camera
    period = 8.0
    for i in range(14):
        y = (i * period - phase) % (14 * period) - 4
        ax.bar3d(-0.35, y, -0.05, 0.7, 3.2, 0.05, color=LANE, shade=False)

    # side poles to give a sense of speed
    for i in range(9):
        y = (i * 12 - (phase * 1.0)) % 108 - 4
        ax.bar3d(-8.6, y, -0.05, 0.35, 0.35, 5.0, color="#64748b", shade=True)
        ax.bar3d(8.25, y, -0.05, 0.35, 0.35, 5.0, color="#64748b", shade=True)

    # the car (body + cabin + 4 wheels)
    body_y, body_x = 6.0, 0.0
    ax.bar3d(body_x - 2.0, body_y - 4.0, 0.35, 4.0, 8.0, 1.5,
             color=CAR, shade=True)
    ax.bar3d(body_x - 1.4, body_y - 1.6, 1.85, 2.8, 4.0, 1.25,
             color=CAR2, shade=True)
    for wx in (body_x - 2.3, body_x + 1.7):
        for wy in (body_y - 3.2, body_y + 2.0):
            ax.bar3d(wx, wy, 0.0, 0.6, 1.2, 0.7, color="#0f172a", shade=True)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-4, 46)
    ax.set_zlim(0, 9)
    ax.set_box_aspect((20, 50, 9))
    ax.view_init(elev=16, azim=-90)
    ax.axis("off")
    ax.set_position([0.15, -0.05, 0.7, 1.05])

    # HUD text drawn in 2D on top of the 3D scene
    fig = ax.figure
    for t in list(fig.texts):
        t.remove()
    fig.text(0.5, 0.92, title, color="white", fontsize=30,
             ha="center", va="center", fontweight="bold")
    fig.text(0.5, 0.85, subtitle, color="#7dd3fc", fontsize=15, ha="center")
    fig.text(0.04, 0.05, "Sample render (approach A)", color="#94a3b8",
             fontsize=11, ha="left")


def main():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))
    total = FPS * SECONDS

    for f in range(total):
        t = f / FPS
        phase = t * 22.0
        title = "DrowsyGuard AI"
        subtitle = "Sample: 3D explainer (approach A)"
        draw_scene(ax, phase, title, subtitle)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        vw.write(frame)
        if f % FPS == 0:
            print(f"  frame {f}/{total}")

    vw.release()
    plt.close(fig)
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
