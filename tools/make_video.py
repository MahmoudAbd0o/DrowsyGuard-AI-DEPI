"""
Video builder for the DrowsyGuard AI promo (9:16, 1080x1920).

Style: cinematic stock footage + red top banner + white caption cards + HUD
overlay + generated music (no external editor required).

Arabic text is shaped with arabic_reshaper + python-bidi because Pillow on this
machine is built without libraqm.

Usage:
    python tools/make_video.py --sample          # 10-second style sample
    python tools/make_video.py --sample --no-audio
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:  # pragma: no cover - env dependent
    raise SystemExit("Missing Arabic text support. Run: pip install arabic-reshaper python-bidi")

ROOT = Path(__file__).resolve().parents[1]
STOCK = ROOT / "demo_video" / "stock"
OUT_DIR = ROOT / "demo_video"

FONT_AR = "C:/Windows/Fonts/tahomabd.ttf"
FONT_AR_REG = "C:/Windows/Fonts/tahoma.ttf"
FONT_LATIN = "C:/Windows/Fonts/arialbd.ttf"

W, H, FPS = 1080, 1920, 30
RED = (211, 47, 47)
WHITE = (255, 255, 255)
MUTED = (185, 195, 205)
GREEN = (60, 220, 140)
ALERT = (255, 80, 80)


# ------------------------------------------------------------------ helpers ---
def ar(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def ease_out(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def fit_text(draw, text, path, max_width, start, floor=26):
    size = start
    while size > floor:
        candidate = font(path, size)
        if draw.textlength(text, font=candidate) <= max_width:
            return candidate
        size -= 2
    return font(path, floor)


def paste_alpha(img, layer, alpha=1.0):
    if alpha < 1:
        arr = np.array(layer)
        arr[..., 3] = (arr[..., 3].astype(float) * clamp01(alpha)).astype(np.uint8)
        layer = Image.fromarray(arr)
    img.alpha_composite(layer)


# ------------------------------------------------------------------ footage ---
@dataclass
class Clip:
    name: str
    start: float
    end: float
    offset: float = 0.0
    hud: str = "driver"
    box: tuple[int, int, int, int] | None = None
    speed: float = 1.0
    track: str | None = None


def load_track(name: str | None):
    """Normalised face boxes recorded once with MediaPipe, lightly smoothed."""
    if not name:
        return None
    path = OUT_DIR / name
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    boxes, last = [], None
    for row in rows:
        if row.get("box"):
            last = row["box"]
        boxes.append(last)

    smoothed = []
    for i in range(len(boxes)):
        window = [b for b in boxes[max(0, i - 4):i + 5] if b]
        if not window:
            smoothed.append(None)
            continue
        smoothed.append(tuple(sum(b[k] for b in window) / len(window) for k in range(4)))
    return {"fps": data["fps"], "boxes": smoothed}


TRACKS: dict = {}


def get_track(name: str | None):
    if not name:
        return None
    if name not in TRACKS:
        TRACKS[name] = load_track(name)
    return TRACKS[name]


def box_at(clip: "Clip", t: float):
    """HUD box for this frame: tracked face box when available."""
    data = get_track(clip.track)
    if not data or not data["boxes"]:
        return clip.box
    src = clip.offset + (t - clip.start) * clip.speed
    idx = max(0, min(len(data["boxes"]) - 1, int(src * data["fps"])))
    b = data["boxes"][idx]
    if not b:
        return clip.box
    pad = 0.05
    return (max(8.0, (b[0] - pad) * W),
            max(170.0, (b[1] - pad) * H),
            min(W - 8.0, (b[2] + pad) * W),
            min(H - 8.0, (b[3] + pad) * H))


def cover_frame(frame, width=W, height=H):
    """Crop the 16:9 source to 9:16 *before* resizing (keeps it cheap)."""
    h, w = frame.shape[:2]
    target = width / height
    if w / h > target:
        crop_w = int(h * target)
        x0 = (w - crop_w) // 2
        frame = frame[:, x0:x0 + crop_w]
    else:
        crop_h = int(w / target)
        y0 = (h - crop_h) // 2
        frame = frame[y0:y0 + crop_h, :]
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
    return cv2.addWeighted(frame, 1.3, blur, -0.3, 0)


def grade(frame):
    """Cool cinematic grade: teal shadows, warm highlights, contrast."""
    img = frame.astype(np.float32)
    img *= np.array([1.06, 0.99, 0.93], dtype=np.float32)      # B,G,R
    img = (img - 127.5) * 1.14 + 127.5
    img[..., 0] += 6.0
    img[..., 2] -= 3.0
    return np.clip(img, 0, 255).astype(np.uint8)


def vignette_mask():
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    mask = np.clip(1.18 - 0.42 * dist, 0.30, 1.0)
    return mask[..., None].astype(np.float32)


VIGNETTE = vignette_mask()


def scene_fade(t: float, start: float, end: float, fade: float = 0.30) -> float:
    return min(clamp01((t - start) / fade), clamp01((end - t) / fade))


# --------------------------------------------------------------------- text ---
def draw_banner(draw, progress: float = 1.0):
    height = int(150 * ease_out(progress))
    if height <= 0:
        return
    draw.rectangle((0, 0, W, height), fill=RED)
    draw.rectangle((0, height - 6, W, height), fill=(150, 24, 24))
    if height > 110:
        fnt = font(FONT_AR, 56)
        text = ar("مشروع تخرج AI")
        width = draw.textlength(text, font=fnt)
        draw.text(((W - width) / 2, 44), text, font=fnt, fill=WHITE)


def draw_caption(img, text: str, alpha: float, rise: float = 0.0):
    if alpha <= 0.01:
        return
    card_w, card_h = 900, 190
    x1 = (W - card_w) // 2
    y1 = int(1540 - rise)

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x1, y1, x1 + card_w, y1 + card_h), radius=34,
                        fill=(255, 255, 255, 240))
    # accent bar on the right (RTL reading side)
    d.rounded_rectangle((x1 + card_w - 44, y1 + 44, x1 + card_w - 36, y1 + 146),
                        radius=4, fill=RED)

    value = ar(text)
    fnt = fit_text(d, value, FONT_AR, card_w - 190, start=52, floor=30)
    width = d.textlength(value, font=fnt)
    d.text(((W - width) / 2, y1 + (card_h - fnt.size) / 2 - 6), value,
           font=fnt, fill=(20, 24, 30))
    paste_alpha(img, layer, alpha)


def draw_hud(draw, t: float, kind: str, phase: float,
             box: tuple[int, int, int, int] | None = None):
    if kind == "none":
        return
    x1, y1, x2, y2 = box or (130, 430, 950, 1180)
    grow = 26 * (1 - ease_out(clamp01(phase / 0.45)))
    x1, y1, x2, y2 = int(x1 - grow), int(y1 - grow), int(x2 + grow), int(y2 + grow)
    arm = 64
    color = GREEN if kind != "alert" else ALERT

    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        draw.line([(cx, cy), (cx + dx * arm, cy)], fill=color, width=4)
        draw.line([(cx, cy), (cx, cy + dy * arm)], fill=color, width=4)

    label = "DRIVER 0.98" if kind != "alert" else "DROWSY 0.97"
    fnt = font(FONT_LATIN, 30)
    tw = draw.textlength(label, font=fnt)
    draw.rectangle((x1, y1 - 48, x1 + tw + 30, y1 - 2), fill=color)
    if kind == "alert":
        draw.text((x1 + 16, y1 - 44), label, font=font(FONT_LATIN, 28),
                  fill=(25, 10, 10))
    else:
        draw.text((x1 + 16, y1 - 44), label, font=font(FONT_LATIN, 28),
                  fill=(8, 22, 14))

    scan_y = y1 + int((y2 - y1) * ((t * 0.5) % 1.0))
    dim = tuple(int(c * 0.6) for c in color)
    draw.line([(x1 + 8, scan_y), (x2 - 8, scan_y)], fill=dim, width=2)

    draw.text((x1, y2 + 20), "AI VISION  ·  LIVE", font=font(FONT_LATIN, 26),
              fill=MUTED)
    tc = f"00:{int(t):02d}:{int((t % 1) * 30):02d}"
    tc_width = draw.textlength(tc, font=font(FONT_LATIN, 26))
    draw.text((W - 60 - tc_width, 186), tc, font=font(FONT_LATIN, 26), fill=MUTED)


def draw_end_card(img, t: float):
    alpha = clamp01((t - END_START) / 0.55)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    fnt = font(FONT_LATIN, 96)
    text = "DrowsyGuard AI"
    d.text(((W - d.textlength(text, font=fnt)) / 2, 790), text, font=fnt, fill=WHITE)
    d.rectangle((W / 2 - 90, 928, W / 2 + 90, 938), fill=RED)

    fnt2 = font(FONT_AR, 52)
    value = ar("حماية تبدأ من نظرة")
    d.text(((W - d.textlength(value, font=fnt2)) / 2, 1000), value, font=fnt2, fill=MUTED)

    fnt3 = font(FONT_LATIN, 28)
    sub = "ENG. MAHMOUD ABDO  ·  DEPI"
    d.text(((W - d.textlength(sub, font=fnt3)) / 2, 1700), sub,
           font=fnt3, fill=(140, 150, 160))
    paste_alpha(img, layer, alpha)


def backdrop():
    top = np.array([8, 11, 16], dtype=float)
    bottom = np.array([22, 28, 38], dtype=float)
    ramp = np.linspace(0, 1, H)[:, None, None]
    rgb = (top * (1 - ramp) + bottom * ramp).astype(np.uint8)
    return cv2.cvtColor(np.repeat(rgb, W, axis=1), cv2.COLOR_RGB2BGR)


BACKDROP = backdrop()


# -------------------------------------------------------------------- music ---
def make_music(path: Path, seconds: float, bpm: int = 96) -> None:
    """Minimal cinematic pulse: drone + pad + kick + hats (royalty-free by design)."""
    sr = 44100
    n = int(sr * seconds)
    t = np.arange(n) / sr

    drone = (0.16 * np.sin(2 * np.pi * 55 * t) +
             0.10 * np.sin(2 * np.pi * 82.5 * t) +
             0.05 * np.sin(2 * np.pi * 110 * t))
    drone *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.08 * t)

    pad = np.zeros(n, dtype=np.float32)
    for freq in (220.0, 261.63, 329.63):
        pad += 0.035 * np.sin(2 * np.pi * freq * t)
    pad *= np.clip(t / 2.5, 0, 1) * 0.9

    kick = np.zeros(n, dtype=np.float32)
    hats = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)
    beat = 60.0 / bpm
    step = 0
    while step * beat < seconds - 0.15:
        at = int(step * beat * sr)
        length = int(0.28 * sr)
        if at + length < n:
            tt = np.arange(length) / sr
            env = np.exp(-tt * 16)
            kick[at:at + length] += (0.55 * env *
                                     np.sin(2 * np.pi * (110 * np.exp(-tt * 22) + 42) * tt))
        hat_at = int((step * beat + beat / 2) * sr)
        hlen = int(0.05 * sr)
        if hat_at + hlen < n:
            noise = rng.normal(0, 1, hlen)
            hats[hat_at:hat_at + hlen] += 0.05 * noise * np.exp(-np.arange(hlen) / (0.01 * sr))
        step += 1

    mono = drone + pad + kick + hats
    mono *= np.clip((seconds - t) / 0.8, 0, 1)
    mono = np.tanh(mono * 1.4) * 0.85
    pcm = (np.clip(np.stack([mono, mono], axis=1), -1, 1) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())


# ------------------------------------------------------------------- render ---
TIMELINE = [
    Clip("aerial_road.mp4", 0.0, 2.8, offset=2.0, hud="none"),
    Clip("drowsy_driver.mp4", 2.8, 4.15, offset=2.96, hud="driver",
         track="tracks/track_drowsy_driver.json"),
    Clip("drowsy_driver.mp4", 4.15, 8.2, offset=4.10, hud="alert", speed=0.25,
         track="tracks/track_drowsy_driver.json"),
]
END_START = 8.2

CAPTIONS = [
    (0.55, 2.55, "نظام ذكي بيراقب السواق لحظة بلحظة"),
    (2.95, 4.05, "بيقرأ عينك… ويحسب نسبة النعاس"),
    (4.6, 7.9, "وينبّهك قبل ما تغمض"),
]


def clip_index_at(t: float) -> int:
    for idx, clip in enumerate(TIMELINE):
        if clip.start <= t < clip.end:
            return idx
    return len(TIMELINE) - 1


def compose(frame_bgr, t: float) -> Image.Image:
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img)

    draw_banner(draw, clamp01(t / 0.5))

    if t >= END_START:
        draw_end_card(img, t)
        return img

    idx = clip_index_at(t)
    clip = TIMELINE[idx]
    if clip.hud == "alert":
        pulse = 0.5 + 0.5 * math.sin((t - clip.start) * 9.0)
        img.alpha_composite(Image.new("RGBA", (W, H), (255, 20, 20, int(22 * pulse))))
    draw_hud(draw, t, clip.hud, t - clip.start, box_at(clip, t))

    for start, end, text in CAPTIONS:
        if start <= t <= end:
            local = t - start
            alpha = min(clamp01(local / 0.25), clamp01((end - t) / 0.25))
            draw_caption(img, text, alpha,
                         rise=10 * (1 - ease_out(clamp01(local / 0.35))))
    return img


def render(out_path: Path, seconds: float, preview_dir: Path) -> None:
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_at = {1.4: "s1_intro.png", 3.4: "s2_face.png",
                  6.0: "s3_alert.png", 9.2: "s4_end.png"}

    cap = None
    current = -1
    src_fps = 25.0
    last_src = -1
    last_raw = None

    for i in range(int(seconds * FPS)):
        t = i / FPS

        if t < END_START:
            idx = clip_index_at(t)
            clip = TIMELINE[idx]

            if idx != current:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(str(STOCK / clip.name))
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                # read sequentially up to the requested start (frame-accurate)
                last_src = int(round(clip.offset * src_fps))
                last_raw = None
                for _ in range(last_src):
                    if not cap.read()[0]:
                        break
                current = idx

            wanted = int(round((clip.offset + (t - clip.start) * clip.speed) * src_fps))
            while last_src < wanted:
                ok, raw = cap.read()
                last_src += 1
                if not ok or raw is None:
                    break
                last_raw = raw

            raw = last_raw if last_raw is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
            frame = grade(cover_frame(raw))
            frame = (frame.astype(np.float32) * VIGNETTE).astype(np.uint8)
            fade = scene_fade(t, clip.start, clip.end)
            if fade < 1:
                frame = (frame.astype(np.float32) * fade).astype(np.uint8)
        else:
            if cap is not None:
                cap.release()
                cap = None
                current = -1
            frame = BACKDROP.copy()

        img = compose(frame, t)
        out = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        writer.write(out)

        for stamp, name in preview_at.items():
            if abs(t - stamp) < 0.5 / FPS:
                cv2.imwrite(str(preview_dir / name), out)

    if cap is not None:
        cap.release()
    writer.release()


def mux_audio(video: Path, audio: Path, final: Path) -> bool:
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", str(final)],
                       check=True, capture_output=True)
        return True
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"ffmpeg mux failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    if not args.sample:
        parser.print_help()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    silent = OUT_DIR / "sample_footage_silent.mp4"
    print("rendering ...")
    render(silent, args.seconds, OUT_DIR / "sample_frames")

    final = OUT_DIR / "sample_footage.mp4"
    if args.no_audio:
        silent.replace(final)
    else:
        music = OUT_DIR / "sample_music.wav"
        make_music(music, args.seconds)
        if mux_audio(silent, music, final):
            silent.unlink(missing_ok=True)
        else:
            silent.replace(final)
    print(f"done -> {final}")


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 10):
        print(f"WARNING: Python {sys.version_info[0]}.{sys.version_info[1]}")
    main()

