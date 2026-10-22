"""
Video builder for the DrowsyGuard AI promo (9:16, 1080x1920).

Everything is drawn programmatically (no external editor): red top banner,
white caption cards, animated code lines and report figures.

Arabic text is shaped with arabic_reshaper + python-bidi because Pillow on this
machine is built without libraqm.

Usage:
    python tools/make_video.py --sample          # 9-second style sample
"""
from __future__ import annotations

import argparse
import math
import sys
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
FONT_REGULAR = "C:/Windows/Fonts/tahoma.ttf"
FONT_BOLD = "C:/Windows/Fonts/tahomabd.ttf"
FONT_LATIN = "C:/Windows/Fonts/arialbd.ttf"

W, H, FPS = 1080, 1920, 30

RED = (211, 47, 47)
RED_DARK = (160, 28, 28)
WHITE = (255, 255, 255)
INK = (18, 22, 28)
MUTED = (138, 150, 162)
GOLD = (240, 190, 80)


# ------------------------------------------------------------------ helpers ---
def ar(text: str) -> str:
    """Shape Arabic + apply bidi so PIL draws it correctly."""
    return get_display(arabic_reshaper.reshape(text))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def ease_out(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def fit(path: str, text: str, draw, max_width: int, start: int, floor: int = 28):
    """Largest font size that keeps ``text`` inside ``max_width``."""
    size = start
    while size > floor:
        candidate = font(path, size)
        if draw.textlength(text, font=candidate) <= max_width:
            return candidate
        size -= 2
    return font(path, floor)


def text_center(draw, text, fnt, y, fill, shaped=True):
    value = ar(text) if shaped else text
    width = draw.textlength(value, font=fnt)
    draw.text(((W - width) / 2, y), value, font=fnt, fill=fill)
    return width


def background() -> np.ndarray:
    """Vertical gradient + soft vignette, as a BGR frame."""
    top = np.array([10, 14, 20], dtype=float)      # RGB
    bottom = np.array([24, 32, 44], dtype=float)
    ramp = np.linspace(0, 1, H)[:, None, None]
    rgb = top * (1 - ramp) + bottom * ramp
    rgb = np.repeat(rgb, W, axis=1).astype(np.uint8)

    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H * 0.45
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.hypot(cx, cy)
    vignette = np.clip(1.15 - 0.55 * dist, 0.35, 1.0)[..., None]
    rgb = (rgb * vignette).clip(0, 255).astype(np.uint8)

    # faint grid
    rgb[::90, :, :] = (rgb[::90, :, :] * 0.86).astype(np.uint8)
    rgb[:, ::90, :] = (rgb[:, ::90, :] * 0.86).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def rounded_card(draw, box, radius, fill, shadow=10):
    x1, y1, x2, y2 = box
    if shadow:
        draw.rounded_rectangle((x1 + 6, y1 + shadow, x2 + 6, y2 + shadow),
                               radius=radius, fill=(0, 0, 0))
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def banner(draw, progress: float):
    """Red top banner that slides down."""
    height = int(168 * ease_out(progress))
    if height <= 0:
        return
    draw.rectangle((0, 0, W, height), fill=RED)
    draw.rectangle((0, height - 8, W, height), fill=RED_DARK)
    if height > 120:
        fnt = font(FONT_BOLD, 58)
        text_center(draw, "مشروع تخرج AI", fnt, 48, WHITE)


def footer(draw, alpha=1.0):
    fnt = font(FONT_LATIN, 30)
    text_center(draw, "ENG. MAHMOUD ABDO  ·  DEPI", fnt, H - 92,
                tuple(int(c * alpha) for c in MUTED), shaped=False)


def draw_card(img, t, start, end, kicker, title, subtitle, accent=RED):
    """Slide-up white card with kicker / title / subtitle (Arabic)."""
    if t < start or t > end:
        return img
    local = t - start
    p = ease_out(local / 0.45)
    alpha = clamp01(local / 0.3) * clamp01((end - t) / 0.3)
    if alpha <= 0.01:
        return img

    card_w, card_h = 900, 430
    x1 = (W - card_w) // 2
    y1 = int(760 + (1 - p) * 80)
    x2, y2 = x1 + card_w, y1 + card_h

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    rounded_card(draw, (x1, y1, x2, y2), 44, WHITE + (255,))
    draw.rounded_rectangle((x1 + 56, y1 + 62, x1 + 56 + 8, y1 + 118), radius=4,
                           fill=accent)

    kicker_font = font(FONT_BOLD, 38)
    draw.text((x1 + 92, y1 + 58), ar(kicker), font=kicker_font, fill=accent)

    title_font = fit(FONT_BOLD, ar(title), draw, card_w - 150, start=76, floor=44)
    draw.text((x1 + 90, y1 + 140), ar(title), font=title_font, fill=INK)

    sub_font = fit(FONT_REGULAR, ar(subtitle), draw, card_w - 150, start=42, floor=28)
    draw.text((x1 + 90, y1 + 280), ar(subtitle), font=sub_font, fill=MUTED)

    if alpha < 1:
        arr = np.array(layer)
        arr[..., 3] = (arr[..., 3].astype(float) * alpha).astype(np.uint8)
        layer = Image.fromarray(arr)
    img.alpha_composite(layer)
    return img


def sample(out_path: Path, frames_dir: Path) -> None:
    seconds = 9.0
    bg = background()
    frames_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    preview_at = {1.6: "frame_01_banner.png",
                  3.9: "frame_02_problem.png",
                  6.4: "frame_03_result.png",
                  8.6: "frame_04_outro.png"}

    total = int(seconds * FPS)
    for i in range(total):
        t = i / FPS
        img = Image.fromarray(cv2.cvtColor(bg.copy(), cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(img)

        banner(draw, clamp01(t / 0.6))
        footer(draw, 1.0 if t > 0.8 else 0.0)

        img = draw_card(img, t, 2.2, 4.9, "المشكلة",
                        "السواق بيسقط من النعاس",
                        "أقل من ثانية غفلة = حادثة على الطريق")
        img = draw_card(img, t, 4.9, 7.2, "النتيجة",
                        "99.4% دقة كشف المطبات",
                        "موديل YOLOv8 اتدرّب على 1,850 صورة")
        img = draw_card(img, t, 7.2, 9.0, "الحل",
                        "DrowsyGuard AI",
                        "حماية تبدأ من نظرة… وتنبّه قبل ما تنام")

        frame = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        writer.write(frame)

        for stamp, name in preview_at.items():
            if abs(t - stamp) < 0.5 / FPS:
                cv2.imwrite(str(frames_dir / name), frame)

    writer.release()
    print(f"sample video -> {out_path}")
    print(f"preview frames -> {frames_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="render the 9s sample")
    parser.add_argument("--out", default=str(ROOT / "demo_video" / "sample_intro.mp4"))
    args = parser.parse_args()

    if args.sample:
        sample(Path(args.out), ROOT / "demo_video" / "sample_frames")
    else:
        parser.print_help()


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 10):
        print(f"WARNING: Python {sys.version_info[0]}.{sys.version_info[1]}")
    main()
