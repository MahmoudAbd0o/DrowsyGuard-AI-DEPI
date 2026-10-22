"""
Build the DrowsyGuard AI evaluation report (charts + Markdown).

Inputs (all optional except the training metrics):
  report/bump_metrics.json     final YOLO training metrics (from Kaggle)
  outputs/simulation_log.csv   scripted session produced by tools/simulate_session.py
  outputs/session_log.csv      a real dashboard session
  --images DIR                 raw images to run the bump model on
  --labels DIR                 YOLO labels for those images (optional -> metrics)

Outputs:
  report/report.md
  report/figures/*.png
  report/bump_samples/*.jpg

Usage:
  python tools/make_report.py
  python tools/make_report.py --log outputs/session_log.csv
  python tools/make_report.py --images data/bump_test/images --labels data/bump_test/labels
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

STATE_COLORS = {
    "ALERT": "#2e9e4f",
    "LOW RISK": "#e0b100",
    "DISTRACTED": "#e07b00",
    "DROWSY": "#d62728",
    "UNKNOWN": "#8c8c8c",
    "CAMERA ERROR": "#5a5a5a",
}

DEFAULT_METRICS = "report/bump_metrics.json"
DEFAULT_SIM_LOG = "outputs/simulation_log.csv"
DEFAULT_REAL_LOG = "outputs/session_log.csv"


# --------------------------------------------------------------------- metrics --
def iou(box_a, box_b) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_label_file(path: Path, width: int, height: int,
                     class_id: int = 0) -> list[tuple[float, float, float, float]]:
    """YOLO ``cls cx cy w h`` (normalised) -> pixel ``(x1, y1, x2, y2)`` boxes."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            continue
        if cls != class_id:
            continue
        boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height))
    return boxes


def match_boxes(predictions, ground_truth, threshold: float = 0.5):
    """Greedy match by confidence -> ``(tp, fp, fn)``."""
    order = sorted(range(len(predictions)),
                   key=lambda i: float(predictions[i][4]), reverse=True)
    used = set()
    tp = 0
    for i in order:
        best_j, best_iou = None, 0.0
        for j, gt in enumerate(ground_truth):
            if j in used:
                continue
            value = iou(predictions[i][:4], gt)
            if value > best_iou:
                best_j, best_iou = j, value
        if best_j is not None and best_iou >= threshold:
            used.add(best_j)
            tp += 1
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ------------------------------------------------------------------ session ----
def load_session(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame["t"] = frame["timestamp"] - frame["timestamp"].min()
    return frame.reset_index(drop=True)


def summarize_session(frame: pd.DataFrame) -> dict:
    total = len(frame)
    counts = frame["state"].value_counts().to_dict()
    duration = (float(frame["timestamp"].max() - frame["timestamp"].min())
                if total else 0.0)
    perclos = frame["perclos"].dropna()
    attention = frame["attention"].dropna()
    return {
        "frames": total,
        "duration_sec": duration,
        "counts": counts,
        "drowsy_pct": 100.0 * counts.get("DROWSY", 0) / total if total else 0.0,
        "distracted_pct": 100.0 * counts.get("DISTRACTED", 0) / total if total else 0.0,
        "unknown_pct": 100.0 * counts.get("UNKNOWN", 0) / total if total else 0.0,
        "perclos_mean": float(perclos.mean()) if len(perclos) else None,
        "perclos_max": float(perclos.max()) if len(perclos) else None,
        "attention_mean": float(attention.mean()) if len(attention) else None,
        "face_coverage": float(frame["face_present"].mean()) if total else 0.0,
    }


# ------------------------------------------------------------------- charts ----
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_training(metrics: dict, path: Path) -> None:
    labels = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    values = [metrics["precision"], metrics["recall"],
              metrics["map50"], metrics["map50_95"]]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    bars = ax.bar(labels, values, color=["#2e7d32", "#1565c0", "#f9a825", "#6a1b9a"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                f"{value * 100:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.08)
    _style(ax, "Speed-bump detector - validation metrics", "", "score (0-1)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_timeline(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    ax2 = ax.twinx()

    # state background bands
    spans = []
    start_t, start_state = float(frame["t"].iloc[0]), str(frame["state"].iloc[0])
    for row in frame.itertuples(index=False):
        if row.state != start_state:
            spans.append((start_t, float(row.t), start_state))
            start_t, start_state = float(row.t), row.state
    spans.append((start_t, float(frame["t"].iloc[-1]), start_state))
    for start, end, state in spans:
        ax.axvspan(start, end, color=STATE_COLORS.get(state, "#cccccc"), alpha=0.15)
        if end - start > 4:
            ax.text((start + end) / 2, 103, state, ha="center", va="bottom",
                    fontsize=7, color=STATE_COLORS.get(state, "#666666"))

    ax.plot(frame["t"], frame["perclos"], color="#d62728", linewidth=1.6,
            label="PERCLOS (%)")
    ax.axhline(config.PERCLOS_HIGH, color="#d62728", linestyle="--", linewidth=1,
               alpha=0.6, label=f"Drowsy threshold ({config.PERCLOS_HIGH:.0f}%)")
    ax.axhline(config.PERCLOS_LOW, color="#e0b100", linestyle=":", linewidth=1,
               alpha=0.8, label=f"Low-risk threshold ({config.PERCLOS_LOW:.0f}%)")
    ax.set_ylim(0, 110)

    ear = frame["ear"].astype(float)
    ax2.plot(frame["t"], ear, color="#1565c0", linewidth=1.1, alpha=0.8,
             label="EAR (eye aspect ratio)")
    ax2.set_ylim(0, max(0.40, float(np.nanmax(ear)) * 1.15 if ear.notna().any() else 0.4))

    _style(ax, "Driver state over time (PERCLOS + EAR)", "time (s)", "PERCLOS (%)")
    ax2.set_ylabel("EAR")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    state_patches = [Patch(color=c, alpha=0.35, label=s)
                     for s, c in STATE_COLORS.items() if s in set(frame["state"])]
    ax.legend(handles1 + handles2 + state_patches, labels1 + labels2 + [p.get_label() for p in state_patches],
              loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_states(summary: dict, path: Path) -> None:
    counts = summary["counts"]
    names = list(counts.keys())
    values = [counts[name] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    colors = [STATE_COLORS.get(name, "#999999") for name in names]
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1])
    total = sum(values) or 1
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value} ({100 * value / total:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, max(values, default=1) * 1.22)
    _style(ax, "State distribution", "frames", "")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def annotate_images(images_dir: Path, labels_dir: Path | None, weights: str,
                    out_dir: Path, class_id: int = 0, limit: int = 12,
                    conf: float = 0.45) -> dict | None:
    try:
        import cv2
    except ImportError:
        print("opencv not available - skipping image evaluation")
        return None

    from vision.road_scene import BumpDetector

    detector = BumpDetector(weights, conf)
    if detector.load() is None:
        print(f"{weights} not found - skipping image evaluation")
        return None

    paths = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not paths:
        print(f"no images in {images_dir}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, tp = fp = fn = 0
    saved = []

    for path in paths[:limit]:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        detections = detector.detect(image)
        preds = [(d.x1, d.y1, d.x2, d.y2, d.confidence) for d in detections]

        gt = []
        if labels_dir is not None:
            gt = parse_label_file(labels_dir / f"{path.stem}.txt", width, height,
                                  class_id)
            a, b, c = match_boxes(preds, gt)
            tp, fp, fn = tp + a, fp + b, fn + c
            rows += 1

        for det in detections:
            cv2.rectangle(image, (det.x1, det.y1), (det.x2, det.y2), (0, 200, 0), 2)
            cv2.putText(image, f"bump {det.confidence:.2f}", (det.x1, det.y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        for box in gt:
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 0, 255), 1)

        target = out_dir / f"det_{path.stem}.jpg"
        cv2.imwrite(str(target), image)
        saved.append(target)

    result = {"images": len(saved), "saved": saved}
    if rows:
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        result.update({"tp": tp, "fp": fp, "fn": fn, "labelled_images": rows,
                       "precision": precision, "recall": recall, "f1": f1})
    return result


def plot_samples(saved: list[Path], path: Path, max_items: int = 6) -> None:
    try:
        import cv2
    except ImportError:
        return
    items = saved[:max_items]
    if not items:
        return
    cols = 2
    rows = (len(items) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.2 * rows), dpi=120)
    axes = np.atleast_1d(axes).ravel()
    for ax, item in zip(axes, items):
        image = cv2.imdecode(np.fromfile(str(item), dtype=np.uint8), cv2.IMREAD_COLOR)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(item.stem, fontsize=9)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ markdown ---
def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_markdown(metrics, session, image_eval, figures, session_name, generated):
    out = [f"# DrowsyGuard AI - Evaluation Report", "",
           f"*Generated: {generated}*", ""]
    section = 0

    if metrics:
        section += 1
        out += [f"## {section}. Speed-bump detector (YOLOv8)", "",
                _table(["Item", "Value"], [
                    ["Base model", metrics.get("model", "YOLOv8n")],
                    ["Image size", metrics.get("image_size", 640)],
                    ["Epochs", metrics.get("epochs", "-")],
                    ["Validation images", metrics.get("val_images", "-")],
                    ["Validation instances", metrics.get("val_instances", "-")],
                    ["Precision", f"{metrics['precision'] * 100:.1f}%"],
                    ["Recall", f"{metrics['recall'] * 100:.1f}%"],
                    ["mAP@50", f"{metrics['map50'] * 100:.1f}%"],
                    ["mAP@50-95", f"{metrics['map50_95'] * 100:.1f}%"],
                ]), ""]
        if "figures/training_metrics.png" in figures:
            out += ["![Training metrics](figures/training_metrics.png)", ""]

    if image_eval and "precision" in image_eval:
        section += 1
        out += [f"## {section}. Speed-bump field test", "",
                _table(["Metric", "Value"], [
                    ["Images", image_eval["labelled_images"]],
                    ["True positives", image_eval["tp"]],
                    ["False positives", image_eval["fp"]],
                    ["False negatives", image_eval["fn"]],
                    ["Precision", f"{image_eval['precision'] * 100:.1f}%"],
                    ["Recall", f"{image_eval['recall'] * 100:.1f}%"],
                    ["F1", f"{image_eval['f1'] * 100:.1f}%"],
                ]), ""]
        if "figures/bump_samples.png" in figures:
            out += ["![Bump detections](figures/bump_samples.png)", ""]

    if session:
        summary = session["summary"]
        section += 1
        out += [f"## {section}. Driver-state session", "",
                f"Log: `{Path(session_name).as_posix()}`", "",
                _table(["Metric", "Value"], [
                    ["Frames", summary["frames"]],
                    ["Duration", f"{summary['duration_sec']:.1f} s"],
                    ["Drowsy frames", f"{summary['drowsy_pct']:.1f}%"],
                    ["Distracted frames", f"{summary['distracted_pct']:.1f}%"],
                    ["Unknown frames", f"{summary['unknown_pct']:.1f}%"],
                    ["Face coverage", f"{summary['face_coverage'] * 100:.1f}%"],
                    ["Mean PERCLOS", "-" if summary["perclos_mean"] is None
                     else f"{summary['perclos_mean']:.1f}%"],
                    ["Max PERCLOS", "-" if summary["perclos_max"] is None
                     else f"{summary['perclos_max']:.1f}%"],
                    ["Mean attention", "-" if summary["attention_mean"] is None
                     else f"{summary['attention_mean']:.0f} / 100"],
                ]), ""]
        if session.get("simulated"):
            out += ["> **Note:** this session is a controlled simulation "
                    "(`tools/simulate_session.py`). The inputs are scripted; the "
                    "states come from the project's own `PerclosMeter` and "
                    "`DecisionEngine`.", ""]
        if "figures/session_timeline.png" in figures:
            out += ["![Session timeline](figures/session_timeline.png)", ""]
        if "figures/state_distribution.png" in figures:
            out += ["![State distribution](figures/state_distribution.png)", ""]

    section += 1
    out += [f"## {section}. Reproduce", "", "```bash",
            "python -m unittest discover -s tests -t .",
            "python tools/simulate_session.py     # scripted session log",
            "python tools/make_report.py          # rebuild this report",
            "```", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------- main ---
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=None,
                        help="session CSV (default: simulation log, else real log)")
    parser.add_argument("--images", default=None, help="folder of test images")
    parser.add_argument("--labels", default=None, help="folder of YOLO labels")
    parser.add_argument("--weights", default="bump_model.pt")
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--out", default="report")
    parser.add_argument("--class-id", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = None
    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        plot_training(metrics, figures_dir / "training_metrics.png")
    else:
        print(f"metrics file not found: {metrics_path}")

    log_path = Path(args.log) if args.log else (
        Path(DEFAULT_SIM_LOG) if Path(DEFAULT_SIM_LOG).exists()
        else Path(DEFAULT_REAL_LOG))

    session = None
    figure_names = set()
    if metrics:
        figure_names.add("figures/training_metrics.png")
    if log_path.exists():
        frame = load_session(log_path)
        session = {"summary": summarize_session(frame),
                   "simulated": "simulation" in log_path.name}
        plot_timeline(frame, figures_dir / "session_timeline.png")
        plot_states(session["summary"], figures_dir / "state_distribution.png")
        figure_names |= {"figures/session_timeline.png",
                         "figures/state_distribution.png"}
    else:
        print(f"session log not found: {log_path}")

    image_eval = None
    if args.images:
        images_dir = Path(args.images)
        labels_dir = Path(args.labels) if args.labels else None
        if labels_dir is None and (images_dir.parent / "labels").exists():
            labels_dir = images_dir.parent / "labels"
        image_eval = annotate_images(images_dir, labels_dir, args.weights,
                                     out / "bump_samples", args.class_id)
        if image_eval and image_eval["saved"]:
            plot_samples(image_eval["saved"], figures_dir / "bump_samples.png")
            figure_names.add("figures/bump_samples.png")

    markdown = build_markdown(metrics, session, image_eval, figure_names,
                              str(log_path), datetime.now().strftime("%Y-%m-%d %H:%M"))
    (out / "report.md").write_text(markdown, encoding="utf-8")

    print(f"report written -> {out / 'report.md'}")
    print(f"figures       -> {figures_dir}")
    if session:
        summary = session["summary"]
        print(f"session: {summary['frames']} frames, "
              f"drowsy {summary['drowsy_pct']:.1f}%, "
              f"distracted {summary['distracted_pct']:.1f}%")


if __name__ == "__main__":
    main()
