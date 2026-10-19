"""
Train a YOLOv8 speed-bump detector and export ``bump_model.pt``.

Local training needs a CUDA build of PyTorch for GPU use. Without CUDA it
still works, but on CPU it is slow.

    python train_bump.py --data datasets/bump/data.yaml --epochs 50 --batch 8

The produced ``bump_model.pt`` is picked up automatically by ``yolo_phase2.py``.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="base weights (yolov8n/s/m .pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None,
                        help="0 = first GPU, 'cpu' = CPU (default: auto)")
    parser.add_argument("--patience", type=int, default=15,
                        help="early-stopping patience")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="bump")
    parser.add_argument("--out", default="bump_model.pt",
                        help="where to copy the best weights")
    return parser.parse_args(argv)


def pick_device(requested: str | None) -> str | int:
    import torch
    if requested is not None:
        return requested
    return 0 if torch.cuda.is_available() else "cpu"


def main(argv=None):
    args = parse_args(argv)
    data = Path(args.data).expanduser().resolve()
    if not data.exists():
        raise SystemExit(f"data.yaml not found: {data}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Missing ultralytics. Run: pip install -r requirements-yolo.txt")

    device = pick_device(args.device)
    if device == "cpu":
        print("WARNING: CUDA is not available - training on CPU will be slow.")
        print("         For GPU on Kaggle use notebooks/train_bump_kaggle.ipynb.")

    print(f"Training {args.model} -> {args.out}")
    print(f"  data={data}  epochs={args.epochs}  imgsz={args.imgsz}  "
          f"batch={args.batch}  device={device}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        print(f"WARNING: best weights not found at {best}")
        return 1

    shutil.copy2(best, args.out)
    print(f"\nOK: {args.out}  (from {best})")
    print(f"Next: keep {Path(args.out).name} next to yolo_phase2.py and run it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
