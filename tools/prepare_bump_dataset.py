"""
Prepare a single-class YOLO dataset for speed-bump detection.

It scans a raw dataset, finds which class id means "speed bump" (from any
``data.yaml``), keeps only those boxes, rewrites them as class 0, splits into
train/val and writes a fresh ``data.yaml`` ready for ``train_bump.py``.

Usage:
    python tools/prepare_bump_dataset.py "C:\\path\\to\\Urban Road Hazard Dataset"
    python tools/prepare_bump_dataset.py <raw> --out datasets/bump --val-fraction 0.15
    python tools/prepare_bump_dataset.py <raw> --class-id 0 --class-name "Speed Bumps"
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_HINTS = ("bump", "speed")


# --------------------------------------------------------------------------- #
# detection helpers
# --------------------------------------------------------------------------- #
def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("Missing pyyaml. Run: pip install pyyaml")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8-sig", errors="ignore")) or {}
    except Exception:
        return {}


def find_bump_class(root: Path, hint: str | None = None):
    """Return ``(class_id, class_name, yaml_path)`` for the bump class."""
    hints = (hint.lower(),) if hint else CLASS_HINTS
    for yaml_path in sorted(root.rglob("*.yaml")):
        names = load_yaml(yaml_path).get("names")
        if not names:
            continue
        if isinstance(names, dict):
            items = [(int(key), str(value)) for key, value in names.items()]
        else:
            items = list(enumerate(str(name) for name in names))
        for idx, name in items:
            if any(hint_text in name.lower() for hint_text in hints):
                return idx, name, yaml_path
    return None, None, None


def detect_single_class(root: Path):
    """Fallback: if every label uses class 0, treat it as the bump class."""
    found = set()
    for label in root.rglob("*.txt"):
        text = label.read_text(encoding="utf-8-sig", errors="ignore")
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                found.add(int(float(parts[0])))
            except ValueError:
                continue
        if len(found) > 5:
            break
    return 0 if found == {0} else None


def build_label_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for label in root.rglob("*.txt"):
        index.setdefault(label.stem, []).append(label)
    return index


def label_for_image(image: Path, root: Path,
                    label_index: dict[str, list[Path]]) -> Path | None:
    """Find the YOLO label file that belongs to ``image``."""
    try:
        parts = image.relative_to(root).parts
    except ValueError:
        parts = image.parts
    if "images" in parts:
        pos = len(parts) - 1 - parts[::-1].index("images")
        candidate = root.joinpath(*parts[:pos], "labels", *parts[pos + 1:]).with_suffix(".txt")
        if candidate.exists():
            return candidate
    same_folder = image.with_suffix(".txt")
    if same_folder.exists():
        return same_folder
    candidates = label_index.get(image.stem, [])
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if candidate.parent.name == "labels":
            return candidate
    return candidates[0] if candidates else None


def bump_lines(label_path: Path, bump_id: int) -> list[str]:
    """Keep only bump boxes and rewrite their class id to 0."""
    kept = []
    try:
        text = label_path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return kept
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id == bump_id:
            # keep the 4 coordinates, drop a possible confidence column
            kept.append("0 " + " ".join(parts[1:5]))
    return kept


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="raw dataset folder")
    parser.add_argument("--out", default="datasets/bump", help="output folder (default: datasets/bump)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-name", default=None,
                        help="substring to match in data.yaml names (e.g. 'bump')")
    parser.add_argument("--class-id", type=int, default=None,
                        help="force the class id when the YAML is missing/unclear")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Source folder not found: {root}")

    bump_id, bump_name, yaml_path = find_bump_class(root, args.class_name)
    if args.class_id is not None:
        bump_id = int(args.class_id)
        bump_name = args.class_name or f"class{bump_id}"
        yaml_path = None
    if bump_id is None:
        bump_id = detect_single_class(root)
        bump_name = args.class_name or "speed_bump"
    if bump_id is None:
        raise SystemExit(
            "Could not detect the speed-bump class.\n"
            "Run again with:  --class-id <id> --class-name 'Speed Bumps'"
        )

    print(f"Dataset : {root}")
    print(f"Class   : id={bump_id} name={bump_name!r}"
          + (f"  (from {yaml_path.name})" if yaml_path else ""))

    label_index = build_label_index(root)
    samples: list[tuple[Path, list[str]]] = []
    images_seen = 0
    for image in root.rglob("*"):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        images_seen += 1
        label = label_for_image(image, root, label_index)
        if label is None:
            continue
        lines = bump_lines(label, bump_id)
        if lines:
            samples.append((image, lines))

    print(f"Images scanned : {images_seen}")
    print(f"With bumps     : {len(samples)}")
    if not samples:
        raise SystemExit("No bump labels found - check --class-id / the dataset layout.")

    random.seed(args.seed)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * max(0.0, min(1.0, args.val_fraction))))
    splits = {"val": samples[:n_val], "train": samples[n_val:]}

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, items in splits.items():
        for index, (image, lines) in enumerate(items):
            stem = f"{split}_{index:06d}"
            shutil.copy2(image, out / "images" / split / (stem + image.suffix.lower()))
            (out / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

    data_yaml = out / "data.yaml"
    class_name = (bump_name or "speed_bump").strip().replace(" ", "_")
    data_yaml.write_text(
        f"path: {out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {class_name}\n",
        encoding="utf-8",
    )

    print("\nOutput dataset:")
    print(f"  train: {len(splits['train'])} images")
    print(f"  val  : {len(splits['val'])} images")
    print(f"  yaml : {data_yaml}")
    print("\nNext:  python train_bump.py --data " + str(data_yaml))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
