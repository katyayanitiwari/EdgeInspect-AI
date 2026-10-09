"""
Prepare the NEU-DET dataset for YOLO training.

What this script does (in order):
  1. Inspect   - count images/annotations, check image size and colour mode
  2. Clean     - detect corrupted images and exact duplicates (MD5 hash)
  3. Verify    - parse every Pascal VOC XML, check class names and box validity
  4. Convert   - VOC (xmin,ymin,xmax,ymax in pixels) -> YOLO (class cx cy w h, normalised 0-1)
  5. Split     - stratified 70/15/15 train/val/test with a fixed seed
  6. Report    - write dataset/processed/neu_det/prep_report.json + data yaml

Usage (from the project root, with the venv active):
  python scripts/prepare_neu_det.py
"""

import hashlib
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# The GitHub mirror stores the 1,800 images in two folder pairs (1,770 + 30).
# We merge them and make our own stratified split below.
SRC = ROOT / "dataset" / "raw" / "neu_src"
SRC_PAIRS = [(SRC / "IMAGES", SRC / "ANNOTATIONS"),
             (SRC / "Validation_Images", SRC / "Validation_Annotations")]
OUT = ROOT / "dataset" / "processed" / "neu_det"

# Fixed class order -> class IDs never change between experiments.
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SEED = 42
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}


def class_from_filename(stem: str) -> str:
    # e.g. "rolled-in_scale_123" -> "rolled-in_scale"
    return re.sub(r"_\d+$", "", stem)


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bb = obj.find("bndbox")
        box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        objects.append((name, box))
    return w, h, objects


def main():
    report = {"issues": defaultdict(list)}
    images, xmls = [], {}
    for img_dir, ann_dir in SRC_PAIRS:
        assert img_dir.is_dir() and ann_dir.is_dir(), f"Missing {img_dir} or {ann_dir}"
        images += [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        xmls.update({p.stem: p for p in ann_dir.glob("*.xml")})
    images.sort(key=lambda p: p.name)
    report["n_images_found"] = len(images)
    report["n_annotations_found"] = len(xmls)

    sizes, modes, hashes = Counter(), Counter(), defaultdict(list)
    samples = []  # (image_path, class_for_stratify, yolo_lines)
    box_counts, boxes_per_img = Counter(), Counter()
    box_rel_areas = defaultdict(list)

    for img_path in images:
        # --- corrupted image check
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                iw, ih = im.size
                sizes[f"{iw}x{ih}"] += 1
                modes[im.mode] += 1
        except Exception as e:
            report["issues"]["corrupted_image"].append(f"{img_path.name}: {e}")
            continue

        # --- duplicate check
        hashes[hashlib.md5(img_path.read_bytes()).hexdigest()].append(img_path.name)

        # --- annotation check
        xml_path = xmls.get(img_path.stem)
        if xml_path is None:
            report["issues"]["missing_annotation"].append(img_path.name)
            continue
        try:
            aw, ah, objects = parse_voc(xml_path)
        except Exception as e:
            report["issues"]["unparseable_xml"].append(f"{xml_path.name}: {e}")
            continue
        if (aw, ah) != (iw, ih):
            report["issues"]["xml_size_mismatch"].append(f"{img_path.name}: xml {aw}x{ah} vs img {iw}x{ih}")

        lines = []
        for name, (x1, y1, x2, y2) in objects:
            if name not in CLASSES:
                report["issues"]["unknown_class"].append(f"{img_path.name}: '{name}'")
                continue
            # clip to image bounds (VOC is 1-based; small overflows are common)
            cx1, cy1 = max(0.0, min(x1, iw)), max(0.0, min(y1, ih))
            cx2, cy2 = max(0.0, min(x2, iw)), max(0.0, min(y2, ih))
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                report["issues"]["box_clipped"].append(f"{img_path.name}: {[x1, y1, x2, y2]}")
            bw, bh = cx2 - cx1, cy2 - cy1
            if bw <= 1 or bh <= 1:
                report["issues"]["degenerate_box_dropped"].append(f"{img_path.name}: {[x1, y1, x2, y2]}")
                continue
            cid = CLASSES.index(name)
            lines.append(f"{cid} {(cx1 + bw / 2) / iw:.6f} {(cy1 + bh / 2) / ih:.6f} {bw / iw:.6f} {bh / ih:.6f}")
            box_counts[name] += 1
            box_rel_areas[name].append((bw * bh) / (iw * ih))

        if not lines:
            report["issues"]["image_with_no_valid_boxes"].append(img_path.name)
        boxes_per_img[len(lines)] += 1
        samples.append((img_path, class_from_filename(img_path.stem), lines))

    # --- exact duplicates: keep the first, drop the rest
    dup_drop = set()
    for names in hashes.values():
        if len(names) > 1:
            report["issues"]["exact_duplicates"].append(names)
            dup_drop.update(names[1:])
    samples = [s for s in samples if s[0].name not in dup_drop]

    # --- stratified split
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)
    split_of = {}
    split_counts = defaultdict(Counter)
    for cls, items in sorted(by_class.items()):
        items.sort(key=lambda s: s[0].name)
        rng.shuffle(items)
        n = len(items)
        n_train, n_val = round(n * SPLIT["train"]), round(n * SPLIT["val"])
        for i, s in enumerate(items):
            sp = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            split_of[s[0].name] = sp
            split_counts[sp][cls] += 1

    # --- write YOLO dataset
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in SPLIT:
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    for img_path, _, lines in samples:
        sp = split_of[img_path.name]
        shutil.copy2(img_path, OUT / "images" / sp / img_path.name)
        (OUT / "labels" / sp / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    # --- data yaml (Ultralytics format)
    yaml_path = ROOT / "training" / "neu_det.yaml"
    yaml_path.write_text(
        f"# NEU-DET, generated by scripts/prepare_neu_det.py (seed={SEED})\n"
        f"path: {OUT.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )

    # --- report
    import statistics as st
    report.update({
        "image_sizes": dict(sizes),
        "image_modes": dict(modes),
        "n_samples_used": len(samples),
        "boxes_per_class": dict(box_counts),
        "boxes_per_image_histogram": dict(sorted(boxes_per_img.items())),
        "box_relative_area_median": {c: round(st.median(v), 4) for c, v in box_rel_areas.items()},
        "box_relative_area_min": {c: round(min(v), 5) for c, v in box_rel_areas.items()},
        "split_counts": {sp: dict(c) for sp, c in split_counts.items()},
        "issues": {k: v for k, v in report["issues"].items()},
        "issue_counts": {k: len(v) for k, v in report["issues"].items()},
    })
    (OUT / "prep_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k != "issues"}, indent=2))
    print(f"\nWrote dataset to {OUT}\nWrote config to {yaml_path}")


if __name__ == "__main__":
    main()
