"""Compare exported ONNX predictions with the source PyTorch checkpoint.

Run from the project root:
  .\.venv\Scripts\python.exe evaluation\compare_onnx.py
  .\.venv\Scripts\python.exe evaluation\compare_onnx.py --limit 0  # all test images

Uses the first 30 held-out test images by default, CPU inference, and identical image size,
confidence, and NMS settings. Writes evaluation/EXP-002/onnx_parity.json.
This checks prediction agreement and speed; it does not claim Pi/Hailo performance.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import onnxruntime
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
PT_PATH = ROOT / "experiments" / "EXP-002" / "weights" / "best.pt"
ONNX_PATH = ROOT / "experiments" / "EXP-002" / "weights" / "best.onnx"
TEST_DIR = ROOT / "dataset" / "processed" / "neu_det" / "images" / "test"
OUTPUT = ROOT / "evaluation" / "EXP-002" / "onnx_parity.json"
IMGSZ = 320
CONF = 0.25
IOU_NMS = 0.7
MATCH_IOU = 0.50


def extract(result):
    boxes = result.boxes
    return {
        "xyxy": boxes.xyxy.cpu().numpy(),
        "cls": boxes.cls.cpu().numpy().astype(int),
        "conf": boxes.conf.cpu().numpy(),
        "infer_ms": float(result.speed.get("inference", 0.0)),
    }


def match_boxes(left, right):
    """Greedy same-class matching; report coordinate overlap and confidence drift."""
    a, b = left["xyxy"], right["xyxy"]
    available = set(range(len(b)))
    ious, confidence_deltas = [], []
    for i, box in enumerate(a):
        if not available:
            continue
        candidates = []
        for j in available:
            if left["cls"][i] != right["cls"][j]:
                continue
            other = b[j]
            ix1, iy1 = max(box[0], other[0]), max(box[1], other[1])
            ix2, iy2 = min(box[2], other[2]), min(box[3], other[3])
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            area_b = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
            union = area_a + area_b - intersection
            candidates.append((intersection / union if union else 0.0, j))
        if not candidates:
            continue
        best_iou, best_j = max(candidates)
        if best_iou >= MATCH_IOU:
            available.remove(best_j)
            ious.append(best_iou)
            confidence_deltas.append(abs(float(left["conf"][i] - right["conf"][best_j])))
    return ious, confidence_deltas


def run(model, source):
    start = time.perf_counter()
    result = model.predict(source, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS,
                           max_det=300, device="cpu", verbose=False)[0]
    wall_ms = (time.perf_counter() - start) * 1000
    return extract(result), wall_ms


def main():
    if not PT_PATH.exists() or not ONNX_PATH.exists():
        raise SystemExit("Missing EXP-002 best.pt or best.onnx. Export ONNX first.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30,
                        help="number of sorted test images to compare; 0 means all (default: 30)")
    args = parser.parse_args()
    all_images = sorted(TEST_DIR.glob("*.jpg"))
    if not all_images:
        raise SystemExit(f"No test images found in {TEST_DIR}")
    images = all_images if args.limit == 0 else all_images[:args.limit]

    pt_model, onnx_model = YOLO(str(PT_PATH)), YOLO(str(ONNX_PATH))
    warm_images = [path for path in images[:5] if cv2.imread(str(path)) is not None]
    if not warm_images:
        raise SystemExit("Could not read any test images.")
    for path in warm_images:
        run(pt_model, str(path))
        run(onnx_model, str(path))

    count_pt = count_onnx = matched = exact_count_images = 0
    all_ious, all_conf_deltas = [], []
    pt_wall, onnx_wall, pt_infer, onnx_infer = [], [], [], []
    processed = 0
    for image_index, path in enumerate(images, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        pt, pt_ms = run(pt_model, str(path))
        onnx, onnx_ms = run(onnx_model, str(path))
        count_pt += len(pt["cls"])
        count_onnx += len(onnx["cls"])
        ious, deltas = match_boxes(pt, onnx)
        matched += len(ious)
        all_ious.extend(ious)
        all_conf_deltas.extend(deltas)
        exact_count_images += len(pt["cls"]) == len(onnx["cls"])
        pt_wall.append(pt_ms)
        onnx_wall.append(onnx_ms)
        pt_infer.append(pt["infer_ms"])
        onnx_infer.append(onnx["infer_ms"])
        processed += 1
        if image_index % 10 == 0 or image_index == len(images):
            print(f"Compared {image_index}/{len(images)} images")

    record = {
        "source_weights": str(PT_PATH.relative_to(ROOT)),
        "exported_weights": str(ONNX_PATH.relative_to(ROOT)),
        "dataset_split": "NEU-DET held-out test",
        "images": processed,
        "test_split_images_available": len(all_images),
        "comparison_limit": args.limit,
        "settings": {"imgsz": IMGSZ, "confidence": CONF, "nms_iou": IOU_NMS,
                     "box_match_iou": MATCH_IOU, "device": "CPU", "batch": 1},
        "prediction_agreement": {
            "pytorch_detection_count": count_pt,
            "onnx_detection_count": count_onnx,
            "same_class_boxes_matched_iou_ge_0_50": matched,
            "mean_matched_box_iou": round(float(statistics.mean(all_ious)), 5) if all_ious else None,
            "mean_absolute_confidence_difference": round(float(statistics.mean(all_conf_deltas)), 5)
                if all_conf_deltas else None,
            "images_with_equal_detection_counts": exact_count_images,
        },
        "cpu_latency_ms_median": {
            "pytorch_model_inference": round(statistics.median(pt_infer), 2),
            "onnx_model_inference": round(statistics.median(onnx_infer), 2),
            "pytorch_call_wall_time": round(statistics.median(pt_wall), 2),
            "onnx_call_wall_time": round(statistics.median(onnx_wall), 2),
        },
        "model_size_MB": {
            "pytorch": round(PT_PATH.stat().st_size / 1e6, 2),
            "onnx": round(ONNX_PATH.stat().st_size / 1e6, 2),
        },
        "environment": {
            "cpu": platform.processor(), "python": platform.python_version(),
            "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "notes": "Laptop CPU only. Does not represent Raspberry Pi or Hailo performance.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()

