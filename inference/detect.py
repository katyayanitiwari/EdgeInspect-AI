"""Laptop inference: image / folder / webcam -> YOLO boxes -> severity engine -> annotated output.

Usage (project root):
  .\.venv\Scripts\python.exe inference\detect.py --weights experiments\EXP-001\weights\best.pt --source dataset\processed\neu_det\images\test\crazing_11.jpg
  .\.venv\Scripts\python.exe inference\detect.py --weights ... --source dataset\processed\neu_det\images\test   (whole folder)
  .\.venv\Scripts\python.exe inference\detect.py --weights ... --source 0        (webcam 0; q = quit, s = save frame)

Outputs (image/folder mode): inference/output/<run>/<image>.jpg (annotated) + <image>.json (defects, severity,
decision, latency) + summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.severity import assess, load_rules  # noqa: E402

COLORS = {"minor": (0, 200, 255), "major": (0, 128, 255), "critical": (0, 0, 255)}  # BGR
BOX_COLORS = [(255, 80, 40), (0, 220, 255), (220, 60, 220), (60, 220, 60),
              (255, 170, 20), (200, 120, 0)]  # BGR; separate IDs are easier to follow
DECISION_COLORS = {"PASS": (0, 170, 0), "PASS_WITH_NOTE": (0, 170, 0), "REWORK": (0, 128, 255),
                   "REJECT": (0, 0, 255), "MANUAL_REVIEW": (200, 120, 0)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def run_one(model, frame, rules, imgsz):
    """Detect + assess one BGR frame. Returns (report, annotated frame)."""
    t0 = time.perf_counter()
    res = model.predict(frame, imgsz=imgsz, conf=rules["confidence"]["min"], verbose=False)[0]
    total_ms = (time.perf_counter() - t0) * 1000
    dets = [{"cls": res.names[int(c)], "conf": float(p), "xyxy": b.tolist()}
            for b, c, p in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf)]
    h, w = frame.shape[:2]
    report = assess(dets, w, h, rules)
    report["latency_ms"] = {"model_inference": round(res.speed["inference"], 1),
                            "end_to_end": round(total_ms, 1)}
    return report, draw(frame, report)


def draw(frame, report):
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(1.0, w / 640)  # scale text so it is readable on small (200x200) and large frames
    if w < 400:  # NEU-DET images are 200x200 -> upscale for a readable picture
        f = 640 / w
        out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    else:
        f = 1.0
    for d in report["defects"]:
        x1, y1, x2, y2 = (int(v * f) for v in d["xyxy"])
        if d.get("id", "").startswith("D") and d["id"][1:].isdigit():
            c = BOX_COLORS[(int(d["id"][1:]) - 1) % len(BOX_COLORS)]
        else:
            c = COLORS[d["severity"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        ref = f"{d['id']} " if d.get("id") else ""
        label = f"{ref}{d['cls']} {d['conf']:.2f} {d['severity']}" + (" ?" if d["uncertain"] else "")
        cv2.putText(out, label, (x1 + 2, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, c, 1)
    banner = f"{report['decision']}  ({report['severity']}, {report['defect_count']} defects)"
    cv2.rectangle(out, (0, 0), (out.shape[1], int(28 * scale)), (30, 30, 30), -1)
    cv2.putText(out, banner, (8, int(20 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
                DECISION_COLORS[report["decision"]], 2)
    return out


def run_files(model, source, rules, args):
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXT) if source.is_dir() else [source]
    if not files:
        raise SystemExit(f"No images found at {source}")
    out_dir = ROOT / "inference" / "output" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, lat = {}, []
    for p in files:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"skip (cannot read): {p}")
            continue
        report, annotated = run_one(model, frame, rules, args.imgsz)
        report["image"] = str(p)
        cv2.imwrite(str(out_dir / f"{p.stem}.jpg"), annotated)
        (out_dir / f"{p.stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        counts[report["decision"]] = counts.get(report["decision"], 0) + 1
        lat.append(report["latency_ms"]["end_to_end"])
        print(f"{p.name:<28} {report['decision']:<15} {report['severity']:<9} "
              f"{report['defect_count']} defects  {report['latency_ms']['end_to_end']:.1f} ms")
    lat_sorted = sorted(lat)
    summary = {"weights": str(args.weights), "source": str(source), "imgsz": args.imgsz, "images": len(lat),
               "decisions": counts, "end_to_end_ms_median": round(lat_sorted[len(lat) // 2], 1),
               "note": "first image includes model warm-up; use evaluation/evaluate.py for benchmark latency"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{json.dumps(summary, indent=2)}\nSaved to {out_dir.relative_to(ROOT)}")


def run_webcam(model, index, rules, args):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open webcam {index}")
    out_dir = ROOT / "inference" / "output" / args.run
    print("Webcam running: q = quit, s = save frame + report")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break
        report, annotated = run_one(model, frame, rules, args.imgsz)
        fps = 1000 / max(report["latency_ms"]["end_to_end"], 1e-3)
        cv2.putText(annotated, f"{fps:.1f} FPS", (8, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("EdgeInspect-AI", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = time.strftime("frame_%Y%m%d_%H%M%S")
            cv2.imwrite(str(out_dir / f"{stem}.jpg"), annotated)
            (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"saved {stem}: {report['decision']}")
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--source", required=True, help="image file, folder, or webcam index (0)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--rules", type=Path, default=ROOT / "inference" / "severity_rules.yaml")
    ap.add_argument("--run", default=time.strftime("run_%Y%m%d_%H%M%S"), help="output subfolder name")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")
    model = YOLO(str(args.weights))
    rules = load_rules(args.rules)
    if args.source.isdigit():
        run_webcam(model, int(args.source), rules, args)
    else:
        src = Path(args.source)
        if not src.exists():
            raise SystemExit(f"Source not found: {src}")
        run_files(model, src, rules, args)


if __name__ == "__main__":
    main()
