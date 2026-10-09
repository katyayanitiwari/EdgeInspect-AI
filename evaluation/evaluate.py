"""
Evaluate a trained experiment on the held-out TEST split and log every metric.

Usage (from the project root, venv active):
  python evaluation/evaluate.py --name EXP-001
  python evaluation/evaluate.py --name EXP-001 --notes "baseline, CPU training"

Measures:
  accuracy : precision, recall, F1, mAP50, mAP50-95, per-class AP50
  model    : parameters, GFLOPs (at eval imgsz), weight file size
  speed    : CPU latency per image (batch 1), split into pre/inference/post, and FPS

Writes:
  evaluation/<name>/metrics.json        full measurement record
  evaluation/<name>/test/...            confusion matrix, PR curves, etc. (Ultralytics plots)
  experiments/results.csv               one row per evaluation (append)
  experiments/EXPERIMENTS.md            results table regenerated from results.csv
"""

import argparse
import csv
import datetime as dt
import json
import platform
import statistics as st
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "neu_det.yaml"
CSV_PATH = ROOT / "experiments" / "results.csv"
MD_PATH = ROOT / "experiments" / "EXPERIMENTS.md"
FIELDS = ["exp_id", "date", "model", "precision_fmt", "imgsz", "epochs_run", "P", "R", "F1",
          "mAP50", "mAP50_95", "params_M", "GFLOPs", "size_MB", "cpu_infer_ms", "cpu_total_ms",
          "cpu_fps", "notes"]

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="experiment folder under experiments/")
p.add_argument("--weights", default=None, help="default: experiments/<name>/weights/best.pt")
p.add_argument("--imgsz", type=int, default=None, help="default: imgsz used in training")
p.add_argument("--latency-images", type=int, default=100)
p.add_argument("--notes", default="")
args = p.parse_args()

exp_dir = ROOT / "experiments" / args.name
weights = Path(args.weights) if args.weights else exp_dir / "weights" / "best.pt"
assert weights.exists(), f"weights not found: {weights}"

# training settings, recorded by Ultralytics in args.yaml
train_args = {}
if (exp_dir / "args.yaml").exists():
    import yaml
    train_args = yaml.safe_load((exp_dir / "args.yaml").read_text())
imgsz = args.imgsz or int(train_args.get("imgsz", 640))
epochs_run = ""
if (exp_dir / "results.csv").exists():
    with open(exp_dir / "results.csv") as f:
        epochs_run = sum(1 for _ in f) - 1

out_dir = ROOT / "evaluation" / args.name
out_dir.mkdir(parents=True, exist_ok=True)
model = YOLO(str(weights))

# ---------------- accuracy on the TEST split ----------------
m = model.val(data=str(DATA), split="test", imgsz=imgsz, batch=16, device="cpu",
              project=str(out_dir), name="test", exist_ok=True, plots=True, verbose=False)
P, R = float(m.box.mp), float(m.box.mr)
F1 = 2 * P * R / (P + R) if (P + R) else 0.0
names = m.names
per_class = {names[int(c)]: {"AP50": round(float(m.box.ap50[i]), 4), "AP50_95": round(float(m.box.ap[i]), 4)}
             for i, c in enumerate(m.box.ap_class_index)}

# ---------------- model complexity ----------------
params = sum(x.numel() for x in model.model.parameters())
gflops = get_flops(model.model, imgsz)
size_mb = weights.stat().st_size / 1e6

# ---------------- CPU latency (batch 1, like a live camera) ----------------
test_imgs = sorted((ROOT / "dataset" / "processed" / "neu_det" / "images" / "test").glob("*.jpg"))
test_imgs = test_imgs[: args.latency_images]
for img in test_imgs[:10]:  # warm-up, not measured
    model.predict(str(img), imgsz=imgsz, device="cpu", verbose=False)
pre, inf, post, wall = [], [], [], []
for img in test_imgs:
    t0 = time.perf_counter()
    r = model.predict(str(img), imgsz=imgsz, device="cpu", verbose=False)[0]
    wall.append((time.perf_counter() - t0) * 1000)
    pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
infer_ms = st.median(inf)
total_ms = st.median([a + b + c for a, b, c in zip(pre, inf, post)])

record = {
    "exp_id": args.name,
    "date": dt.datetime.now().isoformat(timespec="seconds"),
    "weights": str(weights.relative_to(ROOT)),
    "model": Path(str(train_args.get("model", "unknown"))).stem,
    "precision_fmt": "FP32",
    "imgsz": imgsz,
    "epochs_run": epochs_run,
    "accuracy_test_split": {"P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
                            "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                            "per_class": per_class},
    "complexity": {"params": params, "GFLOPs": round(gflops, 2), "size_MB": round(size_mb, 2)},
    "cpu_latency_ms_median": {"preprocess": round(st.median(pre), 2), "inference": round(infer_ms, 2),
                              "postprocess": round(st.median(post), 2), "total": round(total_ms, 2),
                              "wall_clock_incl_file_read": round(st.median(wall), 2),
                              "n_images": len(test_imgs), "batch": 1},
    "cpu_fps": round(1000 / total_ms, 1),
    "environment": {"cpu": platform.processor(), "python": platform.python_version(),
                    "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                    "torch_threads": torch.get_num_threads()},
    "notes": args.notes,
}
(out_dir / "metrics.json").write_text(json.dumps(record, indent=2))

# ---------------- append to results.csv ----------------
row = {"exp_id": args.name, "date": record["date"], "model": record["model"], "precision_fmt": "FP32",
       "imgsz": imgsz, "epochs_run": epochs_run, **{k: record["accuracy_test_split"][k] for k in
       ("P", "R", "F1", "mAP50", "mAP50_95")}, "params_M": round(params / 1e6, 3),
       "GFLOPs": record["complexity"]["GFLOPs"], "size_MB": record["complexity"]["size_MB"],
       "cpu_infer_ms": round(infer_ms, 2), "cpu_total_ms": round(total_ms, 2),
       "cpu_fps": record["cpu_fps"], "notes": args.notes}
new_file = not CSV_PATH.exists()
with open(CSV_PATH, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    w.writerow(row)

# ---------------- regenerate the table in EXPERIMENTS.md ----------------
with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))
header = ("| Exp ID | Date | Model | Fmt | Img | Epochs | P | R | F1 | mAP50 | mAP50-95 | Params (M) "
          "| GFLOPs | Size (MB) | CPU infer (ms) | CPU total (ms) | CPU FPS | Notes |\n"
          "|" + "---|" * 18 + "\n")
table = header + "".join("| " + " | ".join(str(r[k]) for k in FIELDS) + " |\n" for r in rows)
start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
md = MD_PATH.read_text(encoding="utf-8")
md = md[: md.index(start) + len(start)] + "\n" + table + md[md.index(end):]
MD_PATH.write_text(md, encoding="utf-8")

print(json.dumps(record, indent=2))
print(f"\nSaved {out_dir / 'metrics.json'}\nUpdated {CSV_PATH}\nUpdated {MD_PATH}")
