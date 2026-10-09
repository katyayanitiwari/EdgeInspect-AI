"""
Train a YOLOv8 detector on NEU-DET.

Usage (from the project root, venv active):
  python training/train.py --name EXP-001 --imgsz 640 --epochs 100
  python training/train.py --name SANITY --epochs 1          # quick smoke test

Outputs go to experiments/<name>/ (weights/best.pt, results.csv, plots, args.yaml).
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="experiment id, e.g. EXP-001")
p.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained start weights")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--device", default="cpu", help="'cpu' or GPU index like '0'")
p.add_argument("--workers", type=int, default=4)
args = p.parse_args()

# Start from COCO-pretrained weights (transfer learning): the early layers already
# know edges/textures, so 1,259 training images are enough to adapt them.
model = YOLO(str(ROOT / "models" / args.model) if (ROOT / "models" / args.model).exists() else args.model)

model.train(
    data=str(ROOT / "training" / "neu_det.yaml"),
    project=str(ROOT / "experiments"),
    name=args.name,
    exist_ok=False,          # never silently overwrite an experiment
    imgsz=args.imgsz,
    epochs=args.epochs,
    batch=args.batch,
    device=args.device,
    workers=args.workers,

    # --- optimiser & learning-rate schedule (explicit, not "auto", so it is reproducible)
    optimizer="SGD",
    lr0=0.01,                # start learning rate
    lrf=0.01,                # final LR = lr0 * lrf (linear decay)
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3,

    # --- early stopping & checkpoints
    patience=25,             # stop if val fitness does not improve for 25 epochs
    save=True,               # keeps weights/best.pt and weights/last.pt
    save_period=-1,

    # --- reproducibility
    seed=42,
    deterministic=True,

    # --- augmentation (chosen for physically plausible steel-surface variation)
    hsv_h=0.0,               # images are grayscale -> hue shifts are meaningless
    hsv_s=0.0,               # same for saturation
    hsv_v=0.3,               # brightness change = real lighting variation on a line
    degrees=0.0,             # arbitrary rotation inflates axis-aligned boxes; avoided
    translate=0.1,           # part not perfectly centred under the camera
    scale=0.5,               # camera distance / defect size variation
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,              # a steel surface has no left/right meaning
    flipud=0.5,              # ...nor up/down
    mosaic=1.0,              # 4 images tiled -> more defects per batch, good for small data
    close_mosaic=10,         # last 10 epochs without mosaic -> matches real single images
    mixup=0.0,               # blending two images creates physically impossible defects
    copy_paste=0.0,

    plots=True,
    verbose=True,
)
