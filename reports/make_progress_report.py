"""
Build the vision-model progress report as a PDF.

Usage (from the project root, venv active):
  python reports/make_progress_report.py

Reads measured values from:
  dataset/processed/neu_det/prep_report.json   (dataset facts)
  experiments/SANITY/results.csv               (1-epoch smoke test)
Writes:
  reports/EdgeInspect-AI_Vision_Progress.pdf
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "EdgeInspect-AI_Vision_Progress.pdf"
TMP = ROOT / "reports" / "_figs"
TMP.mkdir(exist_ok=True)

INK = colors.HexColor("#13171b")
MUTED = colors.HexColor("#667079")
RULE = colors.HexColor("#d5dadf")
HEAD_BG = colors.HexColor("#eef0f2")
ACCENT = colors.HexColor("#0b6e7a")
OK, WIP, TODO = colors.HexColor("#1d7a3a"), colors.HexColor("#8a5a00"), colors.HexColor("#5b636b")
BAR = "#2a78d6"

# ---------------- measured inputs ----------------
prep = json.loads((ROOT / "dataset/processed/neu_det/prep_report.json").read_text())
with open(ROOT / "experiments/SANITY/results.csv") as f:
    sanity = next(csv.DictReader(f))
epoch_s = float(sanity["time"])
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
boxes = prep["boxes_per_class"]
area = prep["box_relative_area_median"]
split = prep["split_counts"]
dups = prep["issues"].get("exact_duplicates", [])

# ---------------- figures ----------------
def hbar(values, fname, xlabel, fmt, xmax):
    fig, ax = plt.subplots(figsize=(4.2, 2.3), dpi=200)
    y = range(len(CLASSES))[::-1]
    ax.barh(list(y), values, height=0.5, color=BAR)
    ax.set_yticks(list(y), CLASSES, fontsize=8)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, fontsize=8, color="#3f474f")
    ax.tick_params(axis="x", labelsize=7, colors="#667079")
    ax.grid(axis="x", color="#e6e9ec", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d5dadf")
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, fmt(v), va="center", fontsize=7, color="#13171b")
    fig.tight_layout()
    fig.savefig(TMP / fname)
    plt.close(fig)
    return TMP / fname

f_boxes = hbar([boxes[c] for c in CLASSES], "boxes.png", "boxes (count)", lambda v: f"{v:,}", 1200)
f_area = hbar([area[c] * 100 for c in CLASSES], "area.png", "median box size (% of image area)",
              lambda v: f"{v:.1f}%", 65)

# ---------------- styles ----------------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24,
                    alignment=TA_LEFT, textColor=INK, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16,
                    textColor=INK, spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                    textColor=INK, spaceBefore=4, spaceAfter=3)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5,
                      textColor=INK, spaceAfter=4)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.5, leading=11.5, textColor=MUTED)
EYEBROW = ParagraphStyle("E", parent=SMALL, fontName="Helvetica", fontSize=8, textColor=MUTED)
CELL = ParagraphStyle("C", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Helvetica-Bold")
BUL = ParagraphStyle("BL", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
CALL = ParagraphStyle("CL", parent=BODY, backColor=colors.HexColor("#f2f4f5"), borderColor=ACCENT,
                      borderPadding=(6, 8, 6, 8), leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=10)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def tag(text, color):
    return Paragraph(f'<font color="{color.hexval().replace("0x", "#")}"><b>{text.upper()}</b></font>', CELL)


def table(rows, widths, num_cols=(), header=True):
    data = [[c if not isinstance(c, str) else P(c, CELLB if (header and r == 0) else CELL)
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
    t.setStyle(TableStyle(st))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "EdgeInspect-AI  |  Vision model progress report  |  26 Sep 2026")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


W = A4[0] - 4 * cm
s = []

# ---------------- title ----------------
s += [P("BHARAT AI-SOC CHALLENGE 2026-27  |  PS5 MANUFACTURING &amp; INDUSTRY 4.0", EYEBROW),
      P("EdgeInspect-AI: Vision Model Progress Report", H1),
      P("Steel-surface defect detector for the Raspberry Pi 5 inspection pipeline. "
        "Current phase: training the vision model.", BODY),
      Spacer(1, 4),
      table([["Dataset", "Model", "Current status", "Date"],
             ["NEU-DET", "YOLOv8n (COCO-pretrained)", "Pipeline verified; baseline EXP-001 not trained yet",
              "26 Sep 2026"]], [3 * cm, 4 * cm, 7 * cm, W - 14 * cm]),
      Spacer(1, 6),
      P("<b>Reading this report:</b> values marked <font color='#1d7a3a'><b>MEASURED</b></font> were produced by "
        "our own runs. Values marked <font color='#8a5a00'><b>EXPECTED</b></font> are rough ranges from published "
        "work and are not our results. Blank cells have not been measured yet.", CALL)]

# ---------------- progress ----------------
steps = [
    ("Dataset download", "GitHub mirror, 1,800 images", "Done"),
    ("Dataset inspection", "sizes, colour modes, counts", "Done"),
    ("Dataset structure", "YOLO images/ + labels/ folders", "Done"),
    ("Annotation verification", "script checks + visual check", "Done"),
    ("Train / val / test split", "70 / 15 / 15, stratified, seed 42", "Done"),
    ("Preprocessing", "letterbox resize, 0-1 pixel scaling", "Done"),
    ("Augmentation", "configured in training/train.py", "Done"),
    ("Baseline model", "YOLOv8n, COCO-pretrained", "Done"),
    ("Training", "smoke test passed; EXP-001 on Kaggle GPU next", "In progress"),
    ("Validation", "runs during training", "Pending"),
    ("Testing", "evaluation/evaluate.py ready", "Pending"),
    ("Error analysis", "false positives / negatives per class", "Pending"),
    ("Model improvement", "image size, YOLO11n, tuning", "Pending"),
    ("Export", "ONNX", "Pending"),
    ("Quantization preparation", "FP16 / INT8 calibration", "Pending"),
    ("Benchmarking", "laptop CPU, then Raspberry Pi 5", "Pending"),
]
col = {"Done": OK, "In progress": WIP, "Pending": TODO}
s += [P("1. Pipeline progress", H2),
      table([["#", "Step", "Detail", "Status"]] +
            [[f"{i:02d}", a, b, tag(c, col[c])] for i, (a, b, c) in enumerate(steps, 1)],
            [1 * cm, 5 * cm, W - 9 * cm, 3 * cm])]

# ---------------- decisions ----------------
s += [P("2. Decisions", H2), P("Dataset: NEU-DET", H3)]
s += bullets([
    "The only candidate with real bounding-box labels, so it directly gives defect type + box + confidence.",
    "Six defect classes give the severity engine and the Llama advisor something meaningful to reason about.",
    "Small 200x200 images keep experiments fast.",
    "MVTec AD is built for anomaly detection (its training set contains no defects). Kept for a possible later anomaly branch.",
    "KolektorSDD has few defective samples and essentially one class.",
])
s += [KeepTogether([P("Model: YOLOv8n", H3)] + bullets([
    "Nano size: 3.0 M parameters and 8.1 GFLOPs at 640 px (measured, section 5).",
    "Mature export path (ONNX, NCNN, TFLite) and widely used in edge toolchains.",
    "Plain convolutional design that usually quantizes well to INT8.",
    "YOLO11n is planned as a comparison experiment, not the baseline.",
    "Compatibility with Hailo / Raspberry Pi toolchains is <b>not yet verified</b> by us.",
]))]
s += [P("<b>Known risk for the live demo:</b> NEU-DET shows hot-rolled steel strip, which cannot be placed under a "
        "live camera. A small custom dataset captured with the Pi camera is planned for a later phase, after the "
        "baseline is solid.", CALL)]

# ---------------- dataset ----------------
s += [PageBreak(), P("3. Dataset (MEASURED)", H2),
      table([["Item", "Value"],
             ["Images found", f"{prep['n_images_found']:,} (all {', '.join(prep['image_sizes'])})"],
             ["Images used", f"{prep['n_samples_used']:,}"],
             ["Defect boxes", f"{sum(boxes.values()):,}"],
             ["Corrupted images / invalid boxes", "0 / 0"],
             ["Exact duplicates removed", ", ".join(" = ".join(d) for d in dups) +
              " (same class, so not a label conflict)"]],
            [6 * cm, W - 6 * cm]),
      Spacer(1, 8),
      Table([[Image(str(f_boxes), width=W / 2 - 4, height=(W / 2 - 4) * 2.3 / 4.2),
              Image(str(f_area), width=W / 2 - 4, height=(W / 2 - 4) * 2.3 / 4.2)]],
            colWidths=[W / 2, W / 2]),
      P("Every class has 300 images, but box counts differ: inclusion has many small boxes per image, while pitted "
        "surface usually has one large box. Inclusion and scratches are the smallest targets; the smallest scratch "
        f"box covers {prep['box_relative_area_min']['scratches'] * 100:.2f}% of the image.", SMALL),
      P("Split (stratified by class, seed 42)", H3)]
hdr = ["Split"] + ["crazing", "inclusion", "patches", "pitted", "rolled-in", "scratches"] + ["Total"]
rows = [hdr] + [[sp.capitalize()] + [str(split[sp][c]) for c in CLASSES] + [f"{sum(split[sp].values()):,}"]
                for sp in ("train", "val", "test")]
s += [table(rows, [2 * cm] + [(W - 4 * cm) / 6] * 6 + [2 * cm])]

# ---------------- label check ----------------
img = ROOT / "experiments/SANITY/train_batch0.jpg"
s += [P("4. Label check (verified by eye)", H2),
      Table([[Image(str(img), width=8 * cm, height=8 * cm),
              [P("First training batch with labels drawn by Ultralytics. Boxes sit on the defects and class "
                 "numbers match class names (0 crazing, 1 inclusion, 2 patches, 3 pitted_surface, "
                 "4 rolled-in_scale, 5 scratches).", SMALL), Spacer(1, 4)] +
              bullets(["<b>Clear, tight boxes:</b> scratches, inclusion, patches. Expected to be the easiest classes.",
                       "<b>Loose, overlapping boxes:</b> crazing and rolled-in scale. The defect is a texture with no "
                       "clear edge, so box placement is subjective. Expected to be the hardest classes and the main "
                       "drag on mAP50-95.",
                       "Pitted-surface boxes often cover most of the image."])]],
            colWidths=[8.4 * cm, W - 8.4 * cm], style=[("VALIGN", (0, 0), (-1, -1), "TOP")])]

# ---------------- smoke test ----------------
s += [PageBreak(), P("5. Smoke test: 1 epoch on laptop CPU (MEASURED)", H2),
      P("A single-epoch run to confirm the pipeline works end to end and to time it. Its accuracy is meaningless "
        "by design and is <b>not</b> a baseline result.", BODY),
      table([["Item", "Value"],
             ["Hardware", "Intel Core i5-13420H, CPU only, 15.6 GB RAM"],
             ["Software", "Python 3.11.9, torch 2.14.0+cpu, ultralytics 8.4.163"],
             ["Image size / batch", "640 / 16 (79 steps per epoch)"],
             ["Epoch time (train + val)", f"{epoch_s:.1f} s (about {epoch_s / 60:.1f} min)"],
             ["100 epochs, projected", f"about {epoch_s * 100 / 3600:.1f} h on this laptop"],
             ["Val precision / recall", f"{float(sanity['metrics/precision(B)']):.3f} / "
                                        f"{float(sanity['metrics/recall(B)']):.3f}"],
             ["Val mAP50 / mAP50-95", f"{float(sanity['metrics/mAP50(B)']):.3f} / "
                                      f"{float(sanity['metrics/mAP50-95(B)']):.3f}"],
             ["Parameters (fused)", "3,006,818"],
             ["GFLOPs at 640", "8.1"],
             ["Weights file", "6.2 MB (FP32)"]],
            [6 * cm, W - 6 * cm]),
      P("<b>Conclusion:</b> data loads, the model trains and validates, and weights are saved. CPU training is too "
        "slow for many experiments, so full training moves to a free Kaggle GPU. Speed benchmarks stay on the laptop "
        "CPU and later the Raspberry Pi, where the model will actually run.", CALL)]

# ---------------- baseline table ----------------
nm = "not measured"
s += [P("6. Baseline benchmark: EXP-001 (not trained yet)", H2),
      P("Filled only with measured values from evaluation/evaluate.py on the 270-image test split. Parameters, "
        "FLOPs and size come from the smoke-test weights, which have the same architecture.", SMALL),
      table([["Metric", "EXP-001 (YOLOv8n, FP32, 640)", "EXPECTED range (not ours)"],
             ["Precision", nm, "-"], ["Recall", nm, "-"], ["F1", nm, "-"],
             ["mAP50", nm, "about 0.70 - 0.80"], ["mAP50-95", nm, "about 0.35 - 0.45"],
             ["Parameters", "3,006,818", "-"], ["GFLOPs at 640", "8.1", "-"], ["Model size", "6.2 MB", "-"],
             ["CPU latency (batch 1)", nm, "-"], ["CPU FPS", nm, "-"]],
            [5 * cm, 6 * cm, W - 11 * cm]),
      P("Expected ranges are typical of published YOLOv8n-class results on NEU-DET. Papers use different splits, so "
        "they are a sanity check, not a target. Crazing is usually the weakest class. A result far below the range "
        "points to a bug rather than a model limit.", SMALL)]

# ---------------- quantization ----------------
s += [KeepTogether([P("7. Quantization comparison (after baseline)", H2),
      P("Starts only after EXP-001 is reliable. INT8 matters most because the Hailo NPU runs INT8 models; "
        "toolchain support will be verified at that stage.", SMALL),
      table([["Version", "mAP50", "Size", "Latency", "FPS", "Memory"],
             ["FP32", "", "", "", "", ""], ["FP16", "", "", "", "", ""], ["INT8", "", "", "", "", ""]],
            [3 * cm] + [(W - 3 * cm) / 5] * 5)])]

# ---------------- plan ----------------
s += [P("8. Experiment plan", H2),
      table([["ID", "Model", "Img", "Where", "Purpose", "Status"],
             ["SANITY", "YOLOv8n", "640", "Laptop CPU", "Pipeline check and timing", tag("Done", OK)],
             ["EXP-001", "YOLOv8n", "640", "Kaggle GPU", "Reference baseline", tag("Next", WIP)],
             ["EXP-002", "YOLOv8n", "320", "Kaggle GPU", "Speed vs accuracy", tag("Planned", TODO)],
             ["EXP-003", "YOLO11n", "640", "Kaggle GPU", "Newer nano model comparison", tag("Planned", TODO)]],
            [2 * cm, 2.2 * cm, 1.3 * cm, 2.6 * cm, W - 10.6 * cm, 2.5 * cm]),
      Spacer(1, 6),
      P("<b>Training settings for all runs:</b> SGD, learning rate 0.01 decaying to 0.0001, momentum 0.937, "
        "weight decay 0.0005, 3 warm-up epochs, 100 epochs with early stopping after 25 epochs without improvement, "
        "seed 42, deterministic mode.", BODY),
      P("<b>Augmentation:</b> brightness +/-30%, horizontal and vertical flips, translate 10%, scale +/-50%, "
        "mosaic (switched off for the last 10 epochs). Hue, saturation, rotation and mixup are off because they "
        "would create physically unrealistic steel images.", BODY)]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=2 * cm, title="EdgeInspect-AI Vision Progress Report",
                        author="EdgeInspect-AI team", subject="Vision model training progress")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUT}")
