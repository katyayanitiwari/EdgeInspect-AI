"""EdgeInspect-AI prototype dashboard (laptop, fully offline after install).

Image upload or webcam snapshot -> YOLO detection -> severity engine -> QC recommendation
-> SQLite inspection log -> history + statistics.

Run (project root):
  .\.venv\Scripts\python.exe app\dashboard.py
then open http://127.0.0.1:7860 in the browser.
  .\.venv\Scripts\python.exe app\dashboard.py --share   (also prints a temporary public link)

Model: newest of experiments/EXP-001/weights/best.pt, then last.pt, then the SANITY weights.
While EXP-001 is still training, the dashboard reloads the weights when the file changes and shows
how many epochs the model has been trained for.
"""
import csv
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import cv2
import gradio as gr
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.advisor import advise  # noqa: E402
from inference.detect import draw, run_one  # noqa: E402
from inference import inspection_log  # noqa: E402
from inference.severity import load_rules  # noqa: E402

CANDIDATES = [ROOT / "experiments" / "EXP-002" / "weights" / "best.pt",
              ROOT / "experiments" / "EXP-001" / "weights" / "last.pt",
              ROOT / "experiments" / "EXP-001-cpu" / "weights" / "best.pt",
              ROOT / "experiments" / "SANITY" / "weights" / "best.pt"]
IMG_DIR = ROOT / "logs" / "images"
TEST_DIR = ROOT / "dataset" / "processed" / "neu_det" / "images" / "test"
SAMPLE_DIR = ROOT / "app" / "samples"
SAMPLE_CARDS = [
    (str(TEST_DIR / filename), f"Steel defect — {label} (held-out NEU-DET)")
    for filename, label in (
        ("crazing_11.jpg", "crazing"), ("inclusion_187.jpg", "inclusion"),
        ("patches_109.jpg", "patches"), ("pitted_surface_10.jpg", "pitted surface"),
        ("rolled-in_scale_102.jpg", "rolled-in scale"), ("scratches_262.jpg", "scratches"),
    ) if (TEST_DIR / filename).exists()
] + [
    (str(SAMPLE_DIR / "sample_qr_code.png"), "QR demo — inspect, then click the decoded URL"),
    (str(SAMPLE_DIR / "sample_ean13_barcode.png"), "EAN-13 barcode demo"),
    (str(SAMPLE_DIR / "sample_text_poster.png"), "Text/poster — outside steel inspection scope"),
    (str(SAMPLE_DIR / "sample_people.jpg"), "People — outside steel inspection scope"),
]
SAMPLE_CARDS = [(path, label) for path, label in SAMPLE_CARDS if Path(path).exists()]
VIDEO_SAMPLE_CARDS = [
    (str(SAMPLE_DIR / "sample_steel_defects_montage.webm"),
     "Demo montage — six held-out steel-defect images; not recorded footage"),
    (str(SAMPLE_DIR / "sample_qr_and_barcode.webm"),
     "Code demo — scans a QR URL and an EAN-13 barcode"),
]
VIDEO_SAMPLE_CARDS = [(path, label) for path, label in VIDEO_SAMPLE_CARDS if Path(path).exists()]

RULES = load_rules()
PERSON_WEIGHTS = ROOT / "yolov8n.pt"
_person_model = None
_qr_detector = cv2.QRCodeDetector()
_barcode_detector = cv2.barcode.BarcodeDetector() if hasattr(cv2, "barcode") else None
_state = {"model": None, "path": None, "mtime": None}


def model_status(path):
    """Human-readable description of which weights are loaded and how far training got."""
    run_dir = path.parents[1]
    epochs = 0
    rc = run_dir / "results.csv"
    if rc.exists():
        with open(rc, newline="") as f:
            epochs = max(0, sum(1 for _ in csv.reader(f)) - 1)
    if run_dir.name == "SANITY":
        note = "pipeline check only, NOT a real model"
    elif run_dir.name.endswith("-cpu"):
        note = "partial laptop run, stopped early; full model is training on Kaggle"
    else:
        note = "full training run"
    return f"**Model:** `{run_dir.name}/{path.name}` — trained {epochs} epoch(s) ({note})"


def get_model():
    path = next((p for p in CANDIDATES if p.exists()), None)
    if path is None:
        raise gr.Error("No weights found. Train a model first.")
    mtime = path.stat().st_mtime
    if _state["path"] != path or _state["mtime"] != mtime:
        try:
            _state.update(model=YOLO(str(path)), path=path, mtime=mtime)
        except Exception as e:  # file may be mid-write while training saves it
            if _state["model"] is None:
                raise gr.Error(f"Could not load {path.name}: {e}")
    return _state["model"], _state["path"]


def get_person_model():
    global _person_model
    if _person_model is None:
        if not PERSON_WEIGHTS.exists():
            raise gr.Error("Scene-filter weights missing: yolov8n.pt")
        _person_model = YOLO(str(PERSON_WEIGHTS))
    return _person_model


def find_handheld_item(result, frame_shape):
    """Return a confident, non-person object overlapping a person box, if present."""
    h, w = frame_shape[:2]
    boxes = list(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf))
    people = [box.tolist() for box, cls, _ in boxes if int(cls) == 0]
    if not people:
        return None

    background_classes = {"couch", "bed", "chair", "dining table", "tv"}
    candidates = []
    frame_area = float(w * h)
    for box, cls, conf in boxes:
        class_id = int(cls)
        label = result.names[class_id]
        if class_id == 0 or label in background_classes or float(conf) < 0.35:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        item_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        fraction = item_area / frame_area
        if item_area <= 0 or not 0.03 <= fraction <= 0.60:
            continue
        overlap = 0.0
        for person in people:
            px1, py1, px2, py2 = person
            iw = max(0.0, min(x2, px2) - max(x1, px1))
            ih = max(0.0, min(y2, py2) - max(y1, py1))
            overlap = max(overlap, iw * ih / item_area)
        if overlap >= 0.15:
            candidates.append((float(conf), [x1, y1, x2, y2], label))
    return max(candidates, default=None, key=lambda item: item[0])


def scan_codes(frame):
    """Decode visible QR codes and supported 1D barcodes without using the defect model."""
    found = []
    seen = set()

    def add(payload, kind, points=None):
        payload = str(payload).strip()
        if not payload or payload in seen:
            return
        seen.add(payload)
        polygon = []
        if points is not None:
            polygon = [[int(round(float(x))), int(round(float(y)))]
                       for x, y in points.reshape(-1, 2)]
        found.append({"kind": str(kind), "payload": payload, "points": polygon})

    try:
        ok, decoded, points, _ = _qr_detector.detectAndDecodeMulti(frame)
        if ok:
            for i, payload in enumerate(decoded):
                poly = points[i] if points is not None and i < len(points) else None
                add(payload, "QR", poly)
        if not found:
            payload, points, _ = _qr_detector.detectAndDecode(frame)
            add(payload, "QR", points)
    except cv2.error:
        pass

    if _barcode_detector is not None:
        try:
            ok, decoded, kinds, points = _barcode_detector.detectAndDecodeWithType(frame)
            if ok:
                for i, payload in enumerate(decoded):
                    poly = points[i] if points is not None and i < len(points) else None
                    kind = kinds[i] if i < len(kinds) else "Barcode"
                    add(payload, kind, poly)
        except cv2.error:
            pass
    return found


def draw_codes(frame, codes):
    out = frame.copy()
    for index, code in enumerate(codes, 1):
        if code["points"]:
            pts = __import__("numpy").array(code["points"], dtype="int32").reshape((-1, 1, 2))
            cv2.polylines(out, [pts], True, (40, 220, 40), 3)
            x, y = code["points"][0]
            cv2.putText(out, f"{code['kind']} #{index}", (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 180, 40), 2)
    return out


def format_codes(codes):
    if not codes:
        return "No QR code or supported barcode decoded."
    lines = []
    for i, code in enumerate(codes, 1):
        payload = code["payload"]
        parsed = urlparse(payload)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            safe_url = quote(payload, safe=":/?&=#%+-._~")
            lines.append(f"{i}. **{code['kind']}:** [{payload}]({safe_url})")
        else:
            lines.append(f"{i}. **{code['kind']}:** {payload}")
    return "\n\n".join(lines)


def load_sample(evt: gr.SelectData):
    """Load the clicked sample image into the inspection input."""
    index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
    path = SAMPLE_CARDS[int(index)][0]
    image = cv2.imread(path)
    if image is None:
        raise gr.Error(f"Could not open sample image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def out_of_scope_reason(frame):
    """Conservative guard: the validated NEU-DET inputs are grayscale steel close-ups."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    saturated = cv2.threshold(saturation, 60, 255, cv2.THRESH_BINARY)[1]
    color_fraction = cv2.countNonZero(saturated) / float(saturation.size)
    if color_fraction > 0.25:
        return (f"{color_fraction:.0%} of pixels are strongly colored; this looks unlike "
                "the grayscale NEU-DET steel close-ups used to validate the model.")
    return None


def detection_crops(frame, defects):
    """Create one padded, individually labeled crop for every detection."""
    h, w = frame.shape[:2]
    gallery = []
    for defect in defects:
        x1, y1, x2, y2 = [int(round(v)) for v in defect["xyxy"]]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pad_x, pad_y = max(12, int(bw * 0.5)), max(12, int(bh * 0.5))
        left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
        right, bottom = min(w, x2 + pad_x), min(h, y2 + pad_y)
        crop = frame[top:bottom, left:right].copy()
        if crop.size == 0:
            continue
        color = (255, 80, 40) if defect.get("id") else (0, 200, 255)
        cv2.rectangle(crop, (x1 - left, y1 - top), (x2 - left, y2 - top), color, 2)
        caption = (f"{defect.get('id', 'D?')} | {defect['cls']} | "
                   f"confidence {defect['conf']:.2f} | {defect['severity']}")
        cv2.putText(crop, defect.get("id", "D?"), (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        gallery.append((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption))
    return gallery


def inspect(image_rgb, source_label):
    if image_rgb is None:
        raise gr.Error("Upload an image or take a webcam snapshot first.")
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    codes = scan_codes(frame)
    out_of_scope = out_of_scope_reason(frame)
    if out_of_scope:
        guarded = draw_codes(frame, codes)
        cv2.rectangle(guarded, (0, 0), (guarded.shape[1], 72), (35, 35, 35), -1)
        cv2.putText(guarded, "NOT INSPECTED - OUTSIDE VALIDATED INPUT",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
        cv2.putText(guarded, "Use a close-up of the steel surface",
                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        hist, stats = history()
        return (cv2.cvtColor(guarded, cv2.COLOR_BGR2RGB),
                "## Not inspected — image outside the model's validated input\\n"
                f"{out_of_scope} No defect prediction was run. Crop tightly around a "
                "steel surface and try again. This model is trained for NEU-DET steel "
                "defects; it is not validated for posters, packaging, or general objects.",
                [], "No defects were reported because this image was not inspected. "
                "Use a close-up steel-surface image or a NEU-DET test example.",
                hist, stats, [], format_codes(codes))
    scene_result = get_person_model().predict(
        frame, imgsz=320, conf=0.25, verbose=False
    )[0]
    has_person = any(int(cls) == 0 for cls in scene_result.boxes.cls)
    item = find_handheld_item(scene_result, frame.shape) if has_person else None

    if has_person and item is None:
        rejected = draw_codes(frame, codes)
        cv2.putText(rejected, "INPUT SKIPPED - NO SEPARATE ITEM FOUND", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        hist, stats = history()
        return (cv2.cvtColor(rejected, cv2.COLOR_BGR2RGB),
                "## Input skipped: no separate item detected\nNo steel-defect prediction was run.",
                [], "Keep the product visible and in focus. This prototype can inspect a separate "
                "item crop when it is held, but it cannot inspect a hand by itself.", hist, stats, [],
                format_codes(codes))

    model, path = get_model()
    handheld = item is not None
    item_roi = None
    if handheld:
        _, (x1f, y1f, x2f, y2f), _ = item
        x1, y1, x2, y2 = (max(0, int(x1f)), max(0, int(y1f)),
                          min(frame.shape[1], int(x2f)), min(frame.shape[0], int(y2f)))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise gr.Error("Could not crop the handheld item. Retake the image.")
        report, _ = run_one(model, crop, RULES, imgsz=640)
        for defect in report["defects"]:
            bx1, by1, bx2, by2 = defect["xyxy"]
            defect["xyxy"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
        report["decision"] = "MANUAL_REVIEW"
        report["reasons"].append(
            "handheld item crop; confirm candidate marks manually because bottle-surface "
            "performance has not been validated"
        )
        report["latency_ms"]["person_filter"] = round(scene_result.speed["inference"], 1)
        report["latency_ms"]["end_to_end"] += report["latency_ms"]["person_filter"]
        item_roi = (x1, y1, x2, y2)
        advice = {
            "text": "Possible surface marks are shown for manual checking. EXP-002 was trained "
                    "on NEU-DET steel-surface images; bottle-surface accuracy is not validated. "
                    "Confirm each marked area before making a quality decision.",
            "engine": "manual-review guidance",
        }
    else:
        report, _ = run_one(model, frame, RULES, imgsz=640)
        advice = advise(report)

    for index, defect in enumerate(report["defects"], 1):
        defect["id"] = f"D{index:02d}"
    gallery = detection_crops(frame, report["defects"])
    annotated = draw(frame, report)
    annotated = draw_codes(annotated, codes)
    if item_roi:
        x1, y1, x2, y2 = item_roi
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(annotated, "ITEM ROI", (x1 + 4, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
    if handheld:
        refs = ", ".join(d["id"] for d in report["defects"]) or "none"
        advice["text"] += "\nCandidate references: " + refs + "."

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    img_path = IMG_DIR / time.strftime("%Y%m%d_%H%M%S.jpg")
    con = inspection_log.connect()
    inspection_id = inspection_log.log(
        con, report, advice, source=source_label or "upload",
        weights=str(path.relative_to(ROOT)), image_path=str(img_path.relative_to(ROOT))
    )
    cv2.putText(annotated, f"Inspection #{inspection_id}", (8, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(img_path), annotated)

    table = [[d["id"], d["cls"], f"{d['conf']:.2f}", d["area_pct"], d["severity"],
              "yes" if d["uncertain"] else ""] for d in report["defects"]]
    if handheld:
        headline = (f"## Inspection #{inspection_id} — MANUAL_REVIEW, handheld item crop\nPossible candidate mark(s): "
                    f"{report['defect_count']}; {report['latency_ms']['end_to_end']:.0f} ms on laptop CPU\n\n"
                    f"{model_status(path)}")
    else:
        headline = (f"## Inspection #{inspection_id} — {report['decision']}\nSeverity **{report['severity']}**, "
                    f"{report['defect_count']} defect(s), "
                    f"{report['latency_ms']['end_to_end']:.0f} ms on laptop CPU\n\n{model_status(path)}")
    rec = f"{advice['text']}\n\n[advisor: {advice['engine']}]"
    hist, stats = history(con)
    con.close()
    return (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), headline, table, rec,
            hist, stats, gallery, format_codes(codes))

def inspect_video(video_path, source_label):
    """Inspect one frame per second and save a silent annotated preview clip."""
    if not video_path:
        raise gr.Error("Upload a video first.")
    if isinstance(video_path, (tuple, list)):
        video_path = video_path[0]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise gr.Error("Could not open this video. Try MP4, MOV, or AVI.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 1.0
    frame_step = max(1, int(round(fps)))
    max_samples = 120  # process at most the first two minutes at one frame per second
    model, weights = get_model()
    video_dir = ROOT / "logs" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_dir / f"inspection_{time.strftime('%Y%m%d_%H%M%S')}.webm"
    writer = None
    frame_rows, all_detections, code_items = [], [], {}
    frame_index = 0
    sample_index = 0
    total_inference_ms = 0.0

    try:
        while sample_index < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                seconds = frame_index / fps
                timestamp = f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
                codes = scan_codes(frame)
                for code in codes:
                    code_items[(code["kind"], code["payload"])] = code

                reason = out_of_scope_reason(frame)
                if reason:
                    annotated = draw_codes(frame, [])
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 68), (35, 35, 35), -1)
                    cv2.putText(annotated, "FRAME NOT INSPECTED - OUTSIDE VALIDATED INPUT",
                                (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 180, 255), 2)
                    cv2.putText(annotated, "Use a close-up of the steel surface",
                                (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    frame_rows.append([timestamp, "—", "Not inspected", "—", "—", reason])
                else:
                    report, _ = run_one(model, frame, RULES, imgsz=320)
                    total_inference_ms += report["latency_ms"]["end_to_end"]
                    for defect_index, defect in enumerate(report["defects"], 1):
                        defect_id = f"F{sample_index + 1:03d}-D{defect_index:02d}"
                        defect["id"] = defect_id
                        defect["frame_time"] = timestamp
                        all_detections.append(defect)
                        frame_rows.append([timestamp, defect_id, defect["cls"],
                                           f"{defect['conf']:.2f}", defect["area_pct"],
                                           defect["severity"]])
                    if not report["defects"]:
                        frame_rows.append([timestamp, "—", "No candidate", "—", "—",
                                           "No detection above threshold"])
                    annotated = draw(frame, report)
                    annotated = draw_codes(annotated, codes)

                cv2.putText(annotated, timestamp, (8, annotated.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(output_path),
                                             cv2.VideoWriter_fourcc(*"VP90"), 1.0, (w, h))
                    if not writer.isOpened():
                        raise gr.Error("Could not create the annotated preview video.")
                writer.write(annotated)
                sample_index += 1
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if sample_index == 0 or writer is None:
        raise gr.Error("No readable video frames were found.")
    limited = frame_index % frame_step == 0 and sample_index >= max_samples
    decision = "MANUAL_REVIEW"
    severity_order = {"none": 0, "minor": 1, "major": 2, "critical": 3}
    severity = max((d["severity"] for d in all_detections),
                   key=lambda value: severity_order.get(value, 0), default="none")
    advice_text = (
        "Video was sampled at approximately one frame per second. Detection rows are "
        "frame observations, not unique physical defects. Review the annotated clip before "
        "making a quality decision. Audio is not included in the processed preview."
    )
    if limited:
        advice_text += " Processing stopped at the 120-frame limit (about two minutes)."
    report = {
        "decision": decision, "severity": severity,
        "defect_count": len(all_detections), "defects": all_detections,
        "latency_ms": {"end_to_end": round(total_inference_ms, 1)},
    }
    advice = {"text": advice_text, "engine": "video sampling + rule-based review"}
    con = inspection_log.connect()
    inspection_id = inspection_log.log(
        con, report, advice, source=source_label or Path(video_path).name,
        weights=str(weights.relative_to(ROOT)), image_path=str(output_path.relative_to(ROOT))
    )
    hist, stats = history(con)
    con.close()
    codes_text = format_codes(list(code_items.values()))
    headline = (
        f"## Video inspection #{inspection_id} — MANUAL_REVIEW\n"
        f"Sampled **{sample_index} frame(s)** at approximately 1 frame/second; "
        f"**{len(all_detections)} candidate frame hit(s)**. "
        f"Candidate hits are not unique defect counts.\n\n{model_status(weights)}"
    )
    return str(output_path), headline, frame_rows, codes_text, hist, stats


def history(con=None):
    own = con is None
    con = con or inspection_log.connect()
    rows = inspection_log.recent(con)
    counts = inspection_log.decision_counts(con)
    if own:
        con.close()
    total = sum(counts.values())
    stats = f"**Total inspections: {total}**  \n" + "  \n".join(
        f"{k}: {v} ({v / total:.0%})" for k, v in sorted(counts.items())) if total else "No inspections yet."
    return rows, stats


def build():
    with gr.Blocks(title="EdgeInspect-AI") as demo:
        gr.Markdown("# EdgeInspect-AI")
        with gr.Tab("Inspect"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(sources=["upload", "webcam"], type="numpy", label="Part image")
                    source = gr.Textbox(label="Part / batch ID (optional)", placeholder="e.g. coil-17")
                    btn = gr.Button("Inspect", variant="primary")
                    samples_gallery = gr.Gallery(
                        value=SAMPLE_CARDS, label="Sample inputs — click a tile to load it",
                        columns=5, height="auto", object_fit="contain", allow_preview=False
                    )
                    samples_gallery.select(load_sample, None, img_in)
                with gr.Column():
                    img_out = gr.Image(label="Detections", type="numpy")
                    headline = gr.Markdown()
            defects = gr.Dataframe(headers=["defect ID", "candidate label", "confidence", "area % of inspected region", "severity", "low conf"],
                                   label="Defects", interactive=False)
            crops = gr.Gallery(label="Each candidate — zoomed and labeled by defect ID",
                               columns=3, height="auto", object_fit="contain")
            gr.Markdown("### Scanned QR / barcode information")
            codes_out = gr.Markdown("Inspect an image to read a QR code or supported barcode.")
            rec = gr.Textbox(label="QC recommendation", lines=8)
        with gr.Tab("Video"):
            gr.Markdown(
                "Upload a video to inspect about one frame per second. "
                "The preview is silent; frame detections are observations, not unique defect counts. "
                "The first two minutes are processed."
            )
            with gr.Row():
                video_in = gr.Video(sources=["upload", "webcam"], label="Video input — upload or record")
                video_out = gr.Video(label="Annotated preview (silent)")
            gr.Markdown("### Sample videos")
            with gr.Row():
                with gr.Column():
                    gr.Video(value=VIDEO_SAMPLE_CARDS[0][0] if VIDEO_SAMPLE_CARDS else None,
                             label="Steel-defect montage — held-out images, not recorded footage",
                             interactive=False)
                    load_defect_video = gr.Button("Use defect sample")
                with gr.Column():
                    gr.Video(value=VIDEO_SAMPLE_CARDS[1][0] if len(VIDEO_SAMPLE_CARDS) > 1 else None,
                             label="QR + EAN-13 sample",
                             interactive=False)
                    load_codes_video = gr.Button("Use QR/barcode sample")
            load_defect_video.click(lambda: VIDEO_SAMPLE_CARDS[0][0], None, video_in)
            load_codes_video.click(lambda: VIDEO_SAMPLE_CARDS[1][0], None, video_in)
            video_source = gr.Textbox(label="Part / batch ID (optional)")
            video_btn = gr.Button("Inspect video", variant="primary")
            video_headline = gr.Markdown()
            video_table = gr.Dataframe(
                headers=["time", "frame detection ID", "candidate / status",
                         "confidence", "area %", "severity / note"],
                label="Frame-by-frame observations", interactive=False
            )
            video_codes = gr.Markdown(label="QR / barcode information")
        with gr.Tab("History & stats"):
            refresh = gr.Button("Refresh")
            stats = gr.Markdown()
            hist = gr.Dataframe(headers=["id", "time", "part ID", "decision", "severity", "defects", "latency ms"],
                                interactive=False)
        btn.click(inspect, [img_in, source],
                  [img_out, headline, defects, rec, hist, stats, crops, codes_out])
        video_btn.click(
            inspect_video, [video_in, video_source],
            [video_out, video_headline, video_table, video_codes, hist, stats]
        )
        refresh.click(lambda: history(), None, [hist, stats])
        demo.load(lambda: history(), None, [hist, stats])
    return demo


def warm_up():
    """Load weights and run one dummy inference so the first real inspection is not slow."""
    import numpy as np
    model, path = get_model()
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)
    get_person_model().predict(np.zeros((320, 320, 3), dtype=np.uint8),
                               imgsz=320, conf=0.25, classes=[0], verbose=False)
    print(f"Warm-up done with {path.relative_to(ROOT)} and person filter")


if __name__ == "__main__":
    # --share: temporary public *.gradio.live link that tunnels to this laptop (laptop must stay on)
    share = "--share" in sys.argv
    warm_up()
    build().launch(server_name="127.0.0.1", server_port=7860, inbrowser=not share, share=share)
