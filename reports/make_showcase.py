"""Build the prototype showcase page (single self-contained HTML) from measured files only.

Reads: experiments/EXP-002/results.csv (validation metrics per epoch), inference/severity_rules.yaml,
dataset/processed/neu_det/prep_report.json, and runs the current model on a few test images.

Usage (project root):
  .\.venv\Scripts\python.exe reports\make_showcase.py          (HTML only)
  .\.venv\Scripts\python.exe reports\make_showcase.py --pdf    (HTML + A4 PDF, printed by Microsoft Edge)
  .\.venv\Scripts\python.exe reports\make_showcase.py --pdf --site   (also site/index.html for a public website)
Output: reports/showcase/EdgeInspect-Prototype.html, reports/EdgeInspect-AI_Project_Status.pdf
"""
import base64
import csv
import html
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.advisor import advise  # noqa: E402
from inference.detect import run_one  # noqa: E402
from inference.severity import load_rules  # noqa: E402

OUT = ROOT / "reports" / "showcase" / "EdgeInspect-Prototype.html"
EXP = "EXP-002"
EXP_DIR = ROOT / "experiments" / EXP
if not (EXP_DIR / "results.csv").exists() and (ROOT / "experiments" / f"{EXP}-cpu" / "results.csv").exists():
    EXP_DIR = ROOT / "experiments" / f"{EXP}-cpu"  # partial laptop run, until the Kaggle run is downloaded
TEST_DIR = ROOT / "dataset" / "processed" / "neu_det" / "images" / "test"
SAMPLES = ["scratches_262.jpg", "inclusion_187.jpg", "patches_109.jpg"]
TOTAL_EPOCHS = 100
IMGSZ = 320
E = html.escape

DONE = [
    ("Dataset", "NEU-DET chosen, inspected, cleaned (1 duplicate removed), converted to YOLO format, stratified 70/15/15 split",
     "scripts/prepare_neu_det.py, prep_report.json"),
    ("Training", "Reproducible YOLOv8n training script; 1-epoch pipeline check on laptop CPU (633.6 s/epoch)",
     "training/train.py, experiments/SANITY"),
    ("Training", "EXP-001 (YOLOv8n, 640 px), EXP-002 (YOLOv8n, 320 px) and EXP-003 (YOLO11n, 640 px) each completed 100 epochs", "experiments/EXP-001, EXP-002 and EXP-003"),
    ("Model selection", "EXP-002 selected for laptop dashboard: test mAP50 0.770, mAP50-95 0.436; CPU total 36.65 ms/image", "evaluation/EXP-002/metrics.json"),
    ("ONNX export", "EXP-002 FP32 ONNX evaluated on 270 test images; lower mAP than PyTorch, so dashboard remains on best.pt", "deployment/EXP-002_ONNX_NOTES.md, evaluation/EXP-002/onnx_metrics.json"),
    ("Severity", "Rule-based severity engine with thresholds from train-split box sizes; 11 tests pass",
     "inference/severity.py, tests/test_severity.py"),
    ("Inference", "Image / folder / webcam inference with annotated output and JSON reports", "inference/detect.py"),
    ("Advisor", "Rule-based QC recommendation text (placeholder with the same interface Llama will use)", "inference/advisor.py"),
    ("Logging", "Every inspection stored in a local SQLite database", "inference/inspection_log.py"),
    ("Dashboard", "Laptop Gradio app: image/webcam and video inspection, QR/barcode reading, numbered detections, severity advice and SQLite history",
     "app/dashboard.py, inference/inspection_log.py"),
    ("Public demo", "Stable Tailscale Funnel URL is live while the laptop is awake and online; Windows sign-in startup is configured", "https://laptop-46vn7dls.tail278ad4.ts.net"),
    ("Cloud training", "Kaggle GPU training set up and EXP-001 result downloaded",
     "scripts/make_kaggle_bundle.py, scripts/kaggle_run.py"),
    ("Project", "Git repository, 15-day plan, experiment log, official challenge facts, progress reports",
     "PLAN.md, EXPERIMENTS.md, CHALLENGE.md, reports/"),
]

# status: done / prog / todo / block
LEFT = [
    ("Vision model", "Review test errors and labels, especially lower-AP crazing and rolled-in scale", "prog", "Per-class metrics and confusion matrix are generated; inspect examples"),
    ("Vision model", "EXP-001 / EXP-002 / EXP-003 training and held-out evaluation complete; EXP-002 selected for laptop dashboard", "done", "Revisit after target-device benchmarks"),
    ("Vision model", "FP32 ONNX export and test evaluation complete; FP16/INT8 and target-device verification remain", "prog", "Keep PyTorch best.pt until accuracy and runtime are checked"),
    ("Custom data", "Collect real parts, fix lighting and camera position, capture 300 to 500 images", "todo", "Parts and a camera setup"),
    ("Custom data", "Label images, fine-tune (EXP-004), recalibrate severity thresholds", "todo", "Captured images"),
    ("Llama advisor", "Choose a Llama-family model that fits the target device memory", "todo", "Needs verification on the device"),
    ("Llama advisor", "Export the model with ExecuTorch and run it on-device (required by PS5)", "todo", "Model access on Hugging Face (Meta licence)"),
    ("Llama advisor", "Design the QC prompt and replace the rule-based advisor; measure tokens/s and memory", "todo", "ExecuTorch model running"),
    ("Edge device", "Get Raspberry Pi 5 and Camera Module 3 (AI HAT+ optional)", "block", "Team to confirm what hardware is owned"),
    ("Edge device", "Pi OS setup, install script, run the full pipeline on the Pi CPU", "todo", "Hardware"),
    ("Edge device", "Optional: Hailo NPU path (WSL2 Ubuntu, Hailo compiler, HEF model) on AI HAT+", "todo", "AI HAT+ and a Hailo developer account"),
    ("Application", "End-to-end live pipeline: camera, detection, severity, Llama, log", "todo", "Llama advisor and hardware"),
    ("Application", "Dashboard on the Pi with live camera view", "todo", "Hardware"),
    ("Benchmarks", "Latency, FPS, memory and power on the Pi (CPU, and NPU if used); measured values only", "todo", "Pipeline on the Pi"),
    ("Submission", "Team registration completed (confirmed by project owner)", "done", "Keep the registration confirmation for project records"),
    ("Submission", "Source code published on GitHub", "done", "https://github.com/katyayanitiwari/EdgeInspect-AI"),
    ("Submission", "Technical report and demonstration video", "todo", "Results from the steps above"),
    ("Submission", "Phase 1 submission", "todo", "Due 30 Nov 2026"),
]
ST_LABEL = {"done": "Done", "prog": "In progress", "todo": "Not started", "block": "Blocked"}


def read_results():
    rc = EXP_DIR / "results.csv"
    if not rc.exists():
        return []
    with open(rc, newline="") as f:
        rows = list(csv.DictReader(f))
    clean = lambda r: {k.strip(): v.strip() for k, v in r.items()}
    return [{"epoch": int(float(r["epoch"])), "time": float(r["time"]),
             "p": float(r["metrics/precision(B)"]), "r": float(r["metrics/recall(B)"]),
             "map50": float(r["metrics/mAP50(B)"]), "map": float(r["metrics/mAP50-95(B)"])}
            for r in map(clean, rows)]


def pick_weights():
    for p in (EXP_DIR / "weights" / "best.pt", EXP_DIR / "weights" / "last.pt",
              ROOT / "experiments" / "SANITY" / "weights" / "best.pt"):
        if p.exists():
            return p
    return None


def samples(weights):
    from ultralytics import YOLO
    model, rules, out = YOLO(str(weights)), load_rules(), []
    for name in SAMPLES:
        frame = cv2.imread(str(TEST_DIR / name))
        if frame is None:
            continue
        report, annotated = run_one(model, frame, rules, imgsz=IMGSZ)
        annotated = cv2.resize(annotated, (480, 480), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        out.append({"name": name, "true_class": name.rsplit("_", 1)[0], "report": report,
                    "advice": advise(report), "img": base64.b64encode(buf).decode()})
    return out


def chart_svg():
    """Static SVG frame; points are drawn by the inline script from DATA (so hover can reuse them)."""
    return """<svg id="chart" viewBox="0 0 720 300" role="img" aria-labelledby="chart-title chart-desc">
<title id="chart-title">Validation mAP per epoch</title>
<desc id="chart-desc">Line chart of mAP50 and mAP50-95 on the validation split for each finished training epoch.</desc>
<g id="grid"></g><g id="band"></g><g id="lines"></g><g id="dots"></g><g id="hover"></g>
<rect id="hit" x="56" y="16" width="644" height="244" fill="transparent"></rect>
</svg>"""


def build():
    results = read_results()
    weights = pick_weights()
    rules = load_rules()
    prep = json.loads((ROOT / "dataset" / "processed" / "neu_det" / "prep_report.json").read_text())
    smp = samples(weights) if weights else []
    last = results[-1] if results else None
    epochs_done = last["epoch"] if last else 0
    run_dir = weights.parents[1].name if weights else "none"
    model_label = (f"{run_dir}/{weights.name}, {epochs_done} of {TOTAL_EPOCHS} epochs" if run_dir.startswith(EXP)
                   else "SANITY (1-epoch pipeline check, not a real model)")
    mean_epoch_min = ((results[-1]["time"] - results[-2]["time"]) / 60 if len(results) > 1 else (results[-1]["time"] / 60 if results else None))
    test_metrics_path = ROOT / "evaluation" / EXP / "metrics.json"
    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8")) if test_metrics_path.exists() else None
    updated = time.strftime("%d %b %Y, %H:%M IST")

    # ---------- sample cards
    dec_class = {"PASS": "ok", "PASS_WITH_NOTE": "ok", "REWORK": "warn", "REJECT": "crit", "MANUAL_REVIEW": "info"}
    cards = []
    for s in smp:
        rep = s["report"]
        lc = ' <span class="lc">low conf</span>'
        rows = "".join(
            f"<tr><td>{E(d['cls'])}</td><td class='num'>{d['conf']:.2f}</td><td class='num'>{d['area_pct']}</td>"
            f"<td><span class='sev sev-{d['severity']}'>{d['severity']}</span>{lc if d['uncertain'] else ''}</td></tr>"
            for d in rep["defects"]) or "<tr><td colspan='4' class='muted'>No defect above the 0.25 confidence threshold</td></tr>"
        cards.append(f"""
<article class="sample">
  <img src="data:image/jpeg;base64,{s['img']}" alt="Model output on NEU-DET test image {E(s['name'])}" width="480" height="480">
  <div class="sample-body">
    <div class="sample-head"><span class="decision {dec_class[rep['decision']]}">{E(rep['decision'].replace('_', ' '))}</span>
      <span class="muted">true class: <b>{E(s['true_class'])}</b></span></div>
    <div class="tablewrap"><table class="mini"><thead><tr><th>Detected</th><th>Conf.</th><th>Area %</th><th>Severity</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
    <details><summary>QC recommendation</summary><pre>{E(s['advice']['text'])}</pre></details>
  </div>
</article>""")

    # ---------- severity table
    sev_rows = "".join(
        f"<tr><td>{E(n)}</td><td class='num'>{c['train_boxes']}</td><td class='num'>{c['major_area']:.1f}</td>"
        f"<td class='num'>{c['critical_area']:.1f}</td></tr>" for n, c in rules["classes"].items())

    # ---------- epoch table
    ep_rows = "".join(
        f"<tr><td class='num'>{r['epoch']}</td><td class='num'>{r['p']:.3f}</td><td class='num'>{r['r']:.3f}</td>"
        f"<td class='num'>{r['map50']:.3f}</td><td class='num'>{r['map']:.3f}</td><td class='num'>{r['time'] / 60:.1f}</td></tr>"
        for r in results) or "<tr><td colspan='6' class='muted'>No finished epoch yet</td></tr>"
    best50 = max((r["map50"] for r in results), default=None)

    counts = {k: sum(v.values()) for k, v in prep["split_counts"].items()}
    boxes = prep["boxes_per_class"]
    data_json = json.dumps([{"e": r["epoch"], "a": r["map50"], "b": r["map"]} for r in results])

    kpis = [
        ("Training", f"{epochs_done}<small>/{TOTAL_EPOCHS}</small>", f"epochs finished ({EXP}, Kaggle GPU)"),
        ("Best val mAP50", f"{best50:.3f}" if best50 is not None else "&ndash;", "best validation epoch; test metrics are reported separately"),
        ("Last epoch time", f"{mean_epoch_min:.1f}<small> min</small>" if mean_epoch_min else "&ndash;", "Kaggle Tesla T4; recorded in the run output"),
        ("Dataset", f"{prep['n_samples_used']:,}", f"images · train {counts['train']:,} / val {counts['val']} / test {counts['test']}"),
    ]
    if test_metrics:
        ac = test_metrics["accuracy_test_split"]
        lat = test_metrics["cpu_latency_ms_median"]
        kpis.extend([("Test mAP50", f"{ac['mAP50']:.3f}", f"held-out test split; {EXP}"),
                     ("Test mAP50-95", f"{ac['mAP50_95']:.3f}", f"held-out test split; {EXP}"),
                     ("Laptop CPU", f"{lat['total']:.1f}<small> ms</small>", f"median total per image; {test_metrics['cpu_fps']:.1f} FPS")])
    kpi_html = "".join(f"<div class='kpi'><div class='kpi-label'>{a}</div><div class='kpi-value'>{b}</div>"
                       f"<div class='kpi-note'>{c}</div></div>" for a, b, c in kpis)

    stages = [
        ("Camera / image input", "working", "Image upload and webcam snapshot on the laptop. Pi Camera Module 3 later."),
        ("Defect detection", "complete" if epochs_done >= TOTAL_EPOCHS else "training", f"YOLOv8n, 6 NEU-DET classes. {epochs_done} of {TOTAL_EPOCHS} epochs done."),
        ("Severity engine", "working", "Rules per class: box area vs train-split median and 90th percentile, plus confidence."),
        ("QC advisor", "placeholder", "Rule-based text now. Llama via ExecuTorch (required by PS5) replaces it."),
        ("Dashboard + log", "working", "Laptop Gradio app with image/video inspection, QR/barcode reading, inspection IDs and SQLite history. Public link requires the laptop to stay awake and online."),
        ("Edge device", "planned", "Raspberry Pi 5 (CPU; AI HAT+ optional). Not started, hardware not yet confirmed."),
    ]
    chip = {"working": "Working on laptop", "training": "Training now", "complete": "Training complete", "placeholder": "Placeholder", "planned": "Planned"}
    stage_html = "".join(f"<li class='stage st-{s}'><span class='chip'>{chip[s]}</span><h3>{E(t)}</h3><p>{E(d)}</p></li>"
                         for t, s, d in stages)

    done_html = "".join(f"<tr><td>{E(a)}</td><td>{E(t)}</td><td class='muted'><code>{E(ev)}</code></td></tr>" for a, t, ev in DONE)
    left_html = "".join(f"<tr><td>{E(a)}</td><td>{E(t)}</td><td><span class='st st-{st}'>{ST_LABEL[st]}</span></td>"
                        f"<td class='muted'>{E(n)}</td></tr>" for a, t, st, n in LEFT)
    page = TEMPLATE
    for k, v in {"%%UPDATED%%": updated, "%%MODEL%%": E(model_label), "%%KPIS%%": kpi_html, "%%STAGES%%": stage_html,
                 "%%SAMPLES%%": "".join(cards) or "<p class='muted'>No samples yet.</p>", "%%SEVROWS%%": sev_rows,
                 "%%EPROWS%%": ep_rows, "%%CHART%%": chart_svg(), "%%DATA%%": data_json,
                 "%%TOTAL%%": str(TOTAL_EPOCHS), "%%IMGSZ%%": str(IMGSZ), "%%BOXES%%": ", ".join(f"{k} {v:,}" for k, v in boxes.items()),
                 "%%CONFMIN%%": str(rules["confidence"]["min"]), "%%CONFREV%%": str(rules["confidence"]["review"]),
                 "%%ESC%%": str(rules["count_escalation"]), "%%DONE%%": done_html, "%%LEFT%%": left_html}.items():
        page = page.replace(k, v)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB), model: {model_label}")
    return page


TEMPLATE = r"""<title>EdgeInspect-AI Prototype</title>
<meta name="description" content="Live status of the EdgeInspect-AI defect inspection prototype">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --bg:#eef0f2; --panel:#fafbfc; --ink:#15191e; --ink-2:#48505b; --muted:#6a7380; --line:#d3d8de;
  --accent:#1c5d99; --accent-soft:#dde8f3;
  --ok:#1f7a3a; --ok-bg:#e1f1e5; --warn:#9a5b00; --warn-bg:#fbecd2; --crit:#b3261e; --crit-bg:#f9dedb;
  --info:#3b4fa3; --info-bg:#e2e6f7;
  --s1:#2a78d6; --s2:#eb6834; --band:rgba(42,120,214,.10);
  --display:"IBM Plex Sans Condensed","Arial Narrow",Arial,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --bg:#12151a; --panel:#1a1e24; --ink:#eef1f4; --ink-2:#bcc3cc; --muted:#8f98a4; --line:#2e343d;
  --accent:#7fb2e8; --accent-soft:#1d2d40;
  --ok:#6fcf8a; --ok-bg:#16301f; --warn:#f0b454; --warn-bg:#3a2a10; --crit:#f08a82; --crit-bg:#3d1a18;
  --info:#9fb0f0; --info-bg:#1f2545;
  --s1:#3987e5; --s2:#d95926; --band:rgba(57,135,229,.14);
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#12151a; --panel:#1a1e24; --ink:#eef1f4; --ink-2:#bcc3cc; --muted:#8f98a4; --line:#2e343d;
  --accent:#7fb2e8; --accent-soft:#1d2d40;
  --ok:#6fcf8a; --ok-bg:#16301f; --warn:#f0b454; --warn-bg:#3a2a10; --crit:#f08a82; --crit-bg:#3d1a18;
  --info:#9fb0f0; --info-bg:#1f2545;
  --s1:#3987e5; --s2:#d95926; --band:rgba(57,135,229,.14);
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--body);padding:0 16px}
.wrap{max-width:1080px;margin:0 auto;padding-block:28px 56px;display:grid;gap:40px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:clamp(28px,5vw,42px);font-weight:700;letter-spacing:-.01em}
h2{font-size:22px;font-weight:600}
h3{font-size:16px;font-weight:600}
p{margin:0}
.muted{color:var(--muted)}
.eyebrow{font:500 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
header{display:grid;gap:10px;border-bottom:2px solid var(--ink);padding-bottom:18px}
.plate{display:flex;flex-wrap:wrap;gap:8px 18px;font:13px/1.4 var(--mono);color:var(--ink-2)}
.plate b{color:var(--ink);font-weight:500}
.lede{max-width:68ch;color:var(--ink-2)}
section{display:grid;gap:16px}
.section-head{display:grid;gap:6px}
.section-head p{max-width:70ch;color:var(--ink-2)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}
.kpi{background:var(--panel);padding:14px 16px;display:grid;gap:4px;align-content:start}
.kpi-label{font:500 11px/1.2 var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.kpi-value{font:600 30px/1.1 var(--display);font-variant-numeric:tabular-nums}
.kpi-value small{font-size:16px;color:var(--muted);font-weight:500}
.kpi-note{font-size:13px;color:var(--ink-2)}
.pipeline{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;counter-reset:st}
.stage{counter-increment:st;background:var(--panel);border:1px solid var(--line);border-top:4px solid var(--line);padding:12px 14px 14px;display:grid;gap:6px;align-content:start}
.stage h3::before{content:counter(st) " ";font:500 13px var(--mono);color:var(--muted)}
.stage p{font-size:13px;color:var(--ink-2)}
.chip{justify-self:start;font:500 11px/1 var(--mono);letter-spacing:.04em;padding:4px 7px;border-radius:3px}
.st-working{border-top-color:var(--ok)} .st-working .chip{background:var(--ok-bg);color:var(--ok)}
.st-training{border-top-color:var(--accent)} .st-training .chip{background:var(--accent-soft);color:var(--accent)}
.st-placeholder{border-top-color:var(--warn)} .st-placeholder .chip{background:var(--warn-bg);color:var(--warn)}
.st-planned{border-top-color:var(--muted);border-top-style:dashed} .st-planned .chip{background:transparent;color:var(--muted);border:1px dashed var(--muted)}
.samples{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.sample{background:var(--panel);border:1px solid var(--line);display:grid;grid-template-rows:auto 1fr}
.sample img{display:block;width:100%;height:auto;aspect-ratio:1;background:#000}
.sample-body{padding:12px 14px 14px;display:grid;gap:10px;align-content:start}
.sample-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;font-size:13px}
.decision{font:600 14px/1 var(--display);letter-spacing:.04em;padding:6px 9px;border-radius:3px}
.decision.ok{background:var(--ok-bg);color:var(--ok)} .decision.warn{background:var(--warn-bg);color:var(--warn)}
.decision.crit{background:var(--crit-bg);color:var(--crit)} .decision.info{background:var(--info-bg);color:var(--info)}
.sev{font:500 12px var(--mono)} .sev-minor{color:var(--ink-2)} .sev-major{color:var(--warn)} .sev-critical{color:var(--crit)}
.lc{font:500 11px var(--mono);color:var(--info)}
details summary{cursor:pointer;font-weight:500;color:var(--accent)}
details summary:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
pre{white-space:pre-wrap;font:12.5px/1.5 var(--mono);color:var(--ink-2);margin:8px 0 0}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{font:500 11px/1.3 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:6px 8px;border-bottom:1px solid var(--line)}
.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:13px}
th.num{text-align:right}
.mini td,.mini th{padding:4px 6px}
.panel{background:var(--panel);border:1px solid var(--line);padding:16px}
.chart-panel{display:grid;gap:10px}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:18px;height:3px;border-radius:2px;display:inline-block}
.swb{width:14px;height:10px;background:var(--band);border:1px solid var(--s1);display:inline-block}
.chartbox{position:relative}
#chart{width:100%;height:auto;display:block;overflow:visible}
#chart text{fill:var(--muted);font:11px var(--mono)}
#tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font:12px/1.45 var(--mono);padding:6px 8px;border-radius:3px;white-space:nowrap}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;align-items:start}
.rules{display:grid;gap:8px;font-size:14px;color:var(--ink-2);margin:0;padding-left:18px}
.limits{display:grid;gap:8px;margin:0;padding-left:18px;color:var(--ink-2);max-width:75ch}
.dates{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}
.date{background:var(--panel);padding:12px 14px;display:grid;gap:2px}
.date b{font:600 17px var(--display)}
.date.key b{color:var(--accent)}
.plan td{vertical-align:top}
.plan code{font:12px/1.4 var(--mono);color:var(--ink-2);word-break:break-word}
.plan td:first-child{font-weight:600;white-space:nowrap}
.st{font:500 11px/1 var(--mono);padding:4px 7px;border-radius:3px;white-space:nowrap;display:inline-block}
.st-done{background:var(--ok-bg);color:var(--ok)} .st-prog{background:var(--accent-soft);color:var(--accent)}
.st-todo{border:1px solid var(--line);color:var(--ink-2)} .st-block{background:var(--crit-bg);color:var(--crit)}
footer{border-top:1px solid var(--line);padding-top:14px;font-size:13px;color:var(--muted)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Bharat AI-SoC Challenge 2026–27 · PS5 Manufacturing &amp; Industry 4.0</div>
  <h1>EdgeInspect-AI</h1>
  <p class="lede">Real-time industrial defect detection and quality-control assistant for edge devices. A camera image goes through a defect detector, a severity engine and a QC advisor, and every inspection is logged, with no cloud.</p>
  <div class="plate"><span>Updated <b>%%UPDATED%%</b></span><span>Model on this page <b>%%MODEL%%</b></span><span>Stage <b>laptop prototype</b></span></div>
  <div class="plate"><span>Status website <b><a href="https://katyayanitiwari.github.io/edgeinspect-ai-site/">GitHub Pages</a></b></span><!--DEMO--></div>
  <div class="plate"><span>Live dashboard <b><a href="https://laptop-46vn7dls.tail278ad4.ts.net">Open the laptop demo</a></b></span><span>Available while the laptop is awake and online</span></div>
</header>

<section aria-label="Status">
  <div class="kpis">%%KPIS%%</div>
</section>

<section>
  <div class="section-head"><h2>Pipeline</h2>
  <p>Six stages, in the order an image passes through them. The chip on each stage says how far it is.</p></div>
  <ol class="pipeline">%%STAGES%%</ol>
</section>

<section>
  <div class="section-head"><h2>Sample inspections</h2>
  <p>Real output of the current model on held-out NEU-DET test images (200×200 px steel surface). This is a trained baseline, not a production inspection system. Review false and missed detections before relying on the decisions. Uncertain detections go to manual review.</p></div>
  <div class="samples">%%SAMPLES%%</div>
</section>

<section>
  <div class="section-head"><h2>Training progress</h2>
  <p>YOLOv8n (COCO-pretrained), %%IMGSZ%% px, SGD, up to %%TOTAL%% epochs with early stopping after 25 epochs without improvement. The chart shows validation metrics over 100 epochs (270 images per epoch). Final test-split metrics are shown separately above.</p></div>
  <div class="panel chart-panel">
    <div class="legend">
      <span><i class="sw" style="background:var(--s1)"></i>mAP50</span>
      <span><i class="sw" style="background:var(--s2)"></i>mAP50-95</span>
      <span><i class="swb"></i>Published YOLOv8n mAP50 on NEU-DET, 0.70–0.80 (literature, not our result)</span>
    </div>
    <div class="chartbox">%%CHART%%<div id="tip" hidden></div></div>
    <details><summary>Table view</summary>
      <div class="tablewrap"><table><thead><tr><th class="num">Epoch</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">mAP50</th><th class="num">mAP50-95</th><th class="num">Minutes</th></tr></thead>
      <tbody>%%EPROWS%%</tbody></table></div></details>
  </div>
</section>

<section>
  <div class="section-head"><h2>Severity engine</h2>
  <p>Each detected defect is graded by its size compared with defects of the same class in the training data. These thresholds are a data-based starting point, not an industrial standard; a real product would use its own QC specification.</p></div>
  <div class="two">
    <div class="panel tablewrap"><table><thead><tr><th>Class</th><th class="num">Train boxes</th><th class="num">Major ≥ area %</th><th class="num">Critical ≥ area %</th></tr></thead>
    <tbody>%%SEVROWS%%</tbody></table></div>
    <ol class="rules">
      <li>Detections below %%CONFMIN%% confidence are ignored.</li>
      <li>Box area above the class median is <b>major</b>, above the class 90th percentile is <b>critical</b>, otherwise <b>minor</b>.</li>
      <li>The part takes the worst severity. %%ESC%% or more defects raise it one level.</li>
      <li>A confident critical defect means <b>REJECT</b>. Any defect below %%CONFREV%% confidence sends the part to <b>MANUAL REVIEW</b>. Otherwise major means <b>REWORK</b> and minor means <b>PASS WITH NOTE</b>.</li>
    </ol>
  </div>
</section>

<section>
  <div class="section-head"><h2>Dataset</h2></div>
  <p class="lede">NEU-DET hot-rolled steel strip surface defects: 1,800 images found, 1 exact duplicate removed, 0 corrupted files, 0 invalid boxes. Stratified split with seed 42. Boxes per class: %%BOXES%%.</p>
</section>

<section>
  <div class="section-head"><h2>Work completed so far</h2>
  <p>Everything below exists in the project repository and has been run at least once.</p></div>
  <div class="panel tablewrap"><table class="plan"><thead><tr><th>Area</th><th>Done</th><th>Evidence</th></tr></thead>
  <tbody>%%DONE%%</tbody></table></div>
</section>

<section>
  <div class="section-head"><h2>What is left to be done</h2>
  <p>The remaining work for Phase 1 (submission due 30 Nov 2026 per the <a href="https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html">official challenge page</a>). Blocked items need an input from the team before work can start.</p></div>
  <div class="panel tablewrap"><table class="plan"><thead><tr><th>Area</th><th>Task</th><th>Status</th><th>Needs</th></tr></thead>
  <tbody>%%LEFT%%</tbody></table></div>
</section>

<section>
  <div class="section-head"><h2>Challenge dates</h2><p>From the official challenge page.</p></div>
  <div class="dates">
    <div class="date key"><span class="eyebrow">Registration closes</span><b>19 Oct 2026</b></div>
    <div class="date key"><span class="eyebrow">Phase 1 submission</span><b>30 Nov 2026</b></div>
    <div class="date"><span class="eyebrow">Phase 1 results</span><b>Mid-Dec 2026</b></div>
    <div class="date"><span class="eyebrow">Phase 2 PoC</span><b>10 Feb 2027</b></div>
    <div class="date"><span class="eyebrow">Finals</span><b>Early Mar 2027</b></div>
  </div>
</section>

<footer>All numbers on this page are measured by the project's own scripts, except the literature range marked as such.</footer>
</div>

<script>
(function(){
  const DATA = %%DATA%%, TOTAL = %%TOTAL%%;
  const svg = document.getElementById('chart'), NS = 'http://www.w3.org/2000/svg';
  const L = 56, R = 700, T = 16, B = 260;
  const x = e => L + (e / TOTAL) * (R - L), y = v => B - v * (B - T);
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); parent.appendChild(n); return n; };
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const grid = svg.querySelector('#grid');
  for (let v = 0; v <= 1.0001; v += 0.2) {
    el('line', {x1: L, x2: R, y1: y(v), y2: y(v), stroke: 'var(--line)', 'stroke-width': 1}, grid);
    el('text', {x: L - 8, y: y(v) + 4, 'text-anchor': 'end'}, grid).textContent = v.toFixed(1);
  }
  for (let e = 0; e <= TOTAL; e += 20) el('text', {x: x(e), y: B + 18, 'text-anchor': 'middle'}, grid).textContent = e;
  el('text', {x: (L + R) / 2, y: B + 36, 'text-anchor': 'middle'}, grid).textContent = 'epoch';
  el('rect', {x: L, width: R - L, y: y(0.80), height: y(0.70) - y(0.80), fill: 'var(--band)'}, svg.querySelector('#band'));
  const series = [['a', 'var(--s1)'], ['b', 'var(--s2)']];
  const lines = svg.querySelector('#lines'), dots = svg.querySelector('#dots');
  series.forEach(([k, c]) => {
    if (DATA.length > 1) el('polyline', {points: DATA.map(d => x(d.e) + ',' + y(d[k])).join(' '), fill: 'none', stroke: c, 'stroke-width': 2, 'stroke-linejoin': 'round'}, lines);
    const d = DATA[DATA.length - 1];
    if (d) el('circle', {cx: x(d.e), cy: y(d[k]), r: 4, fill: c, stroke: 'var(--panel)', 'stroke-width': 2}, dots);
  });
  if (!DATA.length) el('text', {x: (L + R) / 2, y: (T + B) / 2, 'text-anchor': 'middle'}, grid).textContent = 'No finished epoch yet';
  const tip = document.getElementById('tip'), hover = svg.querySelector('#hover'), hit = svg.querySelector('#hit');
  function show(evt){
    if (!DATA.length) return;
    const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    let best = DATA[0]; DATA.forEach(d => { if (Math.abs(x(d.e) - p.x) < Math.abs(x(best.e) - p.x)) best = d; });
    hover.innerHTML = '';
    el('line', {x1: x(best.e), x2: x(best.e), y1: T, y2: B, stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '3 3'}, hover);
    series.forEach(([k, c]) => el('circle', {cx: x(best.e), cy: y(best[k]), r: 5, fill: c, stroke: 'var(--panel)', 'stroke-width': 2}, hover));
    tip.innerHTML = 'epoch ' + best.e + '<br>mAP50 ' + best.a.toFixed(3) + '<br>mAP50-95 ' + best.b.toFixed(3);
    tip.hidden = false;
    const box = svg.getBoundingClientRect(), sx = box.width / 720;
    let left = x(best.e) * sx + 12; if (left + tip.offsetWidth > box.width) left = x(best.e) * sx - tip.offsetWidth - 12;
    tip.style.left = left + 'px'; tip.style.top = (y(best.a) * sx - 10) + 'px';
  }
  hit.addEventListener('mousemove', show);
  hit.addEventListener('mouseleave', () => { tip.hidden = true; hover.innerHTML = ''; });
})();
</script>
"""

PRINT_CSS = """<style>
@page{size:A4;margin:12mm}
body{background:#fff;padding:0;font-size:12.5px}
.wrap{padding-block:0;gap:26px}
.stage,.sample,.panel,.kpi,.date,tr{break-inside:avoid}
section{break-inside:avoid}
h2,.section-head{break-after:avoid}
.kpis{grid-template-columns:repeat(4,1fr)}
.dates{grid-template-columns:repeat(5,1fr)}
.tablewrap{overflow:visible}
.mini{font-size:11px}
.mini th{font-size:9px;letter-spacing:.03em}
.samples{grid-template-columns:repeat(3,1fr);gap:10px}
.sample-body{padding:10px}
.pipeline{grid-template-columns:repeat(3,1fr)}
.two{grid-template-columns:1fr 1fr}
details summary{list-style:none;color:var(--ink)}
details summary::-webkit-details-marker{display:none}
.chart-panel details{display:none}
.sample details{display:none}
.sample img{height:125px;object-fit:cover}
.sample-body{padding:6px}
.mini td,.mini th{padding:2px 3px}
</style>"""
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
PDF_OUT = ROOT / "reports" / "EdgeInspect-AI_Project_Status.pdf"


def demo_link():
    """Current public Gradio link, read from the running dashboard's log (None if not running with --share)."""
    import re
    log = ROOT / "logs_dashboard.txt"
    if not log.exists():
        return None
    m = re.search(r"Running on public URL:\s*(https://\S+)", log.read_text(encoding="utf-8", errors="ignore"))
    return m.group(1) if m else None


def build_pdf(page):
    """Print the page to A4 PDF with headless Edge: light theme, collapsed sections opened."""
    import os
    import subprocess
    import tempfile
    page = page.replace("<details>", "<details open>")
    demo = demo_link() if "--demo" in sys.argv else None  # temporary laptop link only on request
    if demo:  # live demo link goes in the PDF only, never on the public website
        page = page.replace("<!--DEMO-->", f"<span>Live demo (temporary, runs on the laptop) <b>{demo}</b></span>")
    doc = ('<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8">'
           + PRINT_CSS + "</head><body>" + page + PRINT_CSS + "</body></html>")
    tmp = OUT.with_name("print.html")
    tmp.write_text(doc, encoding="utf-8")
    profile = Path(tempfile.gettempdir()) / "edgeinspect_pdf_profile"
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                    "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={PDF_OUT}", tmp.resolve().as_uri()], check=True, timeout=120)
    # msedge.exe hands off to a child process and returns early: wait until the PDF exists and stops growing
    size, stable = -1, 0
    for _ in range(120):
        time.sleep(0.5)
        cur = PDF_OUT.stat().st_size if PDF_OUT.exists() else -1
        stable = stable + 1 if cur == size and cur > 0 else 0
        size = cur
        if stable >= 4:
            break
    else:
        raise SystemExit("Edge did not write the PDF within 60 s")
    os.remove(tmp)
    print(f"Wrote {PDF_OUT.relative_to(ROOT)} ({PDF_OUT.stat().st_size / 1e3:.0f} kB)")


SITE_OUT = ROOT / "site" / "index.html"


def build_site(page):
    """Standalone website version (any static host, e.g. GitHub Pages): adds the full HTML skeleton."""
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           "<style>html{color-scheme:light dark}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>"
           "</head><body>" + page + "</body></html>")
    SITE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SITE_OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {SITE_OUT.relative_to(ROOT)} ({SITE_OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    html_page = build()
    if "--pdf" in sys.argv:
        build_pdf(html_page)
    if "--site" in sys.argv:
        build_site(html_page)
