# EdgeInspect-AI

A laptop prototype for industrial surface defect inspection and quality-control advice, built for Bharat AI-SoC Challenge 2026–27, PS5 (Manufacturing & Industry 4.0). The current vision model detects six NEU-DET steel-surface defect classes. The dashboard runs locally and records inspections in SQLite.

**Current status (10 Oct 2026):** EXP-001, EXP-002 and EXP-003 each completed 100 training epochs and held-out test evaluation. EXP-002 (YOLOv8n, 320 px) remains the laptop dashboard model because it measured 0.7702 test mAP50, 0.4361 mAP50-95 and 36.65 ms median CPU total per image (27.3 FPS). FP32 ONNX was evaluated but scored lower, so the dashboard stays on PyTorch best.pt. The laptop Gradio dashboard now has image/webcam and video inspection, QR/barcode reading, severity recommendations, numbered inspection records and SQLite history. The public dashboard runs from this laptop through Tailscale Funnel. Raspberry Pi, Hailo and ExecuTorch results are not yet verified.

**GitHub source:** [katyayanitiwari/EdgeInspect-AI](https://github.com/katyayanitiwari/EdgeInspect-AI)  
**Public project status:** [Open the status website](https://katyayanitiwari.github.io/edgeinspect-ai-site/) · Mentor PDF and personal notes PDF are in C:\Users\Katyayani\OneDrive\Desktop\EdgeInspect-AI-code.  
**Live dashboard:** [Open EdgeInspect-AI](https://laptop-46vn7dls.tail278ad4.ts.net) · Runs on this laptop; it must be awake, online, and signed in for the public link to work.

## Measured model comparison

| Experiment | Model | Input | Test mAP50 | Test mAP50-95 | Laptop CPU total | FPS |
|---|---|---:|---:|---:|---:|---:|
| EXP-001 | YOLOv8n | 640 | 0.7724 | 0.4307 | 92.28 ms | 10.8 |
| EXP-002 | YOLOv8n | 320 | 0.7702 | 0.4361 | 36.65 ms | 27.3 |
| EXP-003 | YOLO11n | 640 | 0.7715 | 0.4327 | 93.10 ms | 10.7 |
| EXP-002 ONNX FP32 | YOLOv8n | 320 | 0.7559 | 0.4208 | 31.5 ms validation inference* | — |

`*` ONNX validation inference timing is reported separately from the `evaluation.py` benchmark and should not be treated as a directly comparable application speed claim. See `deployment/EXP-002_ONNX_NOTES.md` and the JSON records in `evaluation/EXP-002/` for methods and limits. All accuracy figures are from the 270-image held-out test split; results are not Raspberry Pi measurements.

## Project layout

- `dataset/` — raw and processed NEU-DET data (not committed; regenerate using the preparation script)
- `training/` — reproducible training code and dataset configuration
- `experiments/` — experiment logs and training outputs
- `evaluation/` — held-out test metrics and export comparison records
- `inference/` — detection, severity rules, recommendations and inspection logging
- `app/dashboard.py` — local Gradio dashboard
- `deployment/` — deployment notes and export details
- `reports/` — status report builder and generated mentor status PDF; personal reference notes stay local

## Live dashboard

The public dashboard URL is [https://laptop-46vn7dls.tail278ad4.ts.net](https://laptop-46vn7dls.tail278ad4.ts.net). It is served from this laptop using Tailscale Funnel. The dashboard is set to start when you sign in to Windows. Keep the laptop awake and connected to the internet; sleep or shutdown makes the link temporarily unavailable. Tailscale Funnel is configured to resume in the background after a restart. Anyone with the public link can open the dashboard.

For direct local use, open http://127.0.0.1:7860. If the dashboard is not running, start it in PowerShell:

<pre>cd C:\Users\Katyayani\EdgeInspect-AI
.\.venv\Scripts\python.exe app\dashboard.py</pre>

Choose a sample or upload an image/video, then select <b>Inspect</b>. Image inference and QR/barcode decoding run on the laptop. Video inspection samples frames at about one frame per second. Inspection history is stored locally in logs/inspections.db. Use Ctrl+C in the PowerShell window only if you started the app manually.

The separate project status page is <a href="https://katyayanitiwari.github.io/edgeinspect-ai-site/">https://katyayanitiwari.github.io/edgeinspect-ai-site/</a>; it is hosted on GitHub Pages and does not depend on this laptop.

## Reproduce training and evaluation

Python 3.11 and the project virtual environment are expected. The original dataset mirror and exact preparation steps are documented in `scripts/prepare_neu_det.py` and `PLAN.md`.

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate.py --name EXP-002
.\.venv\Scripts\python.exe evaluation\evaluate.py --name EXP-001
.\.venv\Scripts\python.exe evaluation\evaluate.py --name EXP-003
```

The trained weights are stored locally and are excluded from Git. Kaggle was used for model training; final target-device inference is still pending.

## What remains

1. Review errors, especially the weaker crazing and rolled-in-scale classes; validate on the intended real parts.
2. Collect and label images from the target inspection setup, then decide whether fine-tuning is needed.
3. Verify Raspberry Pi 5 availability, install the system, and measure the PyTorch model on the Pi CPU.
4. Verify the Llama-family advisor and ExecuTorch path on the target device; the current advisor is rule-based.
5. Treat Hailo AI HAT+ as optional until hardware and supported conversion/runtime versions are verified.
6. Complete end-to-end offline validation, measured benchmarks, report and demonstration video.
7. Registration is complete and the source repo is on GitHub. Finish the technical report and demo video, then verify the submission checklist before Phase 1.

See `PLAN.md`, `experiments/EXPERIMENTS.md`, and `deployment/EXP-002_ONNX_NOTES.md` for the detailed status and evidence.

