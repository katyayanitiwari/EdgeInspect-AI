# EdgeInspect-AI

A laptop prototype for industrial surface defect inspection and quality-control advice, built for Bharat AI-SoC Challenge 2026–27, PS5 (Manufacturing & Industry 4.0). The current vision model detects six NEU-DET steel-surface defect classes. The dashboard runs locally and records inspections in SQLite.

**Current status (9 Oct 2026):** EXP-001, EXP-002 and EXP-003 training and held-out test evaluation are complete. EXP-002 (YOLOv8n, 320 px) is the dashboard model because it offers a useful accuracy/CPU-speed trade-off for the laptop prototype. An FP32 ONNX export was evaluated on the same test split; it scored slightly lower than its PyTorch source, so the dashboard continues to use `best.pt`. No Raspberry Pi, Hailo or ExecuTorch deployment has been verified.

**Public project status:** [Open the status website](https://katyayanitiwari.github.io/edgeinspect-ai-site/) · Mentor PDF and personal notes PDF are in `C:\Users\Katyayani\OneDrive\Desktop\EdgeInspect-AI-code`.

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

## Run the dashboard on Windows

From PowerShell:

```powershell
cd C:\Users\Katyayani\EdgeInspect-AI
.\.venv\Scripts\python.exe app\dashboard.py
```

Open `http://127.0.0.1:7860` if the browser does not open automatically. Choose an image from the sample section or upload one, then select **Inspect**. Press **Ctrl+C** in PowerShell to stop the dashboard. The app uses EXP-002 `best.pt` when that file is present and saves inspection history locally in `logs/inspections.db`.

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
7. Keep the completed registration record and verify current official submission requirements and deadlines before final submission.

See `PLAN.md`, `experiments/EXPERIMENTS.md`, and `deployment/EXP-002_ONNX_NOTES.md` for the detailed status and evidence.

