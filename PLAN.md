# EdgeInspect-AI — 15-Day Plan (full scope)

Start: Day 1 = 27 Sep 2026. End: Day 15 = 11 Oct 2026.
Status keys: [ ] todo, [~] in progress, [x] done.

## Scope (nothing cut)
Vision baseline + improvements (YOLOv8n 640, YOLOv8n 320, YOLO11n), FP32/FP16/INT8 comparison,
custom camera dataset, Raspberry Pi 5 CPU deployment, **Hailo NPU deployment (AI HAT+)**,
severity engine, **Llama + ExecuTorch** on the Pi, full offline dashboard with inspection logging,
end-to-end benchmarks, report, slides, demo video.

## Conditions for this plan to hold
- Raspberry Pi 5, Camera Module 3 **and AI HAT+** in hand by Day 7 (order on Day 1 if not owned).
- 5–6 hours/day of work. No buffer day — any slip moves the end date.
- Hailo compilation needs an x86 Linux environment (WSL2 on this laptop or a Linux PC); set up on Day 3.
- Official deadline and judging criteria to be checked on the challenge website.

## Current status update — 10 Oct 2026

- Training and held-out test evaluation are complete for EXP-001 (YOLOv8n, 640), EXP-002 (YOLOv8n, 320) and EXP-003 (YOLO11n, 640), each trained for 100 epochs.
- EXP-002 is the current laptop dashboard model: test mAP50 0.7702, mAP50-95 0.4361, median laptop CPU total 36.65 ms/image and 27.3 FPS. These figures are laptop measurements on the 270-image held-out split.
- EXP-002 FP32 ONNX export and 270-image evaluation are complete. Its test mAP50 0.7559 and mAP50-95 0.4208 were below the PyTorch checkpoint, so the dashboard still uses PyTorch best.pt. No Pi or Hailo results are implied by this comparison.
- The dashboard accepts images/webcam and video, includes sample steel-defect and QR/barcode demos, reads QR/barcode data, gives rule-based severity/QC advice, assigns inspection numbers and records history in SQLite.
- Public laptop dashboard: https://laptop-46vn7dls.tail278ad4.ts.net. Tailscale Funnel is enabled and configured in the background; the dashboard is set to start at Windows sign-in. The laptop must be awake and connected to the internet for the public link to respond.
- GitHub source repository: https://github.com/katyayanitiwari/EdgeInspect-AI. Public project status page: https://katyayanitiwari.github.io/edgeinspect-ai-site/.
- Challenge registration: complete, as confirmed by the project owner. Phase 1 submission is listed as 30 Nov 2026 on the [official challenge page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html); check the live page again before submitting.
- Raspberry Pi, Camera Module 3 availability, Llama + ExecuTorch, and target-device performance remain unverified. AI HAT+/Hailo is optional in the current challenge description and remains unverified.
- The original 15-day schedule is historical. Use the remaining work list below, not the original day numbering, to plan from 10 Oct.

## Tracks
- **A — Model** (training on Kaggle GPU)
- **B — Hardware & data** (hands-on work: parts, camera, labelling, Pi)
- **C — Software** (written while training runs; validated only after the baseline is stable)

## Day-by-day

| Day | Track A — Model | Track B — Hardware & data | Track C — Code |
|---|---|---|---|
| 1 | [x] EXP-001 training and evaluation complete (YOLOv8n, 640) | [ ] Order Pi 5 (8 GB), Camera Module 3, AI HAT+, 27 W PSU, active cooler, microSD | [x] Severity engine (rules per defect class) — `inference/severity.py`, tests pass |
| 2 | [x] EXP-001/002/003 training and evaluation complete; qualitative error review remains | [ ] Collect real parts: defective + clean | [x] Laptop inference script (image/webcam → boxes → severity) — `inference/detect.py`, done Day 1 |
| 3 | [x] Evaluated EXP-002/003 and selected EXP-002 for the laptop dashboard | [ ] Set up lighting + fixed camera position | [ ] WSL2 + Ubuntu setup for Hailo toolchain; Llama + ExecuTorch export attempt |
| 4 | [~] FP32 ONNX export/evaluation complete; FP16/INT8 and target-device checks remain | [ ] Capture custom images (target 300–500) | [ ] Labelling workflow (model pre-labels, manual correction) |
| 5 | [ ] Hailo compile attempt (ONNX → HEF, INT8 calibration) | [ ] Label custom images | [ ] QC prompt design for Llama |
| 6 | [ ] Fine-tune on custom data (EXP-004) on Kaggle | [ ] Finish labelling | [x] Inspection logging (SQLite) — `inference/inspection_log.py`, done Day 1 |
| 7 | [ ] Evaluate EXP-004; recompile HEF for final model | [ ] Pi OS setup, camera + AI HAT+ test | [ ] Pi install script |
| 8 | [ ] Pi CPU: latency / FPS / memory (FP32, INT8) | [ ] Run tests on Pi | [ ] Fix Pi issues from real errors |
| 9 | [ ] Pi + Hailo NPU: latency / FPS / memory | [ ] Run Hailo tests | [ ] Hailo inference path in pipeline |
| 10 | — | [ ] Run Llama on Pi | [ ] Llama + ExecuTorch on Pi: tokens/s, memory |
| 11 | — | [ ] Test end to end on real parts | [ ] Full pipeline: camera → detect → severity → Llama → log |
| 12 | — | [ ] Test dashboard | [ ] Full dashboard (live view, defects, severity, recommendation, history, stats), offline |
| 13 | [ ] End-to-end benchmarks: CPU vs Hailo, latency, FPS, memory, power (if meter) | [ ] Record measurements | [ ] Benchmark scripts + tables (measured only) |
| 14 | — | [ ] Review report + slides | [ ] Final report + slides |
| 15 | — | [ ] Practice + record demo video | [ ] Final fixes |

## Risks
- Hardware late → every Pi/Hailo step slips day for day.
- Hailo compilation and Llama + ExecuTorch on the Pi are the least certain steps; both start early (Days 3–5) to find problems before Day 9–10.
- Custom dataset quality decides demo quality; do not rush lighting and camera setup.

## Laptop prototype (added Day 1)
`app/dashboard.py`: upload/webcam → YOLO → severity → rule-based advisor → SQLite log → history/stats. Lets us demo before hardware arrives. Llama + ExecuTorch replaces the rule-based advisor later.
