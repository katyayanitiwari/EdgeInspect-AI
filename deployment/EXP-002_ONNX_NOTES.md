# EXP-002 ONNX export and laptop checks

## What was exported

- Source model: `experiments/EXP-002/weights/best.pt` (YOLOv8n, 320 × 320, FP32)
- Exported model: `experiments/EXP-002/weights/best.onnx` (FP32, batch 1, fixed 320 × 320 input)
- Export command used Ultralytics with ONNX opset 12, CPU export, and graph simplification disabled.
- ONNX Runtime used the laptop CPU. No Raspberry Pi or Hailo measurements are available yet.

## Held-out test results

Both rows below use the same 270-image NEU-DET test split, 320 input, batch 1, and CPU.

| Model | Precision | P | R | mAP50 | mAP50-95 | Median inference reported by validation |
|---|---:|---:|---:|---:|---:|---:|
| EXP-002 PyTorch | FP32 | 0.7118 | 0.7093 | 0.7702 | 0.4361 | 48.2 ms |
| EXP-002 ONNX | FP32 | 0.6806 | 0.7338 | 0.7559 | 0.4208 | 31.5 ms |

ONNX measured 0.0143 lower mAP50 and 0.0152 lower mAP50-95 than PyTorch in these runs. The ONNX file is 12.12 MB; the PyTorch checkpoint is 6.22 MB. These are laptop measurements and do not establish Raspberry Pi or Hailo performance.

A separate prediction comparison on 30 held-out images at confidence 0.25 found 31 detections from each model, with all 31 same-class boxes matched at IoU ≥ 0.50, mean matched IoU 1.0, and mean confidence difference 0.0. This is a limited thresholded sample check; it does not override the full-test mAP difference.

## Current model choice

Keep `best.pt` as the dashboard model for now. The ONNX export is available for later runtime checks, but its measured test accuracy is lower and it has not been tested on the target hardware. Do not describe it as Pi/Hailo-ready yet.

## Reproduce the checks

Run these from `C:\Users\Katyayani\EdgeInspect-AI` in PowerShell:

```powershell
.\.venv\Scripts\python.exe evaluation\compare_onnx.py
```

The default compares the first 30 sorted held-out test images and writes `evaluation/EXP-002/onnx_parity.json`.

To rerun the full ONNX held-out test evaluation:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; from pathlib import Path; m=YOLO('experiments/EXP-002/weights/best.onnx'); r=m.val(data='training/neu_det.yaml', split='test', imgsz=320, batch=1, device='cpu', project='evaluation/EXP-002', name='onnx_test_rerun', exist_ok=True); print(r.results_dict)"
```

The already recorded results are in `evaluation/EXP-002/onnx_metrics.json`, `evaluation/EXP-002/pytorch_batch1_metrics.json`, and `evaluation/EXP-002/onnx_parity.json`.

## Remaining deployment work

1. Confirm Raspberry Pi 5 and AI HAT+ availability.
2. Measure the selected model on the actual target device and camera path.
3. Verify the Hailo conversion/runtime path for this YOLO model against official tooling and supported versions before making compatibility claims.
4. Build and measure the Llama-family QC advisor with ExecuTorch on the target device; record model, memory, latency, and offline behavior.
5. Run end-to-end inspections and document failures, false alarms, and limits on the intended part types.
