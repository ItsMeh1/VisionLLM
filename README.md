# VisionLLM

A local-first computer vision training platform. Import images, annotate them, train a real detector on your own
machine, evaluate it on held-out images, and export a verified ONNX model with a working JavaScript integration.

No account. No cloud upload. No mock training.

```bash
npm install
npm run engine     # terminal 1 — local Python trainer on 127.0.0.1:8765
npm run dev        # terminal 2 — the app on 127.0.0.1:5173
```

Open <http://127.0.0.1:5173>.

The first run of the engine installs nothing; it uses `engine/.venv`. Recreate it with:

```bash
cd engine
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

If there is no `uv`, `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt` works.

---

## What is actually trained

Object detection is the primary task. The other two project types are real as well.

| Task | Architecture | Training | Metrics |
| --- | --- | --- | --- |
| Object detection | **VLYOLO** (this repo) — CSP backbone, FPN neck, anchor-free decoupled heads, FCOS-style assignment, GIoU + focal + centerness losses | From scratch on your data, on MPS or CPU | mAP@0.5, mAP@0.5:0.95, per-class AP, precision/recall |
| Image classification | ResNet-18 (torchvision) | ImageNet-pretrained fine-tune, or from scratch if weights are unavailable | accuracy, macro F1, per-class P/R/F1, confusion matrix |
| Segmentation | Compact U-Net (this repo) | From scratch on painted polygon masks | mIoU, per-class IoU, pixel accuracy |

A 48-image synthetic test set with two object classes trains to **mAP@0.5 ≈ 0.97** in 18 epochs at 320 px on Apple
silicon. That number comes from `engine/selftest.py`, which also asserts that inference returns boxes and that the
ONNX graph reproduces PyTorch to within `1e-4`. It is a pipeline test, not a claim about your dataset.

```bash
engine/.venv/bin/python -m engine.selftest
```

## Why this architecture and not Ultralytics YOLO

| Stack | Licence | Verdict |
| --- | --- | --- |
| Ultralytics YOLOv5 / v8 / YOLO11 | AGPL-3.0, and Ultralytics states the licence covers models trained with its code | Not used. It would impose copyleft on every model a user trains. |
| YOLOX | Apache-2.0 | Permissive, but its training pipeline assumes Linux/CUDA tooling. |
| PyTorch, torchvision | BSD-3-Clause | Used for tensors, optimisers, NMS, ResNet-18 and COCO detectors. |
| torchvision FCOS / Faster R-CNN | BSD-3-Clause | Used **only** for annotation suggestions (pretrained COCO weights). |
| ONNX / ONNX Runtime Web | Apache-2.0 / MIT | Used for export and browser inference. |
| VLYOLO, U-Net, trainer, app | Apache-2.0 | Original code in this repository. |

Browser inference is not browser training. Real detector training needs autograd, an optimiser and minutes of compute;
ONNX Runtime Web has no training loop here. So VisionLLM is a **hybrid**: the UI and data live in the browser, a small
local Python engine does the training, and the exported artifact runs in the browser. The engine binds to loopback only.

## Image handling guarantees

- Uploaded bytes are written exactly as received. The browser stores the original `Blob`; the engine stores the same
  bytes under `~/.visionllm/projects/<id>/images/`.
- Every model input is produced by **letterboxing**: scale by `min(size/w, size/h)`, round the fit, pad with 114, centre
  the padding. Nothing is cropped, squashed or centre-cropped into a square.
- The same geometry is used in Python training, Python inference, the browser ONNX path and the generated
  `inference.js`. Predictions are mapped back to original-image pixels.
- Annotation coordinates are stored normalised to the original, independently of the model input size.
- The only derived image copies are UI thumbnails (JPEG, 420 px). They are never used for training.
- Input resolution is configurable per project (detection: 320–640, multiple of 32) and recorded in the checkpoint.

## Workflow

The app is one workspace with three columns, plus two secondary views.

1. **Examples** — drop images, a folder or camera frames into a single library. Import deduplicates by name+size,
   auto-creates a stratified 80/20 split once there are enough images, and opens the first new image for annotation.
   Open any tile to review it in detail.
2. **Annotate** — draw boxes on the image, move or resize them, paint polygons for segmentation, or choose a class for
   classification. The active class comes from the class chips; class cards and counts are derived automatically from
   the annotations. There is no per-class import path — imports always land in the shared library first.
3. **Train** — choose a model and training length; readiness, blockers and warnings come from the real dataset. A live
   engine job reports progress, losses and mAP epoch by epoch.
4. **Preview** — image, video, camera or the exported ONNX model in the browser (WebGPU with a WASM fallback).
5. **Evaluate / Export** — recompute metrics from the saved checkpoint on held-out images; export `model.onnx`,
   labels, `inference.js`, a model card and the dataset zip after ONNX Runtime verification.

## Working with coding agents

This repository is set up to be driven by external coding agents. [AGENTS.md](AGENTS.md) documents the product
contract, architecture, local setup, checks and the loopback engine API; [CLAUDE.md](CLAUDE.md) points Claude Code at
the same instructions.

There is deliberately no embedded chat assistant or agent dock. Claude Code, Codex or Freebuff work directly on the
repository, and the local engine exposes the same real dataset, training, evaluation, prediction, autolabel and export
endpoints the UI uses.

## Data locations

| What | Where |
| --- | --- |
| Projects, originals, annotations, runs | Browser IndexedDB (`visionllm`) |
| Dataset copy, checkpoints, metrics, exports | `~/.visionllm/projects/<project-id>/` |

Deleting a project in the app removes the browser data. The engine folder can be removed separately.

## Repository layout

```
src/                      React app (Vite, TypeScript, hand-written CSS)
  lib/                    IndexedDB, engine client, letterbox geometry, analysis, ONNX
  components/             annotation canvas, charts, UI primitives
  components/workspace/   Examples, Train, Preview, annotation dialog
  screens/                projects, new project, project layout, workspace, evaluate, export, settings
engine/
  server.py               FastAPI on loopback
  vlyolo.py               detector, losses, decoding, ONNX wrapper
  train_detect.py         real training loop + evaluation
  train_cls.py            ResNet-18 fine-tuning
  train_seg.py            U-Net
  pretrained.py           COCO detectors for annotation suggestions
  exporter.py             ONNX export, verification, JS template, zip
  jobs.py                 background job registry
  selftest.py             end-to-end synthetic training test
```

## Honest limitations

- Training is single-process, one run at a time. No distributed training, no hyperparameter search.
- Detection models are compact by design. They are good for tens to low thousands of images, not COCO-scale accuracy.
- Annotation suggestions depend on the 80 COCO classes. A class like "weld seam" will not be proposed automatically.
- Segmentation uses polygons, not a brush; very fine structures need many points.
- Browser ONNX inference is WebGPU-first. On browsers without WebGPU it falls back to WASM and is slower.
- The engine is not sandboxed. It reads only the dataset copies it is given, but it is a local process with your user
  permissions—run it on projects you trust.
