# Architecture

VisionLLM is a hybrid local-first application. The UI and source data live in the
browser; a small Python engine does real training; exported ONNX models run in the
browser. The engine binds only to loopback.

## What is actually trained

Object detection is the primary task. Classification and segmentation are real too.

| Task | Model | Training | Metrics |
| --- | --- | --- | --- |
| Object detection | **VLYOLO** (this repo) — CSP backbone, FPN neck, anchor-free decoupled heads, FCOS-style assignment, GIoU + focal + centerness losses | From scratch on your data, MPS or CPU | mAP@0.5, mAP@0.5:0.95, per-class AP, precision/recall |
| Image classification | ResNet-18 (torchvision) | ImageNet-pretrained fine-tune, or from scratch if weights are unavailable | accuracy, macro F1, per-class P/R/F1, confusion matrix |
| Segmentation | Compact U-Net (this repo) | From scratch on painted polygon masks | mIoU, per-class IoU, pixel accuracy |

The synthetic pipeline test in `engine/selftest.py` trains a small two-class detector
and checks the ONNX graph against PyTorch. Its reported mAP is a pipeline check, not a
claim about your dataset.

## Image handling guarantees

- Uploaded bytes are written exactly as received. The browser stores the original
  `Blob`; the engine stores the same bytes under `~/.visionllm/projects/<id>/images/`.
- Every model input is letterboxed: scale by `min(size/w, size/h)`, round the fit, pad
  with 114 and centre the padding. Nothing is cropped, squashed or centre-cropped.
- The same geometry is used in Python training, Python inference, the browser ONNX path
  and the generated `inference.js`. Predictions map back to original-image pixels.
- Annotation coordinates are normalised to the original and stored separately from it.
- The only derived image copies are UI thumbnails. They are never used for training.

## One library, automatic scoring holdout

The interface has one image library. Importing does not ask the user to assign images
to training or validation sets. When training starts, a small scoring holdout is chosen
automatically so evaluation remains honest. Do not add split controls to the UI.

## Licensing

| Stack | Licence | Use |
| --- | --- | --- |
| Ultralytics YOLOv5 / v8 / YOLO11 | AGPL-3.0 | Not used. |
| YOLOX | Apache-2.0 | Not used. |
| PyTorch, torchvision | BSD-3-Clause | Tensors, optimisers, NMS, ResNet-18 and COCO detectors. |
| torchvision FCOS / Faster R-CNN | BSD-3-Clause | Pretrained COCO weights for annotation suggestions only. |
| ONNX / ONNX Runtime Web | Apache-2.0 / MIT | Export and browser inference. |
| VLYOLO, U-Net, trainer, app | Apache-2.0 | Original code in this repository. |

Browser inference is not browser training. Real detector training needs autograd, an
optimiser and compute time, so the local Python engine handles training.

## Data locations

| What | Where |
| --- | --- |
| Projects, originals, annotations, runs | Browser IndexedDB (`visionllm`) |
| Dataset copy, checkpoints, metrics, exports | `~/.visionllm/projects/<project-id>/` |

Deleting a project removes the browser records and asks the engine to remove its
training copy. Source image files on disk are never touched.

## Honest limits

- Training is single-process and one run at a time. No distributed training or
  hyperparameter search.
- The detector is compact by design: useful for tens to low thousands of images, not
  COCO-scale accuracy.
- Annotation suggestions use the 80 COCO classes. Custom classes will not be proposed
  automatically.
- Segmentation uses polygons, not a brush. Fine structures need many points.
- Browser inference is WebGPU-first and falls back to slower WASM.
- The engine is a local process with your user permissions. Run it on projects you
  trust.
