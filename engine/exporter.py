"""Export trained checkpoints to ONNX plus a working JavaScript integration."""

from __future__ import annotations

import base64
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import torch

from .train_cls import NormalizedClassifier, SoftmaxExport, build_classifier
from .train_seg import UNet
from .vlyolo import ExportWrapper, VLYOLO

LICENSE = "Apache-2.0"

COMMON_JS = r'''export function letterbox(source, size = INPUT_SIZE) {
  const width = source.naturalWidth || source.videoWidth || source.width;
  const height = source.naturalHeight || source.videoHeight || source.height;
  const ratio = Math.min(size / width, size / height);
  const newW = Math.round(width * ratio);
  const newH = Math.round(height * ratio);
  const padX = Math.floor((size - newW) / 2);
  const padY = Math.floor((size - newH) / 2);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, padX, padY, newW, newH);
  return { canvas, ratio, padX, padY, width, height };
}

export function toTensor(canvas) {
  const size = canvas.width;
  const { data } = canvas.getContext("2d").getImageData(0, 0, size, size);
  const area = size * size;
  const out = new Float32Array(area * 3);
  for (let i = 0; i < area; i++) {
    out[i] = data[i * 4] / 255;
    out[area + i] = data[i * 4 + 1] / 255;
    out[2 * area + i] = data[i * 4 + 2] / 255;
  }
  return out;
}

export async function loadModel(url) {
  return ort.InferenceSession.create(url, { executionProviders: ["webgpu", "wasm"] });
}
'''

DETECT_JS = r'''// VisionLLM detector — generated integration for a model you trained locally.
// npm install onnxruntime-web
import * as ort from "onnxruntime-web/webgpu";

export const LABELS = __LABELS__;
export const INPUT_SIZE = __SIZE__;
export const TASK = "detect";
export const META = __META__;

__COMMON__

export function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / Math.max(areaA + areaB - inter, 1e-9);
}

/** Per-class non-maximum suppression. */
export function nms(boxes, threshold = 0.45, maxDetections = 100) {
  const kept = [];
  const byClass = new Map();
  for (const box of boxes) {
    if (!byClass.has(box.cls)) byClass.set(box.cls, []);
    byClass.get(box.cls).push(box);
  }
  for (const group of byClass.values()) {
    group.sort((a, b) => b.score - a.score);
    const taken = [];
    for (const candidate of group) {
      if (taken.every((other) => iou(candidate, other) <= threshold)) taken.push(candidate);
    }
    kept.push(...taken);
  }
  return kept.sort((a, b) => b.score - a.score).slice(0, maxDetections);
}

/**
 * Run detection on an <img>, <video>, <canvas> or ImageBitmap.
 * Boxes are returned in original-image pixels, plus normalised coordinates.
 */
export async function detect(session, source, options = {}) {
  const { threshold = 0.25, iouThreshold = 0.45, maxDetections = 100 } = options;
  const started = performance.now();
  const geo = letterbox(source);
  const input = new ort.Tensor("float32", toTensor(geo.canvas), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const outputs = await session.run({ images: input });
  const output = outputs.detections || Object.values(outputs)[0];
  const [, rows, row] = output.dims;
  const classes = row - 4;
  const data = output.data;
  const raw = [];
  for (let i = 0; i < rows; i++) {
    const o = i * row;
    let best = 0;
    let bestClass = -1;
    for (let c = 0; c < classes; c++) {
      const score = data[o + 4 + c];
      if (score > best) {
        best = score;
        bestClass = c;
      }
    }
    if (best < threshold) continue;
    const cx = data[o] * INPUT_SIZE;
    const cy = data[o + 1] * INPUT_SIZE;
    const w = data[o + 2] * INPUT_SIZE;
    const h = data[o + 3] * INPUT_SIZE;
    raw.push({ cls: bestClass, score: best, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
  }
  const boxes = nms(raw, iouThreshold, maxDetections).map((b) => {
    const x1 = (b.x1 - geo.padX) / geo.ratio;
    const y1 = (b.y1 - geo.padY) / geo.ratio;
    const x2 = (b.x2 - geo.padX) / geo.ratio;
    const y2 = (b.y2 - geo.padY) / geo.ratio;
    return {
      cls: b.cls,
      label: LABELS[b.cls],
      score: b.score,
      x: x1,
      y: y1,
      w: x2 - x1,
      h: y2 - y1,
      xn: x1 / geo.width,
      yn: y1 / geo.height,
      wn: (x2 - x1) / geo.width,
      hn: (y2 - y1) / geo.height,
    };
  });
  return { boxes, width: geo.width, height: geo.height, ms: performance.now() - started };
}

/** Draw detections onto a canvas sized to the source image. */
export function draw(canvas, source, result, colors = __COLORS__) {
  canvas.width = result.width;
  canvas.height = result.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(source, 0, 0, result.width, result.height);
  ctx.lineWidth = Math.max(2, result.width / 260);
  const fontSize = Math.max(12, Math.round(result.width / 40));
  ctx.font = `${fontSize}px ui-monospace, SFMono-Regular, monospace`;
  for (const box of result.boxes) {
    const color = colors[box.cls % colors.length];
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
    const label = `${box.label} ${(box.score * 100).toFixed(0)}%`;
    const width = ctx.measureText(label).width;
    const top = box.y - fontSize - 4 < 0 ? box.y : box.y - fontSize - 4;
    ctx.fillRect(box.x, top, width + 8, fontSize + 4);
    ctx.fillStyle = "#0b0c0b";
    ctx.fillText(label, box.x + 4, top + fontSize);
  }
  return canvas;
}
'''

CLASSIFY_JS = r'''// VisionLLM classifier — generated integration for a model you trained locally.
import * as ort from "onnxruntime-web/webgpu";

export const LABELS = __LABELS__;
export const INPUT_SIZE = __SIZE__;
export const TASK = "classify";

__COMMON__

export async function classify(session, source, topK = 5) {
  const started = performance.now();
  const geo = letterbox(source);
  const input = new ort.Tensor("float32", toTensor(geo.canvas), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const outputs = await session.run({ images: input });
  const output = outputs.probabilities || Object.values(outputs)[0];
  const ranked = Array.from(output.data)
    .map((score, cls) => ({ cls, label: LABELS[cls], score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
  return { predictions: ranked, width: geo.width, height: geo.height, ms: performance.now() - started };
}
'''

SEGMENT_JS = r'''// VisionLLM segmentation model — generated integration.
import * as ort from "onnxruntime-web/webgpu";

export const LABELS = __LABELS__;
export const INPUT_SIZE = __SIZE__;
export const TASK = "segment";
export const PALETTE = __PALETTE__;

__COMMON__

/** Returns a transparent RGBA canvas of the predicted mask at the original size. */
export async function segment(session, source) {
  const started = performance.now();
  const geo = letterbox(source);
  const input = new ort.Tensor("float32", toTensor(geo.canvas), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const outputs = await session.run({ images: input });
  const output = outputs.masks || Object.values(outputs)[0];
  const [, channels, height, width] = output.dims;
  const data = output.data;

  const mask = document.createElement("canvas");
  mask.width = width;
  mask.height = height;
  const maskCtx = mask.getContext("2d");
  const rgba = maskCtx.createImageData(width, height);
  const counts = new Array(channels).fill(0);
  for (let p = 0; p < width * height; p++) {
    let best = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < channels; c++) {
      const value = data[c * width * height + p];
      if (value > bestScore) {
        bestScore = value;
        best = c;
      }
    }
    counts[best]++;
    const hex = PALETTE[best] || "#000000";
    rgba.data[p * 4] = parseInt(hex.slice(1, 3), 16);
    rgba.data[p * 4 + 1] = parseInt(hex.slice(3, 5), 16);
    rgba.data[p * 4 + 2] = parseInt(hex.slice(5, 7), 16);
    rgba.data[p * 4 + 3] = best === 0 ? 0 : 140;
  }
  maskCtx.putImageData(rgba, 0, 0);

  const canvas = document.createElement("canvas");
  canvas.width = geo.width;
  canvas.height = geo.height;
  const ctx = canvas.getContext("2d");
  const contentWidth = Math.round(geo.width * geo.ratio);
  const contentHeight = Math.round(geo.height * geo.ratio);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(mask, geo.padX, geo.padY, contentWidth, contentHeight, 0, 0, geo.width, geo.height);

  const present = counts
    .map((count, cls) => ({ cls, label: cls === 0 ? "background" : LABELS[cls - 1], fraction: count / (width * height) }))
    .filter((entry) => entry.cls > 0 && entry.fraction > 0.001)
    .sort((a, b) => b.fraction - a.fraction);

  return { canvas, present, width: geo.width, height: geo.height, ms: performance.now() - started };
}
'''


def _meta(result: Dict, size: int, task: str) -> Dict:
    return {
        "task": task,
        "input_size": size,
        "classes": result["classes"],
        "license": LICENSE,
        "architecture": result.get("model") or ("resnet18" if task == "classify" else "unet"),
    }


def _model_card(result: Dict, size: int, task: str, checkpoint: Dict) -> Dict:
    return {
        "name": f"VisionLLM {task} model",
        "task": task,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classes": result["classes"],
        "input": {"size": size, "layout": "NCHW", "color": "RGB", "range": "0..1", "padding": 114},
        "training": {
            "epochs_run": result.get("epochs_run"),
            "epochs_requested": result.get("epochs_requested"),
            "best_epoch": result.get("best_epoch"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "device": result.get("device"),
            "config": checkpoint.get("config", {}),
        },
        "evaluation": result.get("metrics") or {},
        "notes": result.get("notes", []),
        "license": LICENSE,
        "limitations": [
            "Only the classes in your dataset can be recognised.",
            "Accuracy depends on how varied and how accurately labelled the source images are.",
            "The model has no notion of objects it was never shown.",
        ],
    }


def _readme(result: Dict, size: int, task: str) -> str:
    classes = ", ".join(result["classes"])
    task_line = {
        "detect": "It draws bounding boxes.",
        "classify": "It returns a class probability for the whole image.",
        "segment": "It returns a class mask for every pixel.",
    }[task]
    example = {
        "detect": "  import { loadModel, detect, draw } from './inference.js'\n"
        "  const session = await loadModel('./model.onnx')\n"
        "  const result = await detect(session, document.getElementById('photo'))\n"
        "  draw(document.getElementById('out'), document.getElementById('photo'), result)",
        "classify": "  import { loadModel, classify } from './inference.js'\n"
        "  const session = await loadModel('./model.onnx')\n"
        "  console.log(await classify(session, document.getElementById('photo')))",
        "segment": "  import { loadModel, segment } from './inference.js'\n"
        "  const session = await loadModel('./model.onnx')\n"
        "  const { canvas } = await segment(session, document.getElementById('photo'))\n"
        "  document.body.append(canvas)",
    }[task]
    return f"""# VisionLLM {task} model

Trained locally on {time.strftime('%Y-%m-%d')} with VisionLLM. Classes: {classes}.
Input: {size}x{size}, RGB, 0..1, aspect-preserving letterbox (pad 114). {task_line}

## Files

- `model.onnx` — the trained model.
- `labels.json` — classes, preprocessing and output specification.
- `inference.js` — a working browser integration (needs `onnxruntime-web`).
- `model-card.json` — training and evaluation details.
- `dataset-yolo.zip` — your annotations in YOLO text format (detection only).

## Use it

```bash
npm install onnxruntime-web
```

```html
<img id="photo" src="photo.jpg" />
<canvas id="out"></canvas>
<script type="module">
{example}
</script>
```

Serve the folder over HTTP (for example `npx serve`) — browsers refuse to load
ONNX files from `file://`.

## Licence

The model architecture and this export are Apache-2.0. Your training images and
annotations remain yours. No data was uploaded anywhere.
"""


def export_run(run_dir: Path, export_dir: Path, dataset_zip: Path | None = None) -> Dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((run_dir / "result.json").read_text())
    task = result["task"]
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes: List[str] = checkpoint["classes"]
    size = int(checkpoint["imgsz"])
    state = checkpoint.get("ema") or checkpoint["state_dict"]

    if task == "detect":
        model = VLYOLO(len(classes), checkpoint["model"])
        model.load_state_dict({k: v for k, v in state.items()})
        wrapper = ExportWrapper(model.eval()).eval()
        output_name = "detections"
        output_spec = "[1, N, 4 + classes] — cx, cy, w, h normalised to input size, then per-class scores"
    elif task == "classify":
        backbone, _ = build_classifier(len(classes), pretrained=False)
        model = NormalizedClassifier(backbone)
        model.load_state_dict({k: v for k, v in state.items()})
        wrapper = SoftmaxExport(model.eval()).eval()
        output_name = "probabilities"
        output_spec = "[1, classes] — softmax probabilities"
    elif task == "segment":
        model = UNet(len(classes) + 1)
        model.load_state_dict({k: v for k, v in state.items()})
        wrapper = model.eval()
        output_name = "masks"
        output_spec = "[1, classes + 1, S, S] — per-pixel logits, channel 0 is background"
    else:
        raise ValueError(f"cannot export task {task!r}")

    dummy = torch.zeros(1, 3, size, size)
    onnx_path = export_dir / "model.onnx"
    common = {"input_names": ["images"], "output_names": [output_name], "opset_version": 17, "do_constant_folding": True}
    try:
        torch.onnx.export(wrapper, dummy, str(onnx_path), dynamo=False, **common)
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(onnx_path), **common)

    verification: Dict = {"ran": False}
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            reference = wrapper(dummy).numpy()
        loaded = session.run(None, {"images": dummy.numpy()})[0]
        verification = {
            "ran": True,
            "max_abs_diff": float(abs(reference - loaded).max()),
            "output_shape": list(loaded.shape),
            "onnxruntime": ort.__version__,
        }
    except Exception as exc:  # reported to the user rather than hidden
        verification = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}

    meta = _meta(result, size, task)
    meta["input"] = {
        "shape": f"1,3,{size},{size}",
        "layout": "NCHW",
        "color": "RGB",
        "range": "0..1",
        "preprocess": "aspect-preserving letterbox into a square canvas, pad value 114",
    }
    meta["output"] = output_spec
    meta["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["provenance"] = "Trained locally with VisionLLM. Source images never left this machine."
    (export_dir / "labels.json").write_text(json.dumps(meta, indent=2))

    colors = ["#ff4a1c", "#4ea8de", "#f2c14e", "#8ac926", "#c77dff", "#ff8fa3", "#4ecdc4", "#d4a373"]
    if task == "detect":
        js = (
            DETECT_JS.replace("__LABELS__", json.dumps(classes))
            .replace("__SIZE__", str(size))
            .replace("__META__", json.dumps(meta))
            .replace("__COMMON__", COMMON_JS.strip())
            .replace("__COLORS__", json.dumps(colors))
        )
    elif task == "classify":
        js = (
            CLASSIFY_JS.replace("__LABELS__", json.dumps(classes))
            .replace("__SIZE__", str(size))
            .replace("__COMMON__", COMMON_JS.strip())
        )
    else:
        palette = ["#000000"] + colors[: len(classes)]
        js = (
            SEGMENT_JS.replace("__LABELS__", json.dumps(classes))
            .replace("__SIZE__", str(size))
            .replace("__COMMON__", COMMON_JS.strip())
            .replace("__PALETTE__", json.dumps(palette))
        )
    (export_dir / "inference.js").write_text(js)

    card = _model_card(result, size, task, checkpoint)
    card["export"] = {
        "onnx_opset": 17,
        "bytes": onnx_path.stat().st_size,
        "verification": verification,
    }
    (export_dir / "model-card.json").write_text(json.dumps(card, indent=2))
    (export_dir / "README.md").write_text(_readme(result, size, task))

    files = ["model.onnx", "labels.json", "inference.js", "model-card.json", "README.md"]
    if dataset_zip and dataset_zip.exists():
        target = export_dir / "dataset-yolo.zip"
        target.write_bytes(dataset_zip.read_bytes())
        files.append("dataset-yolo.zip")

    zip_path = export_dir / "visionllm-model.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(export_dir / name, name)
    files.append("visionllm-model.zip")

    return {
        "task": task,
        "classes": classes,
        "input_size": size,
        "files": files,
        "onnx_bytes": onnx_path.stat().st_size,
        "verification": verification,
        "zip": "visionllm-model.zip",
    }


def read_export_file(export_dir: Path, name: str) -> Path:
    safe = Path(name).name
    path = export_dir / safe
    if not path.is_file():
        raise FileNotFoundError(name)
    return path


def file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
