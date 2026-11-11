"""VisionLLM local training engine.

Runs on the user's machine, bound to loopback. Images are written to
~/.visionllm only for training; they are never uploaded anywhere.
"""

from __future__ import annotations

import io
import json
import math
import os
import platform
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from . import __version__
from . import exporter, pretrained, train_cls, train_detect, train_seg
from .data import load_manifest
from .jobs import JobRegistry
from .letterbox import letterbox_image, letterbox_size
from .train_detect import pick_device
from .vlyolo import SPECS, VLYOLO, decode_predictions

HOME = Path(os.environ.get("VISIONLLM_HOME", Path.home() / ".visionllm"))
PROJECTS = HOME / "projects"
PROJECTS.mkdir(parents=True, exist_ok=True)
REGISTRY = JobRegistry()
MODEL_CACHE: Dict[str, Dict] = {}
MODEL_LOCK = __import__("threading").Lock()
PORT = int(os.environ.get("VISIONLLM_PORT", "8765"))
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = FastAPI(title="VisionLLM Engine", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_COLORS = ["#ff4a1c", "#4ea8de", "#f2c14e", "#8ac926", "#c77dff", "#ff8fa3", "#4ecdc4", "#d4a373"]


def _check_id(value: str, label: str = "id") -> str:
    if not ID_RE.match(value or ""):
        raise HTTPException(400, f"invalid {label}")
    return value


def project_path(project_id: str) -> Path:
    return PROJECTS / _check_id(project_id, "project id")


def run_path(project_id: str, run_id: str) -> Path:
    return project_path(project_id) / "runs" / _check_id(run_id, "run id")


def _sanitise_boxes(boxes, class_count: int):
    clean = []
    for box in boxes or []:
        try:
            cls = int(box["cls"])
            x, y, w, h = (float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"]))
        except (KeyError, TypeError, ValueError):
            continue
        if cls < 0 or cls >= class_count or not all(map(math.isfinite, (x, y, w, h))):
            continue
        x, y = max(0.0, min(0.999, x)), max(0.0, min(0.999, y))
        w, h = max(0.0, min(1.0 - x, w)), max(0.0, min(1.0 - y, h))
        if w <= 0.001 or h <= 0.001:
            continue
        clean.append({"cls": cls, "x": round(x, 6), "y": round(y, 6), "w": round(w, 6), "h": round(h, 6)})
    return clean


def _sanitise_shapes(shapes, class_count: int):
    clean = []
    for shape in shapes or []:
        try:
            cls = int(shape["cls"])
            points = [[float(p[0]), float(p[1])] for p in shape["points"]]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if cls < 0 or cls >= class_count or len(points) < 3:
            continue
        points = [[max(0.0, min(1.0, x)), max(0.0, min(1.0, y))] for x, y in points]
        clean.append({"cls": cls, "points": [[round(x, 6), round(y, 6)] for x, y in points]})
    return clean


@app.get("/")
def root():
    return {"engine": "visionllm", "version": __version__, "docs": "/health"}


@app.get("/health")
def health():
    import torch
    import torchvision

    return {
        "ok": True,
        "engine": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(pick_device("auto")),
        "mps": bool(torch.backends.mps.is_available()),
        "cuda": bool(torch.cuda.is_available()),
        "storage": str(HOME),
        "models": {
            name: {
                "name": spec.name,
                "head": spec.head,
                "width": list(spec.width),
                "depth": list(spec.depth),
                "description": spec.description,
            }
            for name, spec in SPECS.items()
        },
        "pretrained": {
            name: {**meta, "loaded": pretrained.is_loaded(name)} for name, meta in pretrained.CATALOG.items()
        },
        "preprocessing": {
            "letterbox": True,
            "pad_value": 114,
            "notes": "Aspect ratio is always preserved; images are never centre-cropped.",
        },
        "licenses": {
            "visionllm": "Apache-2.0",
            "pytorch": "BSD-3-Clause",
            "torchvision": "BSD-3-Clause",
            "onnxruntime": "MIT",
            "ultralytics": "not used (AGPL-3.0)",
        },
    }


# --------------------------------------------------------------------------- #
# Dataset upload
# --------------------------------------------------------------------------- #


@app.post("/projects/{project_id}/dataset")
async def upload_dataset(
    project_id: str,
    manifest: str = Form(...),
    files: List[UploadFile] = File(...),
):
    root = project_path(project_id)
    try:
        payload = json.loads(manifest)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"manifest is not valid JSON: {exc}")
    classes = [str(c) for c in payload.get("classes") or []]
    task = payload.get("task") or "detect"
    if task not in ("detect", "classify", "segment"):
        raise HTTPException(400, "unknown task")
    if not classes:
        raise HTTPException(400, "define at least one class before uploading")

    images_dir = root / "images"
    labels_dir = root / "labels"
    for directory in (images_dir, labels_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    by_name = {f.filename: f for f in files if f.filename}
    stored_images = []
    skipped = []
    for item in payload.get("images") or []:
        handle = by_name.get(item.get("file"))
        if handle is None:
            skipped.append({"id": item.get("id"), "reason": "file missing from upload"})
            continue
        data = await handle.read()
        try:
            probe = Image.open(io.BytesIO(data))
            probe.load()
            width, height = probe.size
            fmt = probe.format or "PNG"
        except Exception as exc:
            skipped.append({"id": item.get("id"), "reason": f"not a readable image ({type(exc).__name__})"})
            continue
        ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tif"}.get(fmt, ".png")
        image_id = str(item.get("id") or uuid.uuid4().hex[:12])
        image_id = re.sub(r"[^A-Za-z0-9_-]", "", image_id)[:48] or uuid.uuid4().hex[:12]
        stored_name = f"{image_id}{ext}"
        (images_dir / stored_name).write_bytes(data)
        split = item.get("split") if item.get("split") in ("train", "val") else "train"
        record = {
            "id": image_id,
            "file": stored_name,
            "split": split,
            "width": width,
            "height": height,
            "original_name": item.get("originalName"),
        }
        if task == "detect":
            record["boxes"] = _sanitise_boxes(item.get("boxes"), len(classes))
        elif task == "segment":
            record["shapes"] = _sanitise_shapes(item.get("shapes"), len(classes))
        else:
            label = item.get("label")
            if label is None:
                skipped.append({"id": image_id, "reason": "no class label"})
                continue
            record["label"] = int(label)
        stored_images.append(record)

        if task == "detect":
            lines = []
            for box in record.get("boxes", []):
                cx = box["x"] + box["w"] / 2
                cy = box["y"] + box["h"] / 2
                lines.append(f"{box['cls']} {cx:.6f} {cy:.6f} {box['w']:.6f} {box['h']:.6f}")
            (labels_dir / f"{image_id}.txt").write_text("\n".join(lines))

    dataset = {
        "task": task,
        "classes": classes,
        "images": stored_images,
        "uploaded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "train": sum(1 for i in stored_images if i["split"] == "train"),
            "val": sum(1 for i in stored_images if i["split"] == "val"),
        },
    }
    (root / "dataset.json").write_text(json.dumps(dataset, indent=2))
    return {
        "stored": len(stored_images),
        "skipped": skipped,
        "counts": dataset["counts"],
        "storage": str(root),
    }


@app.get("/projects/{project_id}/dataset")
def dataset_summary(project_id: str):
    try:
        manifest = load_manifest(project_path(project_id))
    except FileNotFoundError:
        raise HTTPException(404, "no dataset has been uploaded for this project")
    return {
        "task": manifest["task"],
        "classes": manifest["classes"],
        "counts": manifest.get("counts"),
        "images": len(manifest.get("images", [])),
        "uploaded": manifest.get("uploaded"),
    }


def _dataset_zip(project_id: str) -> Path:
    root = project_path(project_id)
    manifest = load_manifest(root)
    target = root / "dataset-yolo.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", f"names: {json.dumps(manifest['classes'])}\nnc: {len(manifest['classes'])}\n")
        for item in manifest["images"]:
            image_file = root / "images" / item["file"]
            if image_file.exists():
                archive.write(image_file, f"images/{item['split']}/{item['file']}")
            label_file = root / "labels" / f"{item['id']}.txt"
            if label_file.exists():
                archive.write(label_file, f"labels/{item['split']}/{item['id']}.txt")
    return target


@app.get("/projects/{project_id}/dataset.zip")
def download_dataset(project_id: str):
    try:
        path = _dataset_zip(project_id)
    except FileNotFoundError:
        raise HTTPException(404, "no dataset")
    return FileResponse(path, filename="visionllm-dataset.zip", media_type="application/zip")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@app.post("/projects/{project_id}/train")
def start_training(project_id: str, config: dict = Body(...)):
    root = project_path(project_id)
    try:
        manifest = load_manifest(root)
    except FileNotFoundError:
        raise HTTPException(400, "upload a dataset before training")
    task = manifest["task"]
    if task == "detect":
        config.setdefault("model", "vlyolo-s")
        if config["model"] not in SPECS:
            raise HTTPException(400, f"unknown model {config['model']}")
    config.setdefault("epochs", 40 if task == "detect" else 20)
    config.setdefault("batch", 8 if task == "detect" else 16)
    config.setdefault("imgsz", 512 if task == "detect" else (384 if task == "segment" else 224))
    config["task"] = task
    run_id = config.get("run_id") or time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:4]
    run_id = _check_id(run_id, "run id")
    run_dir = run_path(project_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    job = REGISTRY.create("train", run_id, int(config.get("epochs", 0)))
    runner = {"detect": train_detect.train, "classify": train_cls.train, "segment": train_seg.train}[task]
    REGISTRY.start_training(job, runner, root, run_dir, config, job.emit, job.should_stop)
    return {"job_id": job.id, "run_id": run_id, "task": task, "config": config}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.snapshot()


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job.cancel()
    return {"ok": True, "status": job.status}


@app.get("/projects/{project_id}/runs")
def list_runs(project_id: str):
    base = project_path(project_id) / "runs"
    if not base.exists():
        return {"runs": []}
    runs = []
    for directory in sorted(base.iterdir(), reverse=True):
        result_file = directory / "result.json"
        summary = {"run_id": directory.name}
        if result_file.exists():
            try:
                result = json.loads(result_file.read_text())
                summary.update(
                    {
                        "task": result.get("task"),
                        "status": result.get("status"),
                        "classes": result.get("classes"),
                        "epochs_run": result.get("epochs_run"),
                        "metrics": {
                            k: result.get("metrics", {}).get(k)
                            for k in ("map50", "map5095", "accuracy", "miou", "pixel_accuracy")
                            if k in (result.get("metrics") or {})
                        },
                        "elapsed_seconds": result.get("elapsed_seconds"),
                    }
                )
            except json.JSONDecodeError:
                pass
        runs.append(summary)
    return {"runs": runs}


def _run_result(project_id: str, run_id: str) -> Dict:
    path = run_path(project_id, run_id) / "result.json"
    if not path.exists():
        raise HTTPException(404, "run has no result yet")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Model loading and inference
# --------------------------------------------------------------------------- #


def _load_run_model(project_id: str, run_id: str) -> Dict:
    key = f"{project_id}/{run_id}"
    with MODEL_LOCK:
        if key in MODEL_CACHE:
            return MODEL_CACHE[key]
    import torch

    directory = run_path(project_id, run_id)
    result = _run_result(project_id, run_id)
    checkpoint_path = directory / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = directory / "last.pt"
    if not checkpoint_path.exists():
        raise HTTPException(404, "run has no checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    task = result["task"]
    classes = checkpoint["classes"]
    size = int(checkpoint["imgsz"])
    state = checkpoint.get("ema") or checkpoint["state_dict"]
    if task == "detect":
        model = VLYOLO(len(classes), checkpoint["model"])
    elif task == "classify":
        backbone, _ = train_cls.build_classifier(len(classes), pretrained=False)
        model = train_cls.NormalizedClassifier(backbone)
    elif task == "segment":
        model = train_seg.UNet(len(classes) + 1)
    else:
        raise HTTPException(400, f"unsupported task {task}")
    model.load_state_dict({k: v for k, v in state.items()})
    model.eval()
    entry = {
        "task": task,
        "model": model,
        "size": size,
        "classes": classes,
        "result": result,
        "run_id": run_id,
    }
    with MODEL_LOCK:
        if len(MODEL_CACHE) >= 3:
            MODEL_CACHE.pop(next(iter(MODEL_CACHE)))
        MODEL_CACHE[key] = entry
    return entry


def _tensor_for(image: Image.Image, size: int):
    import torch

    canvas, ratio, pad_x, pad_y = letterbox_image(image, size)
    array = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None], ratio, pad_x, pad_y


def _predict_one(entry: Dict, data: bytes, conf: float, iou: float) -> Dict:
    import torch

    image = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = image.size
    size = entry["size"]
    started = time.time()
    tensor, ratio, pad_x, pad_y = _tensor_for(image, size)
    with torch.no_grad():
        output = entry["model"](tensor)
    elapsed = (time.time() - started) * 1000

    if entry["task"] == "detect":
        prediction = decode_predictions(output, size, conf_threshold=conf, iou_threshold=iou)[0]
        boxes = []
        for box, score, label in zip(prediction["boxes"].numpy(), prediction["scores"].numpy(), prediction["labels"].numpy()):
            x1 = float((box[0] - pad_x) / ratio)
            y1 = float((box[1] - pad_y) / ratio)
            x2 = float((box[2] - pad_x) / ratio)
            y2 = float((box[3] - pad_y) / ratio)
            x1, x2 = sorted([max(0.0, min(width, x1)), max(0.0, min(width, x2))])
            y1, y2 = sorted([max(0.0, min(height, y1)), max(0.0, min(height, y2))])
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append(
                {
                    "cls": int(label),
                    "score": float(score),
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "xn": x1 / width,
                    "yn": y1 / height,
                    "wn": (x2 - x1) / width,
                    "hn": (y2 - y1) / height,
                }
            )
        return {"task": "detect", "width": width, "height": height, "boxes": boxes, "ms": round(elapsed, 1)}

    if entry["task"] == "classify":
        probs = output.softmax(1)[0].numpy()
        order = probs.argsort()[::-1][:5]
        return {
            "task": "classify",
            "width": width,
            "height": height,
            "predictions": [{"cls": int(i), "score": float(probs[i])} for i in order],
            "ms": round(elapsed, 1),
        }

    logits = output[0]
    mask = logits.argmax(0).numpy().astype(np.uint8)
    new_w, new_h, px, py = letterbox_size(width, height, size)
    cropped = mask[py : py + new_h, px : px + new_w]
    mask_image = Image.fromarray(cropped, mode="L").resize((width, height), Image.NEAREST)
    arr = np.asarray(mask_image)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    present = []
    total = arr.size
    for cls in range(1, len(entry["classes"]) + 1):
        region = arr == cls
        count = int(region.sum())
        if count:
            color = CLASS_COLORS[(cls - 1) % len(CLASS_COLORS)]
            rgba[region] = (
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16),
                145,
            )
            present.append({"cls": cls - 1, "fraction": count / total})
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    import base64

    return {
        "task": "segment",
        "width": width,
        "height": height,
        "present": sorted(present, key=lambda p: -p["fraction"]),
        "mask_png": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "ms": round(elapsed, 1),
    }


def _predict_many(project_id: str, run_id: str, payloads: List[bytes], conf: float, iou: float) -> List[Dict]:
    entry = _load_run_model(project_id, run_id)
    return [_predict_one(entry, data, conf, iou) for data in payloads]


@app.post("/projects/{project_id}/runs/{run_id}/predict")
async def predict(
    project_id: str,
    run_id: str,
    files: List[UploadFile] = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
):
    if len(files) > 8:
        raise HTTPException(400, "send at most 8 images per request")
    payloads = [await f.read() for f in files]
    try:
        results = await run_in_threadpool(_predict_many, project_id, run_id, payloads, conf, iou)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    return {"results": results, "classes": _load_run_model(project_id, run_id)["classes"]}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@app.post("/projects/{project_id}/runs/{run_id}/evaluate")
def evaluate_run(project_id: str, run_id: str):
    import torch
    from torch.utils.data import DataLoader

    from .data import ClassificationDataset, DetectionDataset, SegmentationDataset, detection_collate

    root = project_path(project_id)
    manifest = load_manifest(root)
    entry = _load_run_model(project_id, run_id)
    task = entry["task"]
    size = entry["size"]
    classes = entry["classes"]
    result = entry["result"]
    device = pick_device("cpu")

    if task == "detect":
        dataset = DetectionDataset(root, manifest, "val", size, augment=False)
        if len(dataset) == 0:
            raise HTTPException(400, "this run has no validation images")
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=detection_collate)
        metrics = train_detect.evaluate_detector(entry["model"], loader, size, len(classes), device)
        return {"task": task, "classes": classes, "images": len(dataset), "metrics": metrics, "result": result}

    if task == "classify":
        dataset = ClassificationDataset(root, manifest, "val", size, augment=False)
        if len(dataset) == 0:
            raise HTTPException(400, "this run has no validation images")
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        metrics = train_cls.evaluate_classifier(entry["model"], loader, device, len(classes))
        return {"task": task, "classes": classes, "images": len(dataset), "metrics": metrics, "result": result}

    dataset = SegmentationDataset(root, manifest, "val", size, augment=False)
    if len(dataset) == 0:
        raise HTTPException(400, "this run has no validation images")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    metrics = train_seg.evaluate_segmentation(entry["model"], loader, device, len(classes) + 1)
    return {"task": task, "classes": classes, "images": len(dataset), "metrics": metrics, "result": result}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


@app.post("/projects/{project_id}/runs/{run_id}/export")
def export(project_id: str, run_id: str):
    directory = run_path(project_id, run_id)
    if not (directory / "result.json").exists():
        raise HTTPException(400, "train the model before exporting")
    dataset_zip = None
    try:
        dataset_zip = _dataset_zip(project_id)
    except FileNotFoundError:
        dataset_zip = None
    info = exporter.export_run(directory, directory / "export", dataset_zip)
    info["downloads"] = {
        name: f"/projects/{project_id}/runs/{run_id}/export/{name}" for name in info["files"]
    }
    return info


@app.get("/projects/{project_id}/runs/{run_id}/export/{name}")
def download_export(project_id: str, run_id: str, name: str):
    directory = run_path(project_id, run_id) / "export"
    try:
        path = exporter.read_export_file(directory, name)
    except FileNotFoundError:
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=path.name)


# --------------------------------------------------------------------------- #
# Pretrained annotation suggestions
# --------------------------------------------------------------------------- #


@app.post("/autolabel")
async def autolabel(
    files: List[UploadFile] = File(...),
    classes: str = Form("[]"),
    model: str = Form("fasterrcnn-mobilenet"),
    conf: float = Form(0.25),
):
    if len(files) > 16:
        raise HTTPException(400, "send at most 16 images per request")
    try:
        class_list = json.loads(classes)
    except json.JSONDecodeError:
        raise HTTPException(400, "classes must be a JSON array")
    if model not in pretrained.CATALOG:
        raise HTTPException(400, "unknown pretrained model")
    payloads = [(f.filename or f"image-{i}", await f.read()) for i, f in enumerate(files)]

    def work():
        results = []
        for name, data in payloads:
            suggestion = pretrained.suggest(data, class_list, model, conf)
            suggestion["file"] = name
            results.append(suggestion)
        return results

    try:
        results = await run_in_threadpool(work)
    except Exception as exc:
        raise HTTPException(500, f"pretrained detector failed: {type(exc).__name__}: {exc}")
    return {"results": results, "model": model, "catalog": pretrained.CATALOG}


# --------------------------------------------------------------------------- #
# Project lifecycle
# --------------------------------------------------------------------------- #


@app.delete("/projects/{project_id}")
def delete_project(project_id: str):
    root = project_path(project_id)
    if root.exists():
        shutil.rmtree(root)
    with MODEL_LOCK:
        for key in [k for k in MODEL_CACHE if k.startswith(f"{project_id}/")]:
            MODEL_CACHE.pop(key, None)
    return {"deleted": project_id}


@app.get("/health/simple")
def simple_health():
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn

    print(f"VisionLLM engine on http://127.0.0.1:{PORT} — data in {HOME}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
