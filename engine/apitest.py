"""Exercise the HTTP API exactly as the browser does: upload, train, evaluate, predict, export.

Run with the engine already listening on 127.0.0.1:8765:
    engine/.venv/bin/python -m engine.apitest
"""

from __future__ import annotations

import io
import json
import random
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = "http://127.0.0.1:8765"
PROJECT = "apitest-" + uuid.uuid4().hex[:6]


def post_json(path: str, payload: dict):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def post_multipart(path: str, fields: dict, files: list[tuple[str, str, bytes]]):
    boundary = "----visionllm" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    for name, filename, data in files:
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: image/png\r\n\r\n"
        ).encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        BASE + path,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def make_image(index: int) -> tuple[bytes, list[dict], int, int]:
    random.seed(1000 + index)
    width, height = random.choice([(320, 240), (240, 320), (300, 300), (420, 210)])
    array = np.full((height, width, 3), 32, dtype=np.uint8)
    array += np.random.randint(0, 24, size=(height, width, 1), dtype=np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    boxes = []
    for _ in range(random.randint(1, 3)):
        cls = random.randint(0, 1)
        size = random.randint(24, min(64, width // 3, height // 3))
        x = random.randint(2, width - size - 2)
        y = random.randint(2, height - size - 2)
        color = (245, 80, 40) if cls == 0 else (60, 180, 240)
        if cls == 0:
            draw.rectangle([x, y, x + size, y + size], fill=color)
        else:
            draw.ellipse([x, y, x + size, y + size], fill=color)
        boxes.append({"cls": cls, "x": x / width, "y": y / height, "w": size / width, "h": size / height})
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), boxes, width, height


def main() -> int:
    health = json.loads(urllib.request.urlopen(BASE + "/health", timeout=20).read())
    print(f"engine {health['engine']} · torch {health['torch']} · device {health['device']}")

    fields: list = []
    files: list[tuple[str, str, bytes]] = []
    for index in range(24):
        data, boxes, width, height = make_image(index)
        name = f"api_{index:03d}"
        files.append(("files", f"{name}.png", data))
        fields.append(
            {
                "id": name,
                "file": f"{name}.png",
                "split": "val" if index % 6 == 0 else "train",
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )

    upload = post_multipart(
        f"/projects/{PROJECT}/dataset",
        {"manifest": json.dumps({"task": "detect", "classes": ["square", "circle"], "images": fields})},
        files,
    )
    print(f"uploaded {upload['stored']} images, split {upload['counts']}")
    assert upload["stored"] == 24, "upload lost images"

    started = post_json(
        f"/projects/{PROJECT}/train",
        {"model": "vlyolo-n", "epochs": 16, "batch": 4, "imgsz": 320, "lr": 0.003},
    )
    print(f"job {started['job_id']} · run {started['run_id']}")
    job = {}
    for _ in range(240):
        job = json.loads(urllib.request.urlopen(f"{BASE}/jobs/{started['job_id']}", timeout=20).read())
        if job["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(1)
    assert job["status"] == "completed", f"job ended {job['status']}: {job.get('error')}"
    epochs = [event["record"] for event in job["events"] if event["type"] == "epoch"]
    print(f"trained {len(epochs)} epochs; last mAP@0.5 = {epochs[-1]['map50']:.3f}; loss {epochs[-1]['train_loss']:.3f}")
    assert len(epochs) == 16, "did not receive one event per epoch"
    assert epochs[-1]["map50"] > 0.25, "metrics did not improve enough to be real training"

    evaluation = post_json(f"/projects/{PROJECT}/runs/{started['run_id']}/evaluate", {})
    print(
        f"evaluation on {evaluation['images']} val images: mAP@0.5 {evaluation['metrics']['map50']:.3f}, "
        f"precision {evaluation['metrics']['precision']:.3f}, recall {evaluation['metrics']['recall']:.3f}"
    )
    assert evaluation["metrics"]["images"], "evaluation returned no per-image predictions"

    sample, _, _, _ = make_image(3)
    boundary_result = post_multipart(
        f"/projects/{PROJECT}/runs/{started['run_id']}/predict",
        {"conf": "0.25", "iou": "0.45"},
        [("files", "sample.png", sample)],
    )
    prediction = boundary_result["results"][0]
    print(f"prediction: {len(prediction['boxes'])} boxes in {prediction['ms']} ms; classes {boundary_result['classes']}")
    assert prediction["width"] > 0, "prediction lost the original dimensions"

    export = post_json(f"/projects/{PROJECT}/runs/{started['run_id']}/export", {})
    print(f"export: {export['files']} · verification {export['verification']}")
    assert export["verification"]["ran"], "ONNX verification did not run"
    assert export["verification"]["max_abs_diff"] < 1e-4, "ONNX graph diverges from PyTorch"

    for name in ("labels.json", "inference.js", "model-card.json", "visionllm-model.zip"):
        assert name in export["files"], f"missing {name}"

    urllib.request.urlopen(urllib.request.Request(f"{BASE}/projects/{PROJECT}", method="DELETE"), timeout=30)
    print("API TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
