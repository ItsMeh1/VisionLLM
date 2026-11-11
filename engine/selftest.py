"""End-to-end self test on a generated dataset.

Proves the pipeline really trains: synthetic images with known objects are
written to a temporary project, the detector is trained for a few epochs, and
the test asserts that validation mAP is meaningfully above chance, that
inference finds the objects, and that ONNX export matches PyTorch.

Run: engine/.venv/bin/python -m engine.selftest
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import exporter, train_detect
from .letterbox import letterbox_image


def make_dataset(root: Path, count: int = 48) -> dict:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    np.random.seed(7)
    classes = ["square", "circle"]
    records = []
    for i in range(count):
        width, height = random.choice([(320, 240), (240, 320), (300, 300), (400, 200)])
        array = np.full((height, width, 3), 30, dtype=np.uint8)
        array += np.random.randint(0, 26, size=(height, width, 1), dtype=np.uint8)
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        boxes = []
        for _ in range(random.randint(1, 2)):
            cls = random.randint(0, 1)
            size = random.randint(28, min(68, width // 3, height // 3))
            x = random.randint(4, width - size - 5)
            y = random.randint(4, height - size - 5)
            color = (240, 90, 40) if cls == 0 else (60, 170, 235)
            if cls == 0:
                draw.rectangle([x, y, x + size, y + size], fill=color)
            else:
                draw.ellipse([x, y, x + size, y + size], fill=color)
            boxes.append(
                {"cls": cls, "x": x / width, "y": y / height, "w": size / width, "h": size / height}
            )
        filename = f"img_{i:03d}.png"
        image.save(images_dir / filename)
        records.append(
            {
                "id": f"img_{i:03d}",
                "file": filename,
                "split": "val" if i % 5 == 0 else "train",
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )
    manifest = {"task": "detect", "classes": classes, "images": records}
    (root / "dataset.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="visionllm-selftest-"))
    try:
        make_dataset(workspace)
        run_dir = workspace / "runs" / "test"
        config = {
            "task": "detect",
            "model": "vlyolo-n",
            "epochs": 18,
            "batch": 6,
            "imgsz": 320,
            "lr": 0.002,
            "device": "auto",
        }
        events = []

        def emit(kind, **fields):
            if kind == "epoch":
                record = fields["record"]
                print(
                    f"  epoch {record['epoch']:2d}  loss {record['train_loss']:.3f}  "
                    f"mAP@0.5 {record['map50']:.3f}  mAP@.5:.95 {record['map5095']:.3f}"
                )
            else:
                events.append(fields)
                print(f"  {fields.get('message', kind)}")

        print("Training VLYOLO-N on synthetic data (aspect ratios vary, none are square-padded by cropping)...")
        result = train_detect.train(workspace, run_dir, config, emit, lambda: False)
        print(f"Final mAP@0.5 = {result['metrics']['map50']:.4f}")
        print(f"Final precision = {result['metrics']['precision']:.3f} recall = {result['metrics']['recall']:.3f}")
        assert result["epochs_run"] == 18, "did not run the requested epochs"
        assert result["history"][0]["train_loss"] > 0, "no training loss recorded"
        assert result["metrics"]["map50"] > 0.35, f"mAP too low to be real training: {result['metrics']['map50']}"

        # Real inference on a held-out image.
        import torch

        from .server import _predict_one
        from .vlyolo import VLYOLO

        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model = VLYOLO(2, "vlyolo-n")
        model.load_state_dict(checkpoint["ema"])
        model.eval()
        entry = {"task": "detect", "model": model, "size": 320, "classes": ["square", "circle"]}
        sample = (workspace / "images" / "img_000.png").read_bytes()
        prediction = _predict_one(entry, sample, 0.25, 0.45)
        print(f"Inference found {len(prediction['boxes'])} boxes: {[round(b['score'], 3) for b in prediction['boxes']]}")
        assert prediction["boxes"], "inference returned nothing"

        info = exporter.export_run(run_dir, run_dir / "export")
        print(f"ONNX export: {info['onnx_bytes']} bytes, verification={info['verification']}")
        assert info["verification"]["ran"], "ONNX Runtime could not load the exported model"
        assert info["verification"]["max_abs_diff"] < 1e-4, "ONNX output does not match PyTorch"
        print("SELFTEST PASSED")
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
