"""Semantic segmentation training: a compact U-Net over painted polygons."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import SegmentationDataset, class_names, load_manifest
from .train_detect import ModelEMA, pick_device

Emit = Callable[..., None]


class DoubleConv(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(3, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([F.interpolate(self.up3(b), size=e3.shape[-2:], mode="nearest"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(self.up2(d3), size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(self.up1(d2), size=e1.shape[-2:], mode="nearest"), e1], 1))
        return self.out(d1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, 1)
    onehot = F.one_hot(target, probs.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    card = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + 1.0) / (card + 1.0)
    return 1 - dice[1:].mean()


@torch.no_grad()
def evaluate_segmentation(model, loader, device, num_classes: int) -> Dict:
    model.eval()
    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)
    correct = 0
    total = 0
    loss_sum = 0.0
    batches = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y))
        batches += 1
        preds = logits.argmax(1)
        for c in range(num_classes):
            p = preds == c
            t = y == c
            intersections[c] += float((p & t).sum())
            unions[c] += float((p | t).sum())
        correct += int((preds == y).sum())
        total += int(y.numel())
    ious = []
    per_class = []
    for c in range(num_classes):
        if unions[c] > 0:
            iou = intersections[c] / unions[c]
            ious.append(iou)
            per_class.append({"class": c, "iou": float(iou)})
        else:
            per_class.append({"class": c, "iou": None})
    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "pixel_accuracy": correct / max(total, 1),
        "per_class": per_class,
        "val_loss": round(loss_sum / max(batches, 1), 4),
    }


def train(project_root: Path, run_dir: Path, config: Dict, emit: Emit, should_stop: Callable[[], bool]) -> Dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(project_root)
    classes = class_names(manifest)
    num_classes = len(classes) + 1  # background
    size = int(config.get("imgsz", 384))
    if size % 8 != 0:
        raise ValueError("input size must be a multiple of 8")
    epochs = int(config.get("epochs", 30))
    batch_size = int(config.get("batch", 4))
    lr = float(config.get("lr", 0.001))
    device = pick_device(config.get("device", "auto"))
    torch.manual_seed(int(config.get("seed", 0)))

    train_ds = SegmentationDataset(project_root, manifest, "train", size, augment=True)
    val_ds = SegmentationDataset(project_root, manifest, "val", size, augment=False)
    notes: List[str] = []
    if len(train_ds) == 0:
        raise ValueError("no training images with painted masks")
    if len(val_ds) == 0:
        notes.append("No validation images were assigned; the last 15% of training images were held out.")
        items = train_ds.items
        cut = max(1, int(len(items) * 0.85))
        train_ds.items = items[:cut]
        val_ds.items = items[cut:] or items[-1:]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(4, len(val_ds)), shuffle=False, num_workers=0)
    emit("stage", message=f"{len(train_ds)} training / {len(val_ds)} validation masks", device=str(device))

    model = UNet(num_classes).to(device)
    emit("stage", message=f"U-Net · {num_classes} classes including background · {size}px")
    ema = ModelEMA(model, decay=0.995)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: List[Dict] = []
    best_iou = -1.0
    best_epoch = 0
    started = time.time()
    cancelled = False

    for epoch in range(1, epochs + 1):
        if should_stop():
            cancelled = True
            break
        model.train()
        loss_sum = 0.0
        batches = 0
        for x, y, _ in train_loader:
            if should_stop():
                cancelled = True
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema.update(model)
            loss_sum += float(loss.detach())
            batches += 1
        if cancelled:
            break
        scheduler.step()
        eval_model = copy.deepcopy(model)
        ema.apply_to(eval_model)
        metrics = evaluate_segmentation(eval_model, val_loader, device, num_classes)
        record = {
            "epoch": epoch,
            "train_loss": round(loss_sum / max(batches, 1), 4),
            "val_loss": metrics["val_loss"],
            "miou": round(metrics["miou"], 4),
            "pixel_accuracy": round(metrics["pixel_accuracy"], 4),
            "elapsed": round(time.time() - started, 1),
        }
        history.append(record)
        payload = {
            "format": "visionllm-unet-1",
            "task": "segment",
            "imgsz": size,
            "classes": classes,
            "config": config,
            "state_dict": model.state_dict(),
            "ema": ema.shadow,
            "metrics": record,
            "epoch": epoch,
        }
        torch.save(payload, run_dir / "last.pt")
        if metrics["miou"] > best_iou + 1e-4:
            best_iou = metrics["miou"]
            best_epoch = epoch
            torch.save(payload, run_dir / "best.pt")
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
        emit("epoch", record=record, history=history)

    best_path = run_dir / "best.pt" if (run_dir / "best.pt").exists() else run_dir / "last.pt"
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    final_model = copy.deepcopy(model)
    final_model.load_state_dict(checkpoint["state_dict"])
    if "ema" in checkpoint:
        final_model.load_state_dict(
            {k: v.to(final_model.state_dict()[k].dtype) for k, v in checkpoint["ema"].items()}
        )
    final_metrics = evaluate_segmentation(final_model, val_loader, device, num_classes)
    result = {
        "task": "segment",
        "status": "cancelled" if cancelled else "completed",
        "classes": classes,
        "imgsz": size,
        "epochs_run": len(history),
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_miou": best_iou if best_iou >= 0 else None,
        "metrics": final_metrics,
        "history": history,
        "elapsed_seconds": round(time.time() - started, 1),
        "notes": notes,
        "checkpoint": str(best_path),
        "device": str(device),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    emit("done", result=result)
    return result
