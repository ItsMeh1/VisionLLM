"""Image classification training: real fine-tuning of a ResNet-18 backbone."""

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

from .data import ClassificationDataset, class_names, load_manifest
from .train_detect import ModelEMA, pick_device

Emit = Callable[..., None]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class NormalizedClassifier(nn.Module):
    """Wraps a classifier so raw 0..1 RGB is the exported input."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


class SoftmaxExport(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def build_classifier(num_classes: int, pretrained: bool = True):
    import torchvision

    used_pretrained = False
    model = None
    if pretrained:
        try:
            model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            used_pretrained = True
        except Exception:
            model = None
    if model is None:
        model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, used_pretrained


@torch.no_grad()
def evaluate_classifier(model, loader, device, num_classes: int) -> Dict:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total = 0
    correct = 0
    loss_sum = 0.0
    images = []
    for x, y, meta in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
        probs = logits.softmax(1)
        preds = probs.argmax(1)
        for t, p, pr, m in zip(y.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy(), meta["image_id"]):
            confusion[t, p] += 1
            total += 1
            correct += int(t == p)
            top = pr.argsort()[-3:][::-1]
            images.append(
                {
                    "image_id": m,
                    "truth": int(t),
                    "prediction": int(p),
                    "top": [{"cls": int(i), "score": float(pr[i])} for i in top],
                }
            )
    per_class = []
    f1s = []
    for c in range(num_classes):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)
        fn = int(confusion[c, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1s.append(f1)
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(confusion[c, :].sum()),
            }
        )
    return {
        "accuracy": correct / max(total, 1),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "val_loss": round(loss_sum / max(total, 1), 4),
        "images": images,
        "count": total,
    }


def train(project_root: Path, run_dir: Path, config: Dict, emit: Emit, should_stop: Callable[[], bool]) -> Dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(project_root)
    classes = class_names(manifest)
    num_classes = len(classes)
    if num_classes < 2:
        raise ValueError("classification needs at least two classes")
    size = int(config.get("imgsz", 224))
    epochs = int(config.get("epochs", 20))
    batch_size = int(config.get("batch", 16))
    lr = float(config.get("lr", 0.0003))
    device = pick_device(config.get("device", "auto"))
    torch.manual_seed(int(config.get("seed", 0)))

    train_ds = ClassificationDataset(project_root, manifest, "train", size, augment=True)
    val_ds = ClassificationDataset(project_root, manifest, "val", size, augment=False)
    notes: List[str] = []
    if len(train_ds) < 2:
        raise ValueError("not enough labelled training images")
    if len(val_ds) == 0:
        notes.append("No validation images were assigned; the last 15% of training images were held out.")
        items = train_ds.items
        cut = max(1, int(len(items) * 0.85))
        train_ds.items = items[:cut]
        val_ds.items = items[cut:] or items[-1:]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(16, len(val_ds)), shuffle=False, num_workers=0)
    emit("stage", message=f"{len(train_ds)} training / {len(val_ds)} validation images", device=str(device))

    backbone, used_pretrained = build_classifier(num_classes, pretrained=bool(config.get("pretrained", True)))
    model = NormalizedClassifier(backbone).to(device)
    emit(
        "stage",
        message=(
            "ImageNet-pretrained ResNet-18 backbone (BSD-3-Clause), fine-tuned on your classes"
            if used_pretrained
            else "ResNet-18 trained from scratch (pretrained weights were unavailable offline)"
        ),
        pretrained=used_pretrained,
    )
    ema = ModelEMA(model, decay=0.995)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: List[Dict] = []
    best_acc = -1.0
    best_epoch = 0
    started = time.time()
    cancelled = False

    for epoch in range(1, epochs + 1):
        if should_stop():
            cancelled = True
            break
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for x, y, _ in train_loader:
            if should_stop():
                cancelled = True
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema.update(model)
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        if cancelled:
            break
        scheduler.step()
        eval_model = copy.deepcopy(model)
        ema.apply_to(eval_model)
        metrics = evaluate_classifier(eval_model, val_loader, device, num_classes)
        record = {
            "epoch": epoch,
            "train_loss": round(loss_sum / max(total, 1), 4),
            "train_acc": round(correct / max(total, 1), 4),
            "val_loss": metrics["val_loss"],
            "val_acc": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed": round(time.time() - started, 1),
        }
        history.append(record)
        payload = {
            "format": "visionllm-resnet18-1",
            "task": "classify",
            "imgsz": size,
            "classes": classes,
            "config": config,
            "state_dict": model.state_dict(),
            "ema": ema.shadow,
            "metrics": record,
            "epoch": epoch,
        }
        torch.save(payload, run_dir / "last.pt")
        if metrics["accuracy"] > best_acc + 1e-4:
            best_acc = metrics["accuracy"]
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
    final_metrics = evaluate_classifier(final_model, val_loader, device, num_classes)
    result = {
        "task": "classify",
        "status": "cancelled" if cancelled else "completed",
        "classes": classes,
        "imgsz": size,
        "epochs_run": len(history),
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
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
