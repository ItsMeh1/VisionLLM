"""Real object-detection training loop for VLYOLO."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DetectionDataset, class_names, detection_collate, load_manifest
from .metrics import mean_average_precision, precision_recall_at
from .vlyolo import SPECS, VLYOLO, count_parameters, decode_predictions, detection_loss

Emit = Callable[..., None]


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(decay).add_(value.detach().float(), alpha=1 - decay)
            else:
                self.shadow[key].copy_(value)

    def apply_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        model.load_state_dict({k: v.to(state[k].dtype) for k, v in self.shadow.items()})


def pick_device(requested: str = "auto") -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _xyxy_to_normalised(boxes: np.ndarray, meta: Dict, score: np.ndarray, labels: np.ndarray) -> List[Dict]:
    width, height = meta["width"], meta["height"]
    ratio, pad_x, pad_y = meta["ratio"], meta["pad_x"], meta["pad_y"]
    out = []
    for box, sc, lab in zip(boxes, score, labels):
        x1 = (box[0] - pad_x) / ratio
        y1 = (box[1] - pad_y) / ratio
        x2 = (box[2] - pad_x) / ratio
        y2 = (box[3] - pad_y) / ratio
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append(
            {
                "cls": int(lab),
                "score": float(sc),
                "x": float(x1 / width),
                "y": float(y1 / height),
                "w": float((x2 - x1) / width),
                "h": float((y2 - y1) / height),
            }
        )
    return out


@torch.no_grad()
def evaluate_detector(
    model: VLYOLO,
    loader: DataLoader,
    size: int,
    num_classes: int,
    device: torch.device,
    conf: float = 0.25,
) -> Dict:
    model.eval()
    preds: List[Dict[str, np.ndarray]] = []
    gts: List[Dict[str, np.ndarray]] = []
    images_out: List[Dict] = []
    metas: List[Dict] = []
    val_loss = 0.0
    batches = 0
    for images, targets, meta in loader:
        images = images.to(device)
        feats = model(images)
        loss, _ = detection_loss(feats, [{k: v.to(device) for k, v in t.items()} for t in targets], size, num_classes)
        val_loss += float(loss)
        batches += 1
        decoded = decode_predictions(feats, size, conf_threshold=0.001, iou_threshold=0.5)
        for pred, target, m in zip(decoded, targets, meta):
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy().astype(np.int64)
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})
            gts.append({"boxes": target["boxes"].numpy(), "labels": target["labels"].numpy()})
            metas.append(m)
            images_out.append(
                {
                    "image_id": m["image_id"],
                    "width": m["width"],
                    "height": m["height"],
                    "boxes": _xyxy_to_normalised(boxes, m, scores, labels),
                }
            )
    metrics = mean_average_precision(preds, gts, num_classes)
    metrics.update(precision_recall_at(preds, gts, num_classes, conf_threshold=conf))
    metrics["val_loss"] = round(val_loss / max(batches, 1), 4)
    metrics["images"] = images_out
    metrics["ground_truth"] = [
        {
            "image_id": m["image_id"],
            "boxes": [
                {
                    "cls": int(lab),
                    "x": float(max(0.0, min(1.0, (b[0] - m["pad_x"]) / m["ratio"] / m["width"]))),
                    "y": float(max(0.0, min(1.0, (b[1] - m["pad_y"]) / m["ratio"] / m["height"]))),
                    "w": float(max(0.0, min(1.0, (b[2] - b[0]) / m["ratio"] / m["width"]))),
                    "h": float(max(0.0, min(1.0, (b[3] - b[1]) / m["ratio"] / m["height"]))),
                }
                for b, lab in zip(gt["boxes"], gt["labels"])
            ],
        }
        for gt, m in zip(gts, metas)
    ]
    return metrics


def train(
    project_root: Path,
    run_dir: Path,
    config: Dict,
    emit: Emit,
    should_stop: Callable[[], bool],
) -> Dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(project_root)
    classes = class_names(manifest)
    num_classes = max(1, len(classes))
    size = int(config.get("imgsz", 512))
    if size % 32 != 0:
        raise ValueError("input size must be a multiple of 32")
    variant = config.get("model", "vlyolo-s")
    if variant not in SPECS:
        raise ValueError(f"unknown model {variant!r}")
    epochs = int(config.get("epochs", 40))
    batch_size = int(config.get("batch", 8))
    lr = float(config.get("lr", 0.001))
    patience = int(config.get("patience", 0))
    seed = int(config.get("seed", 0))
    conf = float(config.get("conf", 0.25))
    device = pick_device(config.get("device", "auto"))

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = DetectionDataset(project_root, manifest, "train", size, augment=True)
    val_ds = DetectionDataset(project_root, manifest, "val", size, augment=False)
    if len(train_ds) == 0:
        raise ValueError("no training images with labels; add images and draw boxes first")
    notes: List[str] = []
    if len(val_ds) == 0:
        notes.append("No validation images were assigned, so the last 15% of training images were held out for validation.")
        items = train_ds.items
        cut = max(1, int(len(items) * 0.85))
        train_ds.items = items[:cut]
        val_ds.items = items[cut:] or items[-1:]

    emit("stage", message=f"Loaded {len(train_ds)} training and {len(val_ds)} validation images", device=str(device))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=detection_collate, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(4, len(val_ds)), shuffle=False, num_workers=0, collate_fn=detection_collate
    )

    model = VLYOLO(num_classes, variant).to(device)
    params = count_parameters(model)
    emit(
        "stage",
        message=f"{variant} · {params / 1e6:.2f}M parameters · {num_classes} classes · {size}px letterbox",
        parameters=params,
    )

    ema = ModelEMA(model, decay=0.999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config.get("weight_decay", 0.0005)))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup = min(50, max(1, int(total_steps * 0.05)))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    history: List[Dict] = []
    best_map = -1.0
    best_epoch = 0
    epochs_without_gain = 0
    started = time.time()
    step = 0
    cancelled = False

    def save_checkpoint(path: Path, metrics: Dict) -> None:
        torch.save(
            {
                "format": "visionllm-vlyolo-1",
                "task": "detect",
                "model": variant,
                "imgsz": size,
                "classes": classes,
                "config": config,
                "state_dict": model.state_dict(),
                "ema": ema.shadow,
                "metrics": metrics,
                "epoch": len(history),
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            path,
        )

    for epoch in range(1, epochs + 1):
        if should_stop():
            cancelled = True
            break
        model.train()
        running = {"total": 0.0, "cls": 0.0, "box": 0.0, "ctr": 0.0}
        batches = 0
        for images, targets, _ in train_loader:
            if should_stop():
                cancelled = True
                break
            images = images.to(device)
            moved = [{k: v.to(device) for k, v in t.items()} for t in targets]
            feats = model(images)
            loss, parts = detection_loss(feats, moved, size, num_classes)
            if not torch.isfinite(loss):
                raise RuntimeError("training diverged (non-finite loss). Lower the learning rate in advanced settings.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            step += 1
            batches += 1
            running["total"] += float(loss.detach())
            for key in ("cls", "box", "ctr"):
                running[key] += parts[key]
        if cancelled:
            break

        ema_model = copy.deepcopy(model)
        ema.apply_to(ema_model)
        val_metrics = evaluate_detector(ema_model, val_loader, size, num_classes, device, conf=conf)
        record = {
            "epoch": epoch,
            "train_loss": round(running["total"] / max(batches, 1), 4),
            "cls_loss": round(running["cls"] / max(batches, 1), 4),
            "box_loss": round(running["box"] / max(batches, 1), 4),
            "ctr_loss": round(running["ctr"] / max(batches, 1), 4),
            "val_loss": val_metrics["val_loss"],
            "map50": round(val_metrics["map50"], 4),
            "map5095": round(val_metrics["map5095"], 4),
            "precision": round(val_metrics["precision"], 4),
            "recall": round(val_metrics["recall"], 4),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed": round(time.time() - started, 1),
        }
        history.append(record)
        improved = record["map50"] > best_map + 1e-4
        if improved:
            best_map = record["map50"]
            best_epoch = epoch
            epochs_without_gain = 0
            save_checkpoint(run_dir / "best.pt", {**record, "per_class_ap50": val_metrics["per_class_ap50"]})
        else:
            epochs_without_gain += 1
        save_checkpoint(run_dir / "last.pt", record)
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
        emit("epoch", record=record, history=history)
        if patience and epochs_without_gain >= patience:
            notes.append(f"Early stopping at epoch {epoch}: mAP@0.5 had not improved for {patience} epochs.")
            break

    elapsed = time.time() - started
    final = history[-1] if history else {}
    # Final, full evaluation of the best checkpoint.
    best_path = run_dir / "best.pt" if (run_dir / "best.pt").exists() else run_dir / "last.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        eval_model = copy.deepcopy(model)
        if "ema" in checkpoint:
            eval_model.load_state_dict({k: v.to(eval_model.state_dict()[k].dtype) for k, v in checkpoint["ema"].items()})
        final_metrics = evaluate_detector(eval_model, val_loader, size, num_classes, device, conf=conf)
        result = {
            "task": "detect",
            "status": "cancelled" if cancelled else "completed",
            "model": variant,
            "parameters": params,
            "imgsz": size,
            "classes": classes,
            "epochs_run": len(history),
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "best_map50": best_map if best_map >= 0 else None,
            "metrics": {k: v for k, v in final_metrics.items() if k != "ground_truth"},
            "history": history,
            "elapsed_seconds": round(elapsed, 1),
            "notes": notes,
            "checkpoint": str(best_path),
            "device": str(device),
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
        emit("done", result=result)
        return result

    raise RuntimeError("training produced no checkpoint")
