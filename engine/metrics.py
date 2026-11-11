"""Real metric computation: COCO-style AP for detection, plus IoU helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return float("nan")
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    grid = np.linspace(0, 1, 101)
    # np.interp needs increasing x; mrec is non-decreasing by construction.
    return float(np.mean(np.interp(grid, mrec, mpre)))


def mean_average_precision(
    predictions: Sequence[Dict[str, np.ndarray]],
    ground_truth: Sequence[Dict[str, np.ndarray]],
    num_classes: int,
    thresholds: Sequence[float] = tuple(np.arange(0.5, 1.0, 0.05)),
) -> Dict[str, object]:
    """Compute per-class AP across IoU thresholds and the aggregate mAP."""
    per_class = np.full((num_classes, len(thresholds)), np.nan, dtype=np.float64)
    for cls in range(num_classes):
        for ti, thr in enumerate(thresholds):
            scores: List[float] = []
            tps: List[int] = []
            positives = 0
            for pred, gt in zip(predictions, ground_truth):
                gt_mask = gt["labels"] == cls
                gt_boxes = gt["boxes"][gt_mask]
                positives += len(gt_boxes)
                p_mask = pred["labels"] == cls
                p_boxes = pred["boxes"][p_mask]
                p_scores = pred["scores"][p_mask]
                order = np.argsort(-p_scores)
                p_boxes, p_scores = p_boxes[order], p_scores[order]
                if len(gt_boxes) == 0:
                    scores.extend(p_scores.tolist())
                    tps.extend([0] * len(p_scores))
                    continue
                ious = box_iou_np(p_boxes, gt_boxes)
                matched = np.zeros(len(gt_boxes), dtype=bool)
                for i in range(len(p_boxes)):
                    tp = 0
                    if ious.shape[1]:
                        j = int(np.argmax(ious[i]))
                        if ious[i, j] >= thr and not matched[j]:
                            matched[j] = True
                            tp = 1
                    tps.append(tp)
                    scores.append(float(p_scores[i]))
            if positives == 0:
                continue
            arr_scores = np.asarray(scores, dtype=np.float64)
            arr_tp = np.asarray(tps, dtype=np.float64)
            order = np.argsort(-arr_scores)
            arr_tp = arr_tp[order]
            tp_cum = np.cumsum(arr_tp)
            fp_cum = np.cumsum(1 - arr_tp)
            recall = tp_cum / positives
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            per_class[cls, ti] = _ap_101(recall, precision)

    ap50 = per_class[:, 0]
    valid50 = ap50[~np.isnan(ap50)]
    valid_all = per_class[~np.isnan(per_class)]
    return {
        "map50": float(np.mean(valid50)) if valid50.size else 0.0,
        "map5095": float(np.mean(valid_all)) if valid_all.size else 0.0,
        "per_class_ap50": [None if np.isnan(v) else float(v) for v in ap50],
        "per_class_ap5095": [
            None if np.isnan(v) else float(v) for v in np.nanmean(per_class, axis=1)
        ],
        "per_class_ap_matrix": [
            [None if np.isnan(value) else float(value) for value in row] for row in per_class
        ],
        "thresholds": [float(t) for t in thresholds],
    }


def precision_recall_at(
    predictions: Sequence[Dict[str, np.ndarray]],
    ground_truth: Sequence[Dict[str, np.ndarray]],
    num_classes: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    tp = fp = 0
    positives = 0
    for pred, gt in zip(predictions, ground_truth):
        positives += len(gt["boxes"])
        keep = pred["scores"] >= conf_threshold
        boxes = pred["boxes"][keep]
        labels = pred["labels"][keep]
        matched = np.zeros(len(gt["boxes"]), dtype=bool)
        if len(gt["boxes"]) == 0:
            fp += len(boxes)
            continue
        ious = box_iou_np(boxes, gt["boxes"])
        for i in range(len(boxes)):
            candidates = np.where(gt["labels"] == labels[i])[0]
            best_j, best_iou = -1, 0.0
            for j in candidates:
                if not matched[j] and ious[i, j] > best_iou:
                    best_iou, best_j = ious[i, j], j
            if best_iou >= iou_threshold and best_j >= 0:
                matched[best_j] = True
                tp += 1
            else:
                fp += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(positives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": tp,
        "false_positives": fp,
        "ground_truth": positives,
    }
