"""VLYOLO — VisionLLM's anchor-free YOLO-family detector.

A deliberately small, readable, fully trainable implementation:
  CSP-style backbone -> FPN neck -> decoupled anchor-free heads
  FCOS-style point assignment, ltrb regression with GIoU loss,
  sigmoid focal classification loss and a centerness branch.

Why this exists instead of wrapping Ultralytics: the Ultralytics code and the
weights it produces are AGPL-3.0. This implementation is original work and is
Apache-2.0, so models trained here can be used without copyleft obligations.
The architecture follows the published anchor-free detection literature
(FCOS, YOLOX) rather than copying any AGPL source.

Licence: Apache-2.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDES: Tuple[int, ...] = (8, 16, 32)
REG_RANGES: Tuple[Tuple[float, float], ...] = ((0.0, 64.0), (64.0, 128.0), (128.0, 1e8))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    width: Tuple[int, int, int, int, int]
    depth: Tuple[int, int, int, int]
    head: int
    description: str


SPECS: Dict[str, ModelSpec] = {
    "vlyolo-n": ModelSpec(
        "vlyolo-n", (16, 32, 64, 128, 256), (1, 2, 2, 1), 64,
        "Nano. Fastest to train, best for a few hundred images and live webcam use.",
    ),
    "vlyolo-s": ModelSpec(
        "vlyolo-s", (24, 48, 96, 192, 384), (1, 2, 2, 1), 96,
        "Small. The balanced default for most projects.",
    ),
    "vlyolo-m": ModelSpec(
        "vlyolo-m", (32, 64, 128, 256, 512), (2, 3, 3, 2), 128,
        "Medium. Slower, more capacity for large datasets and small objects.",
    ),
}


class Conv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = Conv(c, c, 3)
        self.cv2 = Conv(c, c, 3)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """Cross-stage partial block with two chunks and n bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class Backbone(nn.Module):
    def __init__(self, w: Sequence[int], d: Sequence[int]):
        super().__init__()
        self.stem = Conv(3, w[0], 3, 2)
        self.stage1 = nn.Sequential(Conv(w[0], w[1], 3, 2), C2f(w[1], w[1], d[0]))
        self.stage2 = nn.Sequential(Conv(w[1], w[2], 3, 2), C2f(w[2], w[2], d[1]))
        self.stage3 = nn.Sequential(Conv(w[2], w[3], 3, 2), C2f(w[3], w[3], d[2]))
        self.stage4 = nn.Sequential(Conv(w[3], w[4], 3, 2), C2f(w[4], w[4], d[3]))

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class Head(nn.Module):
    def __init__(self, c: int, nc: int, h: int):
        super().__init__()
        self.stem = nn.Sequential(Conv(c, h, 3), Conv(h, h, 3))
        self.cls = nn.Conv2d(h, nc, 1)
        self.reg = nn.Conv2d(h, 4, 1)
        self.ctr = nn.Conv2d(h, 1, 1)

    def forward(self, x: torch.Tensor):
        f = self.stem(x)
        return self.cls(f), self.reg(f), self.ctr(f)


class VLYOLO(nn.Module):
    def __init__(self, num_classes: int, variant: str = "vlyolo-s"):
        super().__init__()
        if variant not in SPECS:
            raise ValueError(f"unknown variant {variant!r}")
        spec = SPECS[variant]
        self.spec = spec
        self.num_classes = num_classes
        self.backbone = Backbone(spec.width, spec.depth)
        c3, c4, c5 = spec.width[2], spec.width[3], spec.width[4]
        h = spec.head
        self.lat3, self.lat4, self.lat5 = Conv(c3, h, 1), Conv(c4, h, 1), Conv(c5, h, 1)
        self.smooth3, self.smooth4 = Conv(h, h, 3), Conv(h, h, 3)
        self.heads = nn.ModuleList(Head(h, num_classes, h) for _ in STRIDES)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        prior = 0.01
        bias = -math.log((1 - prior) / prior)
        for head in self.heads:
            nn.init.constant_(head.cls.bias, bias)
            nn.init.constant_(head.ctr.bias, bias)
            nn.init.zeros_(head.reg.bias)

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        p3, p4, p5 = self.backbone(x)
        f5 = self.lat5(p5)
        f4 = self.smooth4(self.lat4(p4) + F.interpolate(f5, size=p4.shape[-2:], mode="nearest"))
        f3 = self.smooth3(self.lat3(p3) + F.interpolate(f4, size=p3.shape[-2:], mode="nearest"))
        outs = []
        for feat, stride in zip((f3, f4, f5), STRIDES):
            cls, reg, ctr = self.heads[STRIDES.index(stride)](feat)
            outs.append({"cls": cls, "reg": reg, "ctr": ctr, "stride": stride})
        return outs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Target assignment
# --------------------------------------------------------------------------- #


def _points(h: int, w: int, stride: int, device: torch.device):
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
    return xs[None, :].expand(h, w), ys[:, None].expand(h, w)


@torch.no_grad()
def build_targets(
    targets: Sequence[Dict[str, torch.Tensor]],
    size: int,
    num_classes: int,
    device: torch.device,
    center_radius: float = 1.5,
):
    """Assign ground-truth boxes to feature-map locations, FCOS-style."""
    batch = len(targets)
    scale = size / 512.0
    cls_t, box_t, ctr_t, pos_t, stride_t = [], [], [], [], []

    for level, stride in enumerate(STRIDES):
        h = w = size // stride
        px, py = _points(h, w, stride, device)
        cls = torch.zeros(batch, h, w, num_classes, device=device)
        box = torch.zeros(batch, h, w, 4, device=device)
        ctr = torch.zeros(batch, h, w, 1, device=device)
        best_area = torch.full((batch, h, w), float("inf"), device=device)
        lo = REG_RANGES[level][0] * scale
        hi = REG_RANGES[level][1] * scale

        for b, target in enumerate(targets):
            for k in range(target["labels"].numel()):
                x1, y1, x2, y2 = target["boxes"][k].tolist()
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                left, top = px - x1, py - y1
                right, bottom = x2 - px, y2 - py
                inside = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)
                if not bool(inside.any()):
                    continue
                max_d = torch.maximum(torch.maximum(left, right), torch.maximum(top, bottom))
                in_level = (max_d >= lo) & (max_d < hi)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                near = ((px - cx).abs() < center_radius * stride) & ((py - cy).abs() < center_radius * stride)
                mask = inside & in_level & near
                if not bool(mask.any()):
                    continue
                area = (x2 - x1) * (y2 - y1)
                take = mask & (area < best_area[b])
                if not bool(take.any()):
                    continue
                best_area[b] = torch.where(take, torch.full_like(best_area[b], area), best_area[b])
                cls[b][take] = F.one_hot(
                    torch.full((int(take.sum()),), int(target["labels"][k].item()), device=device),
                    num_classes,
                ).float()
                box[b][take, 0] = left[take]
                box[b][take, 1] = top[take]
                box[b][take, 2] = right[take]
                box[b][take, 3] = bottom[take]
                lr = torch.minimum(left, right)[take] / torch.clamp(torch.maximum(left, right)[take], min=1e-6)
                tb = torch.minimum(top, bottom)[take] / torch.clamp(torch.maximum(top, bottom)[take], min=1e-6)
                ctr[b][take, 0] = torch.sqrt(torch.clamp(lr * tb, min=0.0))

        cls_t.append(cls.reshape(batch, -1, num_classes))
        box_t.append(box.reshape(batch, -1, 4))
        ctr_t.append(ctr.reshape(batch, -1, 1))
        pos_t.append(cls.reshape(batch, -1, num_classes).sum(-1) > 0)
        stride_t.append(torch.full((batch, h * w, 1), float(stride), device=device))

    return (
        torch.cat(cls_t, 1),
        torch.cat(box_t, 1),
        torch.cat(ctr_t, 1),
        torch.cat(pos_t, 1),
        torch.cat(stride_t, 1),
    )


def _xyxy_from_ltrb(px: torch.Tensor, py: torch.Tensor, ltrb: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [px - ltrb[:, 0], py - ltrb[:, 1], px + ltrb[:, 2], py + ltrb[:, 3]], dim=-1
    )


def detection_loss(
    feats: List[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    size: int,
    num_classes: int,
    weights: Tuple[float, float, float] = (1.0, 2.0, 1.0),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = feats[0]["cls"].device
    batch = feats[0]["cls"].shape[0]
    cls_logits, reg_logits, ctr_logits = [], [], []
    strides = []
    for f in feats:
        b, c, h, w = f["cls"].shape
        cls_logits.append(f["cls"].permute(0, 2, 3, 1).reshape(b, h * w, c))
        reg_logits.append(f["reg"].permute(0, 2, 3, 1).reshape(b, h * w, 4))
        ctr_logits.append(f["ctr"].permute(0, 2, 3, 1).reshape(b, h * w, 1))
        strides.append(torch.full((b, h * w, 1), float(f["stride"]), device=device))
    cls_logits = torch.cat(cls_logits, 1)
    reg_logits = torch.cat(reg_logits, 1)
    ctr_logits = torch.cat(ctr_logits, 1)
    stride_t = torch.cat(strides, 1)

    cls_t, box_t, ctr_t, pos, _ = build_targets(targets, size, num_classes, device)

    # Classification: sigmoid focal loss over every location.
    p = torch.sigmoid(cls_logits)
    ce = F.binary_cross_entropy_with_logits(cls_logits, cls_t, reduction="none")
    p_t = p * cls_t + (1 - p) * (1 - cls_t)
    alpha_t = 0.25 * cls_t + 0.75 * (1 - cls_t)
    cls_loss = (ce * (1 - p_t).pow(2) * alpha_t).sum()

    num_pos = pos.sum().clamp(min=1)
    if pos.any():
        points = _points_grid(size, device)
        px_all, py_all = points[:, 0], points[:, 1]
        px = px_all.unsqueeze(0).expand_as(pos)[pos]
        py = py_all.unsqueeze(0).expand_as(pos)[pos]
        pred_ltrb = F.softplus(reg_logits[pos]).clamp(max=100.0) * stride_t[pos]
        pred_boxes = _xyxy_from_ltrb(px, py, pred_ltrb)
        tgt_boxes = _xyxy_from_ltrb(px, py, box_t[pos])
        giou = _giou(pred_boxes, tgt_boxes)
        reg_loss = (1 - giou).sum()
        ctr_loss = F.binary_cross_entropy_with_logits(ctr_logits[pos], ctr_t[pos], reduction="sum")
    else:
        reg_loss = cls_logits.sum() * 0.0
        ctr_loss = reg_loss

    w_cls, w_reg, w_ctr = weights
    total = w_cls * cls_loss / num_pos + w_reg * reg_loss / num_pos + w_ctr * ctr_loss / num_pos
    parts = {
        "cls": float((cls_loss / num_pos).detach()),
        "box": float((reg_loss / num_pos).detach()),
        "ctr": float((ctr_loss / num_pos).detach()),
        "positives": float(pos.sum().detach()),
    }
    return total, parts


def _points_grid(size: int, device):
    """Per-location centre coordinates for concatenated levels, in pixels."""
    coords = []
    for stride in STRIDES:
        h = w = size // stride
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        coords.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], -1))
    return torch.cat(coords, 0)


def _giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area_a + area_b - inter + 1e-9
    iou = inter / union
    lt_c = torch.minimum(a[:, :2], b[:, :2])
    rb_c = torch.maximum(a[:, 2:], b[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1] + 1e-9
    return iou - (area_c - union) / area_c


# --------------------------------------------------------------------------- #
# Inference decoding
# --------------------------------------------------------------------------- #


@torch.no_grad()
def decode_predictions(
    feats: List[Dict[str, torch.Tensor]],
    size: int,
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    """Decode raw heads into per-image boxes (pixels in the letterboxed input)."""
    from torchvision.ops import batched_nms

    device = feats[0]["cls"].device
    batch = feats[0]["cls"].shape[0]
    all_boxes: List[List[torch.Tensor]] = [[] for _ in range(batch)]
    all_scores: List[List[torch.Tensor]] = [[] for _ in range(batch)]
    all_labels: List[List[torch.Tensor]] = [[] for _ in range(batch)]

    for f in feats:
        stride = f["stride"]
        b, c, h, w = f["cls"].shape
        px, py = _points(h, w, stride, device)
        cls = f["cls"].sigmoid().permute(0, 2, 3, 1).reshape(b, h * w, c)
        ctr = f["ctr"].sigmoid().permute(0, 2, 3, 1).reshape(b, h * w, 1).sqrt()
        reg = (F.softplus(f["reg"].permute(0, 2, 3, 1).reshape(b, h * w, 4)) * stride).clamp(max=size * 2)
        scores = cls * ctr
        best, labels = scores.max(-1)
        for i in range(b):
            keep = best[i] > conf_threshold
            if not bool(keep.any()):
                continue
            l = reg[i, keep, 0]
            t = reg[i, keep, 1]
            r = reg[i, keep, 2]
            bo = reg[i, keep, 3]
            x1, y1 = px.reshape(-1)[keep] - l, py.reshape(-1)[keep] - t
            x2, y2 = px.reshape(-1)[keep] + r, py.reshape(-1)[keep] + bo
            all_boxes[i].append(torch.stack([x1, y1, x2, y2], -1))
            all_scores[i].append(best[i][keep])
            all_labels[i].append(labels[i][keep])

    results = []
    for i in range(batch):
        if not all_boxes[i]:
            results.append(
                {
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                }
            )
            continue
        boxes = torch.cat(all_boxes[i])
        scores = torch.cat(all_scores[i])
        labels = torch.cat(all_labels[i])
        keep = batched_nms(boxes, scores, labels, iou_threshold)[:max_detections]
        results.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
    return results


class ExportWrapper(nn.Module):
    """Fixed-size ONNX graph: [1,3,S,S] -> [1, A, 4+C] (cx,cy,w,h + class scores).

    Boxes are in letterboxed input pixels and scores already include centerness,
    so a ~40 line JavaScript postprocessor is enough for browser inference.
    """

    def __init__(self, model: VLYOLO):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        size = x.shape[-1]
        outputs = []
        for f in feats:
            b, c, h, w = f["cls"].shape
            stride = f["stride"]
            px, py = _points(h, w, stride, x.device)
            reg = F.softplus(f["reg"].permute(0, 2, 3, 1).reshape(b, h * w, 4)) * stride
            cls = f["cls"].sigmoid().permute(0, 2, 3, 1).reshape(b, h * w, c)
            ctr = f["ctr"].sigmoid().permute(0, 2, 3, 1).reshape(b, h * w, 1).sqrt()
            x1 = px.reshape(1, -1) - reg[..., 0]
            y1 = py.reshape(1, -1) - reg[..., 1]
            x2 = px.reshape(1, -1) + reg[..., 2]
            y2 = py.reshape(1, -1) + reg[..., 3]
            cx = (x1 + x2) / 2 / size
            cy = (y1 + y2) / 2 / size
            bw = (x2 - x1) / size
            bh = (y2 - y1) / size
            out = torch.cat([cx[..., None], cy[..., None], bw[..., None], bh[..., None], cls * ctr], -1)
            outputs.append(out)
        return torch.cat(outputs, 1)
