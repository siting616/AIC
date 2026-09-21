"""Evaluation metrics for normalized bounding boxes."""

from __future__ import annotations

import math
from numbers import Real
from typing import Dict, Iterable, Optional


def is_valid_bbox(bbox) -> bool:
    """Return True when bbox is [x1, y1, x2, y2] in normalized coordinates."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False

    values = []
    for value in bbox:
        if not isinstance(value, Real):
            return False
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return False
        values.append(value)

    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        return False
    if min(values) < -1e-6 or max(values) > 1.0 + 1e-6:
        return False
    return True


def compute_iou(box_a, box_b) -> float:
    """Compute IoU for two normalized bboxes in [x1, y1, x2, y2] format."""
    if not is_valid_bbox(box_a) or not is_valid_bbox(box_b):
        return 0.0

    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_acc_at_05(
    pred_dict: Dict[str, Iterable[float]],
    gt_dict: Dict[str, Iterable[float]],
    threshold: float = 0.5,
) -> float:
    """Compute the ratio of samples whose predicted IoU is at least threshold."""
    if not gt_dict:
        return 0.0

    correct = 0
    total = 0
    for sample_id, gt_bbox in gt_dict.items():
        pred_bbox = pred_dict.get(sample_id)
        if compute_iou(pred_bbox, gt_bbox) >= threshold:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def per_sample_iou(
    pred_dict: Dict[str, Iterable[float]],
    gt_dict: Dict[str, Iterable[float]],
) -> Dict[str, float]:
    """Return IoU for every ground-truth sample id."""
    return {
        sample_id: compute_iou(pred_dict.get(sample_id), gt_bbox)
        for sample_id, gt_bbox in gt_dict.items()
    }
