"""Bounding-box postprocessing helpers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, List, Optional

from .metrics import is_valid_bbox


DEFAULT_FALLBACK = [0.25, 0.25, 0.75, 0.75]


def sanitize_bbox(bbox, fallback: Optional[Iterable[float]] = None) -> List[float]:
    """Clip and repair a bbox into a valid normalized [x1, y1, x2, y2] box."""
    fallback_box = list(fallback) if fallback is not None else DEFAULT_FALLBACK.copy()

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return fallback_box

    values = []
    for value in bbox:
        if not isinstance(value, Real):
            return fallback_box
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return fallback_box
        values.append(min(1.0, max(0.0, value)))

    x1, y1, x2, y2 = values
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    min_size = 1e-6
    if x2 - x1 < min_size:
        center = (x1 + x2) / 2.0
        x1 = max(0.0, center - min_size / 2.0)
        x2 = min(1.0, center + min_size / 2.0)
    if y2 - y1 < min_size:
        center = (y1 + y2) / 2.0
        y1 = max(0.0, center - min_size / 2.0)
        y2 = min(1.0, center + min_size / 2.0)

    repaired = [x1, y1, x2, y2]
    return repaired if is_valid_bbox(repaired) else fallback_box
