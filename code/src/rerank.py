"""CPU-side multimodal reranking for GroundingDINO Top-K candidates."""

from __future__ import annotations

import re
from pathlib import Path


def _has(text: str, *terms: str) -> bool:
    return any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in terms)


def query_intents(query: str):
    """Extract conservative spatial and modality intents from an English query."""
    text = query.lower()
    from_left = bool(
        re.search(r"\bfrom\s+(?:the\s+)?left(?:\s+to\s+(?:the\s+)?right)?\b", text)
        or re.search(r"\bleft\s+to\s+right\b", text)
        or re.search(r"\bcounting\s+from\s+(?:the\s+)?left\b", text)
    )
    from_right = bool(
        re.search(r"\bfrom\s+(?:the\s+)?right(?:\s+to\s+(?:the\s+)?left)?\b", text)
        or re.search(r"\bright\s+to\s+left\b", text)
        or re.search(r"\bcounting\s+from\s+(?:the\s+)?right\b", text)
    )
    return {
        "left": from_left or (_has(text, "left", "leftmost") and not from_right),
        "right": from_right or (_has(text, "right", "rightmost") and not from_left),
        "center": _has(text, "center", "middle", "central"),
        "top": _has(text, "top", "upper", "above"),
        "bottom": _has(text, "bottom", "lower", "below"),
        "near": _has(text, "near", "nearest", "close", "closest", "front", "foreground"),
        "far": _has(text, "far", "farthest", "behind", "background"),
        "thermal": _has(
            text,
            "person", "people", "man", "woman", "boy", "girl", "pedestrian",
            "dog", "cat", "bird", "animal",
        ),
    }


def _center(bbox):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def position_score(bbox, intents):
    """Return a 0..1 score for explicit image-position terms."""
    center_x, center_y = _center(bbox)
    scores = []
    if intents["left"]:
        scores.append(1.0 - center_x)
    if intents["right"]:
        scores.append(center_x)
    if intents["top"]:
        scores.append(1.0 - center_y)
    if intents["bottom"]:
        scores.append(center_y)
    if intents["center"] and not (intents["left"] or intents["right"]):
        scores.append(1.0 - min(1.0, 2.0 * abs(center_x - 0.5)))
    return sum(scores) / len(scores) if scores else 0.5


def _region_values(image_path, bbox, kind):
    import numpy as np
    from PIL import Image

    with Image.open(Path(image_path)) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array.astype("float32").mean(axis=2)
    height, width = array.shape[:2]
    x1, y1, x2, y2 = bbox
    px1 = max(0, min(width - 1, int(float(x1) * width)))
    py1 = max(0, min(height - 1, int(float(y1) * height)))
    px2 = max(px1 + 1, min(width, int(float(x2) * width)))
    py2 = max(py1 + 1, min(height, int(float(y2) * height)))
    region = array[py1:py2, px1:px2].astype("float32")
    if kind == "depth" and array.dtype.kind in "ui" and array.dtype.itemsize >= 2:
        region = region[(region >= 300) & (region <= 20000)]
    else:
        region = region[region > 0]
    if region.size == 0:
        return None
    return {
        "median": float(np.median(region)),
        "mean": float(np.mean(region)),
        "std": float(np.std(region)),
    }


def _minmax_scores(values, reverse=False):
    valid = [value for value in values if value is not None]
    if not valid:
        return [0.5] * len(values)
    low, high = min(valid), max(valid)
    if high - low < 1e-9:
        return [0.5] * len(values)
    scores = []
    for value in values:
        if value is None:
            scores.append(0.5)
            continue
        score = (value - low) / (high - low)
        scores.append(1.0 - score if reverse else score)
    return scores


def rerank_candidates(sample, candidates, weights=None):
    """Rerank candidates using query position, depth and infrared evidence."""
    if not candidates:
        return []
    weights = {
        "model": 0.70,
        "position": 0.15,
        "depth": 0.10,
        "infrared": 0.05,
        **(weights or {}),
    }
    intents = query_intents(sample["query"])
    model_values = [float(candidate.get("score", 0.0)) for candidate in candidates]
    model_scores = _minmax_scores(model_values)
    position_scores = [
        position_score(candidate["bbox"], intents) for candidate in candidates
    ]

    depth_stats = [
        _region_values(sample["depth_path"], candidate["bbox"], "depth")
        if weights["depth"] > 0 and (intents["near"] or intents["far"])
        else None
        for candidate in candidates
    ]
    depth_values = [item["median"] if item else None for item in depth_stats]
    depth_scores = _minmax_scores(depth_values, reverse=intents["near"])
    if weights["depth"] <= 0 or not (intents["near"] or intents["far"]):
        depth_scores = [0.5] * len(candidates)

    infrared_stats = [
        _region_values(sample["infrared_path"], candidate["bbox"], "infrared")
        if weights["infrared"] > 0 and intents["thermal"]
        else None
        for candidate in candidates
    ]
    infrared_values = [item["mean"] if item else None for item in infrared_stats]
    infrared_scores = _minmax_scores(infrared_values)
    if weights["infrared"] <= 0 or not intents["thermal"]:
        infrared_scores = [0.5] * len(candidates)

    reranked = []
    for index, candidate in enumerate(candidates):
        active_position = any(
            intents[key] for key in ("left", "right", "top", "bottom", "center")
        )
        score = weights["model"] * model_scores[index]
        score += weights["position"] * (
            position_scores[index] if active_position else 0.5
        )
        score += weights["depth"] * depth_scores[index]
        score += weights["infrared"] * infrared_scores[index]
        enriched = dict(candidate)
        enriched["rerank_score"] = float(score)
        enriched["signals"] = {
            "model": model_scores[index],
            "position": position_scores[index],
            "depth": depth_scores[index],
            "infrared": infrared_scores[index],
            "depth_median": depth_values[index],
            "infrared_mean": infrared_values[index],
        }
        enriched["intents"] = intents
        reranked.append(enriched)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item["rerank_rank"] = rank
    return reranked
