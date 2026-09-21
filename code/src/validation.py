"""Offline diagnostics for candidate-based visual grounding systems."""

from __future__ import annotations

import re
from collections import defaultdict

from .metrics import compute_iou, is_valid_bbox


CATEGORY_PATTERNS = {
    "horizontal_position": re.compile(r"\b(left|right|leftmost|rightmost)\b", re.I),
    "vertical_position": re.compile(r"\b(top|bottom|upper|lower|above|below|under)\b", re.I),
    "ordinal": re.compile(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b",
        re.I,
    ),
    "depth_relation": re.compile(
        r"\b(closest|nearest|farthest|far|foreground|background|behind|in front of)\b",
        re.I,
    ),
    "relation": re.compile(
        r"\b(left of|right of|next to|beside|near|between|behind|in front of|above|below|under)\b",
        re.I,
    ),
    "color": re.compile(
        r"\b(red|blue|green|yellow|white|black|pink|purple|brown|gray|grey|orange)\b",
        re.I,
    ),
    "action_pose": re.compile(
        r"\b(sitting|standing|walking|holding|wearing|eating|riding|flying|looking|lying|squatting)\b",
        re.I,
    ),
    "person": re.compile(r"\b(person|people|man|men|woman|women|girl|boy|child|pedestrian)\b", re.I),
    "animal": re.compile(r"\b(animal|monkey|bird|zebra|elephant|dog|cat|horse)\b", re.I),
}


def query_categories(query: str):
    """Return all diagnostic categories matched by a referring expression."""
    categories = [name for name, pattern in CATEGORY_PATTERNS.items() if pattern.search(query or "")]
    return categories or ["other"]


def _candidate_boxes(record):
    if isinstance(record, (list, tuple)) and len(record) == 4:
        return [record]
    if not isinstance(record, dict):
        return []
    candidates = record.get("candidates") or record.get("reranked_candidates") or []
    boxes = [item.get("bbox") for item in candidates if isinstance(item, dict)]
    if not boxes and record.get("bbox") is not None:
        boxes = [record.get("bbox")]
    return boxes


def evaluate_candidate_records(annotations, candidate_records, top_ks=(1, 5, 10, 20), threshold=0.5):
    """Measure top-1 and oracle hit rates for labeled annotations.

    ``annotations`` uses the competition JSON shape. ``candidate_records`` may
    contain saved detailed records (with ``candidates``) or plain bbox values.
    """
    top_ks = tuple(sorted(set(int(k) for k in top_ks if int(k) > 0)))
    if not top_ks:
        raise ValueError("top_ks must contain at least one positive integer")

    rows = []
    skipped_unlabeled = 0
    for sample_id, annotation in annotations.items():
        gt = annotation.get("bbox") if isinstance(annotation, dict) else None
        if not is_valid_bbox(gt):
            skipped_unlabeled += 1
            continue
        boxes = _candidate_boxes(candidate_records.get(sample_id))
        ious = [compute_iou(box, gt) for box in boxes]
        valid_count = sum(is_valid_bbox(box) for box in boxes)
        row = {
            "sample_id": sample_id,
            "query": annotation.get("query", ""),
            "categories": query_categories(annotation.get("query", "")),
            "candidate_count": len(boxes),
            "valid_candidate_count": valid_count,
            "top1_iou": ious[0] if ious else 0.0,
            "best_iou": max(ious, default=0.0),
            "best_rank": (ious.index(max(ious)) + 1) if ious else None,
            "hits": {str(k): max(ious[:k], default=0.0) >= threshold for k in top_ks},
        }
        rows.append(row)

    def aggregate(items):
        count = len(items)
        total_candidates = sum(row["candidate_count"] for row in items)
        valid_candidates = sum(row["valid_candidate_count"] for row in items)
        return {
            "samples": count,
            "candidate_coverage": sum(row["candidate_count"] > 0 for row in items) / count if count else 0.0,
            "invalid_candidate_rate": (total_candidates - valid_candidates) / total_candidates if total_candidates else 0.0,
            "mean_top1_iou": sum(row["top1_iou"] for row in items) / count if count else 0.0,
            "mean_best_iou": sum(row["best_iou"] for row in items) / count if count else 0.0,
            "acc_at_05": sum(row["top1_iou"] >= threshold for row in items) / count if count else 0.0,
            "oracle_acc": {
                str(k): sum(row["hits"][str(k)] for row in items) / count if count else 0.0
                for k in top_ks
            },
        }

    grouped = defaultdict(list)
    for row in rows:
        for category in row["categories"]:
            grouped[category].append(row)

    return {
        "threshold": threshold,
        "top_ks": list(top_ks),
        "labeled_samples": len(rows),
        "skipped_unlabeled": skipped_unlabeled,
        "overall": aggregate(rows),
        "by_category": {name: aggregate(items) for name, items in sorted(grouped.items())},
        "samples": rows,
    }
