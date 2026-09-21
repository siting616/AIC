"""Conservative direction-aware routing for explicit ordinal queries."""

from __future__ import annotations

import re

from .metrics import compute_iou


SECOND_PATTERN = re.compile(r"\b(second|2nd)\b", re.I)


def explicit_horizontal_ordinal(query):
    """Return (ordinal, direction), or None for expressions outside the safe gate."""
    text = str(query).lower()
    if not SECOND_PATTERN.search(text):
        return None
    if re.search(r"\bfrom\s+(?:the\s+)?right\b", text):
        return 2, "right"
    if re.search(r"\bfrom\s+(?:the\s+)?left\b", text):
        return 2, "left"
    return None


def deduplicate_candidates(candidates, threshold=0.85):
    """Remove near-identical boxes, retaining the strongest detector candidate."""
    kept = []
    ordered = sorted(
        enumerate(candidates),
        key=lambda pair: (-float(pair[1].get("score", 0.0)), pair[0]),
    )
    for _, candidate in ordered:
        if all(compute_iou(candidate["bbox"], item["bbox"]) < threshold for item in kept):
            kept.append(candidate)
    return kept


def select_ordinal_candidate(query, candidates, dedup_iou=0.85):
    parsed = explicit_horizontal_ordinal(query)
    if not parsed:
        return None
    ordinal, direction = parsed
    unique = deduplicate_candidates(candidates, dedup_iou)
    if len(unique) < ordinal:
        return None
    ordered = sorted(
        unique,
        key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2,
        reverse=direction == "right",
    )
    return ordered[ordinal - 1]


def apply_ordinal_router(annotations, cache_records, base_records, dedup_iou=0.85):
    """Override V1 top-1 only when the frozen conservative gate fires."""
    output = {}
    stats = {"samples": len(annotations), "eligible": 0, "changed": 0, "insufficient_candidates": 0}
    for sample_id, annotation in annotations.items():
        base = base_records[sample_id]
        base_candidates = [dict(item) for item in base.get("candidates", [])]
        cached = cache_records.get(sample_id, {}).get("candidates", [])
        parsed = explicit_horizontal_ordinal(annotation.get("query", ""))
        selected = select_ordinal_candidate(annotation.get("query", ""), cached, dedup_iou)
        if parsed:
            stats["eligible"] += 1
        if parsed and selected is None:
            stats["insufficient_candidates"] += 1
        if selected is not None:
            selected_bbox = selected["bbox"]
            selected_index = next(
                (i for i, item in enumerate(base_candidates) if item["bbox"] == selected_bbox), None
            )
            if selected_index is not None and selected_index != 0:
                chosen = base_candidates.pop(selected_index)
                chosen["ordinal_route"] = "second_from_left_or_right"
                base_candidates.insert(0, chosen)
                stats["changed"] += 1
        for rank, item in enumerate(base_candidates, 1):
            item["rank"] = rank
        output[sample_id] = {
            "bbox": base_candidates[0]["bbox"] if base_candidates else base.get("bbox"),
            "candidates": base_candidates,
        }
    return output, stats
