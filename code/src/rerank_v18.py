"""V18 query-routed post-ranker built on the frozen V17 scoring behavior."""

from __future__ import annotations

import re

from .rerank import rerank_candidates as rerank_v17_candidates


ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def query_route(query: str):
    """Return a conservative deterministic route, or None for V17 behavior."""
    text = query.lower()
    ordinal = None
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", text)
    if match:
        ordinal = int(match.group(1))
    else:
        for word, value in ORDINALS.items():
            if re.search(rf"\b{word}\b", text):
                ordinal = value
                break

    from_left = bool(re.search(r"\b(?:from|counting from)\s+(?:the\s+)?left\b|\bleft\s+to\s+right\b", text))
    from_right = bool(re.search(r"\b(?:from|counting from)\s+(?:the\s+)?right\b|\bright\s+to\s+left\b", text))
    relational_reference = bool(
        re.search(r"\b(?:left|right)\s+of\b|\b(?:next|close)\s+to\b|\bbehind\b|\bin front of\b|\bimmediately\b", text)
    )
    ordinal_request = bool(
        re.match(r"\s*(?:from\s+(?:the\s+)?(?:left|right)|(?:count|number|select|find)\b)", text)
        or re.search(r"\bthe\s+(?:\d+(?:st|nd|rd|th)|" + "|".join(ORDINALS) + r")\s+[^,.]{0,50}\s+from\s+(?:the\s+)?(?:left|right)\s*$", text)
    )
    if ordinal and (from_left or from_right) and ordinal_request and not relational_reference:
        return {"kind": "ordinal_x", "index": ordinal, "reverse": from_right}

    # Only unambiguous image-global extremes become hard routes. Relational
    # phrases such as "left of the towel" deliberately stay on V17 scoring.
    if re.search(r"\bleftmost\b|\bfar left\b", text):
        return {"kind": "extreme_x", "reverse": False}
    if re.search(r"\brightmost\b|\bfar right\b", text):
        return {"kind": "extreme_x", "reverse": True}
    return None


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _deduplicate(candidates, threshold=0.85):
    """Remove near-identical boxes before positional/ordinal sorting."""
    kept = []
    for candidate in candidates:
        if all(_iou(candidate["bbox"], item["bbox"]) < threshold for item in kept):
            kept.append(candidate)
    return kept


def rerank_candidates(sample, candidates, weights=None, route_mode="all"):
    ranked = rerank_v17_candidates(sample, candidates, weights=weights)
    route = query_route(sample["query"])
    enabled_kind = {
        "all": {"extreme_x", "ordinal_x"},
        "extreme": {"extreme_x"},
        "ordinal": {"ordinal_x"},
        "ordinal_first": {"ordinal_x"},
        "ordinal_later": {"ordinal_x"},
        "ordinal_second": {"ordinal_x"},
        "ordinal_third_plus": {"ordinal_x"},
    }
    if route_mode not in enabled_kind:
        raise ValueError(f"Unsupported route_mode: {route_mode}")
    if not ranked or route is None or route["kind"] not in enabled_kind[route_mode]:
        return ranked
    if route_mode == "ordinal_first" and route["index"] != 1:
        return ranked
    if route_mode == "ordinal_later" and route["index"] == 1:
        return ranked
    if route_mode == "ordinal_second" and route["index"] != 2:
        return ranked
    if route_mode == "ordinal_third_plus" and route["index"] < 3:
        return ranked

    unique = _deduplicate(ranked)
    if route["kind"] == "ordinal_x":
        index = route["index"]
        # Never guess an ordinal that the saved candidate pool cannot support.
        if len(unique) < index:
            return ranked
        ordered = sorted(
            unique,
            key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2.0,
            reverse=route["reverse"],
        )
        selected = ordered[index - 1]
    else:
        selected = max(
            unique,
            key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2.0,
        ) if route["reverse"] else min(
            unique,
            key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2.0,
        )

    result = [selected] + [item for item in ranked if item is not selected]
    for rank, item in enumerate(result, start=1):
        item["rerank_rank"] = rank
        item["v18_route"] = route if rank == 1 else None
    return result
