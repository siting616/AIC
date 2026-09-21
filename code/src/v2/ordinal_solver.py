"""Proposal-only scene-level ordinal solver.

No proposal produced here is automatically applied.  The solver builds a
shared family pool from existing candidates, removes near duplicates and
probable multi-instance containers, then reports deterministic assignments.
"""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import median
from typing import Any

from src.metrics import compute_iou

from .query_parser_v2 import parse_query
from .scene_planner import scene_id


def canonical_target(query: str) -> str | None:
    text = query.lower().strip(" .,;:!?")
    relation = re.search(r"\b(?:behind|in front of|left of|right of|next to|beside|near|under|above|on)\b", text)
    if relation:
        text = text[: relation.start()]
    text = re.sub(r"^(?:from\s+)?(?:left\s+to\s+right|right\s+to\s+left)\s*,?\s*", "", text)
    text = re.sub(r"^(?:find|locate|select|identify|choose|count)\s+", "", text)
    text = re.sub(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", "", text)
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text)
    text = re.sub(r"\b(?:from\s+(?:the\s+)?left|from\s+(?:the\s+)?right|leftmost|rightmost|far\s+left|far\s+right|left\s+to\s+right|right\s+to\s+left)\b", "", text)
    text = re.sub(r"\b(?:the|a|an)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text or None


def _box(candidate: dict[str, Any]) -> list[float] | None:
    value = candidate.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    box = [float(item) for item in value]
    return box if box[0] < box[2] and box[1] < box[3] else None


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _contains(outer: list[float], inner: list[float], fraction: float = 0.75) -> bool:
    x1, y1 = max(outer[0], inner[0]), max(outer[1], inner[1])
    x2, y2 = min(outer[2], inner[2]), min(outer[3], inner[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return overlap >= fraction * max(_area(inner), 1e-12)


def clean_candidates(
    candidates: list[dict[str, Any]],
    nms_iou: float = 0.60,
    group_area_ratio: float = 2.5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized = []
    for order, candidate in enumerate(candidates):
        box = _box(candidate)
        if box is None:
            continue
        normalized.append({
            **candidate,
            "bbox": box,
            "score": float(candidate.get("rerank_score", candidate.get("score", 0.0))),
            "_order": order,
        })
    normalized.sort(key=lambda item: (-item["score"], item["_order"]))
    kept = []
    for candidate in normalized:
        if all(compute_iou(candidate["bbox"], other["bbox"]) < nms_iou for other in kept):
            kept.append(candidate)

    areas = [_area(item["bbox"]) for item in kept]
    typical = median(areas) if areas else 0.0
    output = []
    group_count = 0
    for candidate in kept:
        smaller = [
            other for other in kept
            if other is not candidate
            and _area(other["bbox"]) < _area(candidate["bbox"])
            and _contains(candidate["bbox"], other["bbox"])
        ]
        is_group = bool(
            typical > 0
            and _area(candidate["bbox"]) > group_area_ratio * typical
            and len(smaller) >= 2
            and candidate["score"] <= max(item["score"] for item in smaller) + 0.05
        )
        if is_group:
            group_count += 1
        else:
            candidate.pop("_order", None)
            output.append(candidate)
    return output, {
        "raw": len(candidates),
        "valid": len(normalized),
        "after_nms": len(kept),
        "group_containers_removed": group_count,
        "clean": len(output),
    }


def _detail_candidates(detail: dict[str, Any]) -> list[dict[str, Any]]:
    return list(detail.get("reranked_candidates") or detail.get("candidates") or [])


def build_ordinal_proposals(
    records: dict[str, dict[str, Any]],
    details: dict[str, dict[str, Any]],
    nms_iou: float = 0.60,
    group_area_ratio: float = 2.5,
) -> dict[str, Any]:
    families: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for query_id, sample in records.items():
        parsed = parse_query(sample.get("query", ""), query_id)
        ordinal = parsed["ordinal"]
        target = canonical_target(sample.get("query", ""))
        if ordinal["enabled"] and target:
            key = (scene_id(sample, query_id), target, ordinal["direction"])
            families[key].append({"query_id": query_id, "sample": sample, "parsed": parsed})

    reports = []
    proposals = []
    summary = defaultdict(int)
    for (sid, target, direction), members in sorted(families.items()):
        distinct_indices = {item["parsed"]["ordinal"]["index"] for item in members}
        if len(members) < 2 or len(distinct_indices) < 2:
            summary["singleton_families"] += 1
            continue
        summary["joint_families"] += 1
        raw_pool = []
        for member in members:
            qid = member["query_id"]
            for candidate in _detail_candidates(details.get(qid, {})):
                enriched = dict(candidate)
                enriched["source_query_id"] = qid
                raw_pool.append(enriched)
        clean, cleanup = clean_candidates(raw_pool, nms_iou, group_area_ratio)
        reverse = direction == "right_to_left"
        ordered = sorted(clean, key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2, reverse=reverse)
        required = max(item["parsed"]["ordinal"]["index"] for item in members)
        selected_areas = [_area(item["bbox"]) for item in ordered[:required]]
        area_ratio = (
            max(selected_areas) / max(min(selected_areas), 1e-12)
            if len(selected_areas) >= required else None
        )
        # Repeated instances may vary with perspective, but extreme scale
        # dispersion usually means the pool mixes a group box/background object
        # with individual instances.  Such families must go to tiled recovery.
        area_consistent = area_ratio is not None and area_ratio <= 8.0
        enough = len(ordered) >= required and area_consistent
        report = {
            "scene_id": sid,
            "target": target,
            "direction": direction,
            "query_ids": [item["query_id"] for item in members],
            "required_instances": required,
            "cleanup": cleanup,
            "candidate_count": len(ordered),
            "selected_area_ratio": area_ratio,
            "area_consistent": area_consistent,
            "needs_tiled_recovery": not enough,
            "proposals": [],
        }
        if not enough:
            summary["needs_tiled_recovery"] += 1
        else:
            summary["assignable_families"] += 1
            for member in members:
                index = member["parsed"]["ordinal"]["index"]
                selected = ordered[index - 1]
                before = member["sample"].get("bbox")
                changed = not before or compute_iou(before, selected["bbox"]) < 0.85
                proposal = {
                    "query_id": member["query_id"],
                    "query": member["sample"].get("query", ""),
                    "v1_bbox": before,
                    "v2_bbox": selected["bbox"],
                    "ordinal_index": index,
                    "route": "ordinal_joint",
                    "changed": changed,
                    "override": False,
                    "confidence_state": "review_required",
                    "reason": "Shared candidate pool has enough cleaned instances; proposal only.",
                }
                report["proposals"].append(proposal)
                if changed:
                    proposals.append(proposal)
                    summary["changed_proposals"] += 1
        reports.append(report)
    return {"summary": dict(summary), "families": reports, "change_proposals": proposals}
