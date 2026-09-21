"""Merge candidate sources while preserving diversity and a fixed budget."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .metrics import compute_iou, is_valid_bbox


def normalize_extra_candidate(item, index, source_name):
    if not isinstance(item, dict) or not is_valid_bbox(item.get("bbox")):
        return None
    return {
        "bbox": [float(value) for value in item["bbox"]],
        "score": float(item.get("score", 0.0) or 0.0),
        "label": str(item.get("label", "")),
        "source_rank": int(item.get("rank", index + 1) or index + 1),
        "source": str(item.get("source", source_name)),
        "source_origin": source_name,
    }


def merge_candidate_lists(base, extras, max_candidates=20, dedup_iou=0.85):
    """Keep base recall, then add score-sorted non-duplicate extra candidates."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    merged = []
    # Base candidates retain their original rank priority, but are also
    # deduplicated. Otherwise several detector variants of the same object can
    # consume most of the fixed candidate budget before tiled sources enter.
    for item in base:
        if len(merged) >= max_candidates:
            break
        if all(compute_iou(item["bbox"], kept["bbox"]) < dedup_iou for kept in merged):
            merged.append(dict(item))
    ordered_extras = sorted(
        enumerate(extras), key=lambda pair: (-float(pair[1].get("score", 0.0)), pair[0])
    )
    for _, item in ordered_extras:
        if len(merged) >= max_candidates:
            break
        if all(compute_iou(item["bbox"], kept["bbox"]) < dedup_iou for kept in merged):
            merged.append(dict(item))
    return merged


def merge_tiled_records(base_cache, tiled_records, max_candidates=20, dedup_iou=0.85,
                         source_name="tiled", candidate_field="new_candidates"):
    records = {}
    stats = {
        "samples": len(base_cache["records"]), "samples_with_tiled_records": 0,
        "samples_with_added_candidates": 0, "base_candidates": 0,
        "valid_extra_candidates": 0, "added_candidates": 0,
        "max_candidates": max_candidates, "dedup_iou": dedup_iou,
    }
    for sample_id, base_record in base_cache["records"].items():
        base = base_record.get("candidates", [])
        stats["base_candidates"] += len(base)
        source_record = tiled_records.get(sample_id, {})
        if source_record:
            stats["samples_with_tiled_records"] += 1
        extras = []
        for index, item in enumerate(source_record.get(candidate_field, [])):
            normalized = normalize_extra_candidate(item, index, source_name)
            if normalized:
                extras.append(normalized)
        stats["valid_extra_candidates"] += len(extras)
        merged = merge_candidate_lists(base, extras, max_candidates, dedup_iou)
        base_diverse_count = len(merge_candidate_lists(base, [], max_candidates, dedup_iou))
        added = max(0, len(merged) - base_diverse_count)
        stats["added_candidates"] += added
        stats["samples_with_added_candidates"] += int(added > 0)
        records[sample_id] = {
            "query": base_record.get("query", ""),
            "image_group": base_record.get("image_group", sample_id),
            "candidates": merged,
        }
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": f"{base_cache.get('source_name', 'base')}+{source_name}",
        "candidate_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "merge_stats": stats,
        "stats": {
            "samples": len(records),
            "samples_with_candidates": sum(bool(record["candidates"]) for record in records.values()),
            "missing_samples": sum(not record["candidates"] for record in records.values()),
            "total_candidates": sum(len(record["candidates"]) for record in records.values()),
            "mean_candidates": sum(len(record["candidates"]) for record in records.values()) / len(records),
        },
        "records": records,
    }
