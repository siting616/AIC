"""Canonical candidate-cache construction and validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .metrics import is_valid_bbox


def extract_candidates(record):
    if not isinstance(record, dict):
        return []
    source = record.get("candidates") or record.get("reranked_candidates") or []
    if not source and record.get("bbox") is not None:
        source = [{"bbox": record["bbox"], "score": record.get("score", 0.0)}]
    candidates = []
    for index, item in enumerate(source):
        if not isinstance(item, dict) or not is_valid_bbox(item.get("bbox")):
            continue
        candidates.append({
            "bbox": [float(value) for value in item["bbox"]],
            "score": float(item.get("score", 0.0) or 0.0),
            "label": str(item.get("label", "")),
            "source_rank": int(item.get("rank", index + 1) or index + 1),
            "source": str(item.get("source", "detector")),
        })
    return candidates


def build_candidate_cache(annotations, source_records, source_name="unknown"):
    records = {}
    missing = 0
    total_candidates = 0
    for sample_id, annotation in annotations.items():
        candidates = extract_candidates(source_records.get(sample_id))
        if not candidates:
            missing += 1
        total_candidates += len(candidates)
        records[sample_id] = {
            "query": annotation.get("query", ""),
            "image_group": str(annotation.get("source_image_id", annotation.get("visible", sample_id))),
            "candidates": candidates,
        }
    fingerprint_payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "candidate_fingerprint": hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest(),
        "stats": {
            "samples": len(records),
            "samples_with_candidates": len(records) - missing,
            "missing_samples": missing,
            "total_candidates": total_candidates,
            "mean_candidates": total_candidates / len(records) if records else 0.0,
        },
        "records": records,
    }


def validate_candidate_cache(cache, expected_ids=None):
    if cache.get("schema_version") != 1 or not isinstance(cache.get("records"), dict):
        raise ValueError("unsupported candidate cache schema")
    if expected_ids is not None and set(cache["records"]) != set(expected_ids):
        raise ValueError("candidate cache IDs do not exactly match annotations")
    for sample_id, record in cache["records"].items():
        if not isinstance(record.get("candidates"), list):
            raise ValueError(f"invalid candidates list for {sample_id}")
        if any(not is_valid_bbox(item.get("bbox")) for item in record["candidates"]):
            raise ValueError(f"invalid bbox in cache for {sample_id}")
    return True
