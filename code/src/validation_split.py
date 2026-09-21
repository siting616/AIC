"""Deterministic, leakage-safe validation dataset splitting."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def image_group_key(sample_id, annotation):
    """Return a stable image identity, preferring dataset-native identifiers."""
    if isinstance(annotation, dict):
        if annotation.get("source_image_id") is not None:
            dataset = annotation.get("source_dataset", "dataset")
            return f"{dataset}:{annotation['source_image_id']}"
        if annotation.get("visible"):
            return str(annotation["visible"]).replace("\\", "/").casefold()
    # Last-resort fallback keeps the function usable for small custom fixtures.
    return f"sample:{sample_id}"


def dataset_fingerprint(annotations):
    """Fingerprint split-relevant annotation content without machine-local paths."""
    canonical = [
        {
            "sample_id": sample_id,
            "image_group": image_group_key(sample_id, annotation),
            "query": annotation.get("query", "") if isinstance(annotation, dict) else "",
            "bbox": annotation.get("bbox") if isinstance(annotation, dict) else None,
        }
        for sample_id, annotation in sorted(annotations.items())
    ]
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_grouped_split(annotations, holdout_fraction=0.30, seed=20260814):
    """Split annotations by image group with deterministic seeded ordering."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError("annotations must be a non-empty JSON object")

    groups = defaultdict(list)
    for sample_id, annotation in annotations.items():
        groups[image_group_key(sample_id, annotation)].append(sample_id)
    if len(groups) < 2:
        raise ValueError("at least two image groups are required for a split")

    ordered_groups = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    target = round(len(annotations) * holdout_fraction)
    holdout_groups = set()
    holdout_count = 0
    for group in ordered_groups:
        size = len(groups[group])
        if holdout_count < target or not holdout_groups:
            holdout_groups.add(group)
            holdout_count += size
        else:
            break

    holdout_ids = sorted(
        sample_id for group in holdout_groups for sample_id in groups[group]
    )
    holdout_set = set(holdout_ids)
    dev_ids = sorted(sample_id for sample_id in annotations if sample_id not in holdout_set)
    if not dev_ids or not holdout_ids:
        raise ValueError("split produced an empty partition")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "grouped_deterministic_sha256",
        "group_key": "source_dataset+source_image_id, fallback visible path",
        "seed": int(seed),
        "requested_holdout_fraction": float(holdout_fraction),
        "dataset_fingerprint": dataset_fingerprint(annotations),
        "counts": {
            "samples": len(annotations),
            "image_groups": len(groups),
            "dev_samples": len(dev_ids),
            "holdout_samples": len(holdout_ids),
            "dev_image_groups": len(groups) - len(holdout_groups),
            "holdout_image_groups": len(holdout_groups),
        },
        "splits": {"dev": dev_ids, "holdout": holdout_ids},
    }


def validate_split_manifest(annotations, manifest):
    """Reject stale, incomplete, overlapping, or image-leaking manifests."""
    if manifest.get("dataset_fingerprint") != dataset_fingerprint(annotations):
        raise ValueError("split manifest does not match the annotation dataset fingerprint")
    dev = set(manifest.get("splits", {}).get("dev", []))
    holdout = set(manifest.get("splits", {}).get("holdout", []))
    expected = set(annotations)
    if dev & holdout:
        raise ValueError("dev and holdout sample IDs overlap")
    if dev | holdout != expected:
        raise ValueError("split manifest does not cover exactly all annotation sample IDs")
    dev_groups = {image_group_key(i, annotations[i]) for i in dev}
    holdout_groups = {image_group_key(i, annotations[i]) for i in holdout}
    if dev_groups & holdout_groups:
        raise ValueError("image-group leakage detected between dev and holdout")
    return True


def subset_annotations(annotations, manifest, split):
    if split not in ("dev", "holdout"):
        raise ValueError("split must be 'dev' or 'holdout'")
    validate_split_manifest(annotations, manifest)
    return {sample_id: annotations[sample_id] for sample_id in manifest["splits"][split]}


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
