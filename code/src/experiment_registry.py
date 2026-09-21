"""Append-only experiment registry for comparable offline evaluations."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_FIELDS = [
    "timestamp_utc", "experiment", "split", "run_fingerprint",
    "dataset_fingerprint", "samples", "acc_at_05", "mean_top1_iou",
    "oracle_at_5", "oracle_at_20", "candidate_coverage",
    "invalid_candidate_rate", "threshold", "candidates_path", "metrics_path",
]


def run_fingerprint(dataset_fingerprint, candidates_path, split, threshold, top_ks):
    path = Path(candidates_path)
    candidate_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps({
        "dataset": dataset_fingerprint,
        "candidates": candidate_digest,
        "split": split,
        "threshold": threshold,
        "top_ks": list(top_ks),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_registry(registry_path, row):
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in REGISTRY_FIELDS})


def build_registry_row(experiment, split, fingerprint, dataset_fp, result,
                       candidates_path, metrics_path):
    overall = result["overall"]
    oracle = overall["oracle_acc"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "split": split,
        "run_fingerprint": fingerprint,
        "dataset_fingerprint": dataset_fp,
        "samples": result["labeled_samples"],
        "acc_at_05": overall["acc_at_05"],
        "mean_top1_iou": overall["mean_top1_iou"],
        "oracle_at_5": oracle.get("5", ""),
        "oracle_at_20": oracle.get("20", ""),
        "candidate_coverage": overall["candidate_coverage"],
        "invalid_candidate_rate": overall["invalid_candidate_rate"],
        "threshold": result["threshold"],
        "candidates_path": str(Path(candidates_path).resolve()),
        "metrics_path": str(Path(metrics_path).resolve()),
    }
