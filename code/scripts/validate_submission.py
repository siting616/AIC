"""Validate that a generated submission preserves the official query JSON."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import is_valid_bbox


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an AIC submission JSON/zip.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--zip", dest="zip_path")
    return parser.parse_args()


def resolve(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    original_path = resolve(args.original)
    submission_path = resolve(args.submission)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    submission = json.loads(submission_path.read_text(encoding="utf-8"))

    if list(original.keys()) != list(submission.keys()):
        raise ValueError("Query IDs or their order changed.")

    invalid = []
    changed = []
    for query_id, original_item in original.items():
        submitted_item = submission[query_id]
        expected_keys = list(original_item.keys()) + (
            [] if "bbox" in original_item else ["bbox"]
        )
        if list(submitted_item.keys()) != expected_keys:
            changed.append((query_id, "field names/order"))
            continue
        for key, value in original_item.items():
            if key != "bbox" and submitted_item.get(key) != value:
                changed.append((query_id, key))
        if not is_valid_bbox(submitted_item.get("bbox")):
            invalid.append(query_id)

    if changed:
        raise ValueError(f"Non-bbox content changed: {changed[:10]}")
    if invalid:
        raise ValueError(f"Invalid bboxes: {invalid[:10]}")

    if args.zip_path:
        zip_path = resolve(args.zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            if names != [submission_path.name]:
                raise ValueError(f"Unexpected zip contents: {names}")
            if archive.read(names[0]) != submission_path.read_bytes():
                raise ValueError("JSON inside zip differs from submission JSON.")

    print(f"VALIDATION PASSED: {len(submission)} queries")


if __name__ == "__main__":
    main()
