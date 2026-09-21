"""Verify the frozen best competition submission against its manifest."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "outputs/leaderboard/best_baseline.json"


def fail(message):
    raise SystemExit(f"BEST BASELINE INVALID: {message}")


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    submission = PROJECT_ROOT / manifest["submission_zip"]
    if not submission.is_file():
        fail(f"missing file: {submission}")
    zip_hash = hashlib.sha256(submission.read_bytes()).hexdigest()
    if zip_hash != manifest["zip_sha256"]:
        fail(f"ZIP SHA256 mismatch: {zip_hash}")
    with zipfile.ZipFile(submission) as archive:
        if archive.namelist() != [manifest["zip_member"]]:
            fail(f"unexpected ZIP members: {archive.namelist()}")
        predictions = json.loads(archive.read(manifest["zip_member"]).decode("utf-8"))
    if len(predictions) != manifest["sample_count"]:
        fail(f"expected {manifest['sample_count']} samples, found {len(predictions)}")
    required = {"visible", "infrared", "depth", "query", "bbox"}
    for sample_id, record in predictions.items():
        if not required.issubset(record):
            fail(f"missing fields for {sample_id}")
        box = record["bbox"]
        if not (isinstance(box, list) and len(box) == 4 and
                0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            fail(f"invalid bbox for {sample_id}: {box}")
    canonical = json.dumps(
        predictions, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    content_hash = hashlib.sha256(canonical).hexdigest()
    if content_hash != manifest["canonical_prediction_sha256"]:
        fail(f"prediction content SHA256 mismatch: {content_hash}")
    print(f"OK: {manifest['version']} score={manifest['leaderboard_score']:.4f}")
    print(f"samples={len(predictions)}")
    print(f"zip_sha256={zip_hash}")


if __name__ == "__main__":
    main()
