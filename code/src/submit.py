"""Submission generation utilities."""

from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .postprocess import sanitize_bbox
from .metrics import is_valid_bbox


def save_predictions(predictions: Dict[str, Iterable[float]], output_path: str) -> str:
    """Save raw prediction boxes keyed by sample id."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    return str(path)


def generate_submission(
    original_json_path: str,
    predictions: Dict[str, Iterable[float]],
    output_json_path: str,
    output_zip_path: str,
) -> Tuple[str, str]:
    """Write predicted bboxes into a copy of the original JSON and zip it."""
    original_path = Path(original_json_path)
    if not original_path.exists():
        raise FileNotFoundError(f"Original JSON not found: {original_path}")

    with original_path.open("r", encoding="utf-8") as f:
        original = json.load(f)

    submission = deepcopy(original)
    missing_ids = [sample_id for sample_id in submission if sample_id not in predictions]
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"Predictions are incomplete: missing {len(missing_ids)} query IDs "
            f"(first: {preview})"
        )

    for sample_id, item in submission.items():
        pred_bbox = predictions[sample_id]
        item["bbox"] = sanitize_bbox(pred_bbox)
        if not is_valid_bbox(item["bbox"]):
            raise ValueError(f"Invalid bbox after sanitization: {sample_id}")

    output_json = Path(output_json_path)
    output_zip = Path(output_zip_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_json, arcname=output_json.name)

    return str(output_json), str(output_zip)
