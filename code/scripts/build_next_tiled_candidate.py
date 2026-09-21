"""Build a small, manually selected tiled-rescue candidate from a parent submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SELECTED_IDS = [
    "003383_002",
    "004489_001",
    "001311_003",
    "001017_009",
    "004006_003",
    "001066_005",
    "001269_003",
    "011944_010",
    "000380_001",
    "009921_001",
]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path: str):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--eligible", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-ids-json")
    args = parser.parse_args()

    parent = load(args.parent)
    eligible = load(args.eligible)
    selected_ids = DEFAULT_SELECTED_IDS
    if args.selected_ids_json:
        selected_ids = load(args.selected_ids_json)
    eligible_by_id = {item["sample_id"]: item for item in eligible}
    missing = [sample_id for sample_id in selected_ids if sample_id not in eligible_by_id]
    if missing:
        raise ValueError(f"Selected ids missing from eligible plan: {missing}")

    predictions = json.loads(json.dumps(parent, ensure_ascii=False))
    changes = []
    for sample_id in selected_ids:
        item = eligible_by_id[sample_id]
        before = predictions[sample_id]["bbox"]
        after = item["after"]
        predictions[sample_id]["bbox"] = after
        changes.append(
            {
                "sample_id": sample_id,
                "query": item["query"],
                "before": before,
                "after": after,
                "candidate_iou": item["candidate_iou"],
                "candidate_score": item["candidate_score"],
                "candidate_rank": item["candidate_rank"],
                "area_ratio": item["area_ratio"],
                "stage1_agreement": item["stage1_agreement"],
                "stage1_votes": item["stage1_votes"],
                "pairwise_checks": item["pairwise_checks"],
            }
        )

    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "prediction.json"
    zip_path = output / f"{output.name}.zip"
    plan_path = output / "accepted_change_plan.json"
    metrics_path = output / "metrics.json"

    prediction_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, arcname="prediction.json")
    plan_path.write_text(
        json.dumps(changes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics = {
        "parent": args.parent,
        "selected_count": len(changes),
        "selection_mode": "manual_visual_audit",
        "submission_created": True,
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
