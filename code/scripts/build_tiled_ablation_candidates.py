"""Build singleton tiled-rescue submissions for leaderboard attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path: str):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def write_submission(parent: dict, item: dict, output_dir: Path) -> dict:
    sample_id = item["sample_id"]
    predictions = json.loads(json.dumps(parent, ensure_ascii=False))
    predictions[sample_id]["bbox"] = item["after"]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.json"
    zip_path = output_dir / f"{output_dir.name}.zip"
    plan_path = output_dir / "accepted_change_plan.json"
    metrics_path = output_dir / "metrics.json"
    prediction_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, arcname="prediction.json")
    plan_path.write_text(
        json.dumps([item], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "parent": "outputs/leaderboard/submission_03023_plus_expanded_tiled25/prediction.json",
        "selected_count": 1,
        "sample_id": sample_id,
        "selection_mode": "singleton_ablation",
        "submission_created": True,
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--change-plan", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    parent = load(args.parent)
    changes = load(args.change_plan)
    output_root = resolve(args.output_root)
    summaries = []
    for index, item in enumerate(changes, start=1):
        name = f"submission_03035_single_{index:02d}_{item['sample_id']}"
        summaries.append(write_submission(parent, item, output_root / name))
    summary_path = output_root / "singleton_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"created": len(summaries), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
