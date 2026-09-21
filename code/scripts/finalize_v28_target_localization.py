"""Apply a manually audited target-localization plan to frozen V28."""

from __future__ import annotations

import argparse
import json
import zipfile
from copy import deepcopy
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v28", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--zip-name", default="submission_v28_targetloc_audited45.zip")
    args = parser.parse_args()

    baseline = load(args.v28)
    plan = load(args.plan)
    decisions = load(args.decisions)
    plan_ids = {item["sample_id"] for item in plan}
    rejected = set(decisions.get("rejected_sample_ids", []))
    accepted_ids = set(decisions.get("accepted_sample_ids", []))
    referenced = accepted_ids if args.accepted_only else rejected
    unknown = referenced - plan_ids
    if unknown:
        raise ValueError(f"Decision IDs are absent from plan: {sorted(unknown)}")

    accepted = ([item for item in plan if item["sample_id"] in accepted_ids]
                if args.accepted_only else
                [item for item in plan if item["sample_id"] not in rejected])
    prediction = deepcopy(baseline)
    for item in accepted:
        prediction[item["sample_id"]]["bbox"] = item["after"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.json"
    prediction_path.write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "accepted_change_plan.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "baseline": str(Path(args.v28)),
        "proposed": len(plan),
        "selection_mode": "accepted_only" if args.accepted_only else "reject_list",
        "manually_rejected": len(rejected),
        "manually_deferred": len(plan) - len(accepted) if args.accepted_only else 0,
        "accepted_changes": len(accepted),
        "rejected_sample_ids": sorted(rejected),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    zip_path = output_dir / args.zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, arcname="prediction.json")
    print(json.dumps({**metrics, "zip": str(zip_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
