"""Build a diagnostic two-change ordinal batch from explicit visual review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v2_delta import load_frozen_baseline, load_json, resolve


ACCEPTED = {
    "016004_002": "Second zebra from right: V2 selects the next zebra left of the rightmost instance.",
    "017187_001": "First black bear from right: V2 selects the rightmost bear instead of the adjacent bear.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="outputs/leaderboard/best_baseline.json")
    parser.add_argument("--proposals", default="outputs/v2/ordinal/audit_low/ranked_proposals.json")
    parser.add_argument("--output-dir", default="outputs/v2/ordinal/reviewed_batch2")
    args = parser.parse_args()
    registry, records, _ = load_frozen_baseline(resolve(args.registry))
    proposals = {item["query_id"]: item for item in load_json(resolve(args.proposals))}
    missing = set(ACCEPTED) - set(proposals)
    if missing:
        raise SystemExit(f"Accepted IDs missing from proposal file: {sorted(missing)}")
    before = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    changes = []
    for query_id, reason in ACCEPTED.items():
        proposal = proposals[query_id]
        old = list(records[query_id]["bbox"])
        new = list(proposal["v2_bbox"])
        records[query_id]["bbox"] = new
        changes.append({
            "query_id": query_id,
            "query": records[query_id]["query"],
            "before": old,
            "after": new,
            "route": "ordinal_joint_visual_review",
            "review_decision": "accept",
            "reason": reason,
        })
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "prediction.json"
    prediction.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output / "submission_diagnostic_only.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(prediction, arcname="prediction.json")
    (output / "accepted_change_plan.json").write_text(
        json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    unchanged = 0
    baseline_records = json.loads(before)
    for query_id in records:
        if query_id not in ACCEPTED and records[query_id] == baseline_records[query_id]:
            unchanged += 1
    summary = {
        "status": "diagnostic_only_do_not_submit_without_approval",
        "parent_version": registry["version"],
        "parent_score": registry.get("leaderboard_score"),
        "sample_count": len(records),
        "changed_bbox_count": len(changes),
        "unchanged_record_count": unchanged,
        "accepted_query_ids": sorted(ACCEPTED),
        "prediction_sha256": sha256(prediction),
        "zip_sha256": sha256(archive),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
