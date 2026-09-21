"""Build a conservative, non-submission competition change plan."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.metrics import compute_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="outputs/experiments/joint_assignment_v31/competition_groups.json")
    parser.add_argument("--selected", default="outputs/experiments/tiled_competition/reranked/selected_candidates.json")
    parser.add_argument("--records", default="outputs/experiments/tiled_competition/run/tiled_candidate_records.json")
    parser.add_argument("--v28", default="outputs/experiments/joint_assignment_v31/v28/prediction.json")
    parser.add_argument("--output-dir", default="outputs/experiments/v31_change_plan")
    args = parser.parse_args()
    resolve = lambda x: Path(x) if Path(x).is_absolute() else PROJECT_ROOT / x
    load = lambda x: json.loads(resolve(x).read_text(encoding="utf-8"))
    groups, selected, records, v28 = load(args.groups), load(args.selected), load(args.records), load(args.v28)
    accepted_groups, changes = [], {}
    for group in groups:
        if not group.get("assignable") or not group.get("conflicts"):
            continue
        ids = group["query_ids"]
        if not all(qid in selected and selected[qid]["reason"] != "model_score" for qid in ids):
            continue
        boxes = [selected[qid]["bbox"] for qid in ids]
        if any(compute_iou(boxes[i], boxes[j]) >= .85 for i in range(len(boxes)) for j in range(i+1, len(boxes))):
            continue
        changed = []
        for qid in ids:
            before = records[qid]["baseline_bbox"]
            after = selected[qid]["bbox"]
            overlap = compute_iou(before, after)
            if overlap >= .85:
                continue
            changes[qid] = {
                "query": selected[qid]["query"], "before": before, "after": after,
                "iou_vs_v28": overlap, "reason": selected[qid]["reason"],
                "candidate_score": selected[qid].get("score", 0),
                "image_id": group["image_id"], "object_class": group["object_class"],
            }
            changed.append(qid)
        if changed:
            accepted_groups.append({
                "image_id": group["image_id"], "object_class": group["object_class"],
                "query_ids": ids, "changed_query_ids": changed,
                "original_conflicts": group["conflicts"],
            })
    diagnostic = json.loads(json.dumps(v28))
    for qid, change in changes.items():
        diagnostic[qid]["bbox"] = change["after"]
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("changes.json", changes), ("accepted_groups.json", accepted_groups),
                        ("prediction_diagnostic_only.json", diagnostic)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {
        "accepted_groups": len(accepted_groups), "changed_queries": len(changes),
        "reasons": dict(Counter(x["reason"] for x in changes.values())),
        "submission_created": False,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Diagnostic only; no submission archive was created.")


if __name__ == "__main__":
    main()
