"""Build an auditable top-50 V28 target correction plan; no submission ZIP."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_iou

LABELS = "ABCDEFGHIJKLMNOPQRST"


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v28", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--pairwise", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    v28, cache, stage1, pairwise = load(args.v28), load(args.candidates), load(args.stage1), load(args.pairwise)
    records = cache.get("records", cache)
    proposals = []
    for sample_id, decision in pairwise.items():
        if not decision.get("accept_challenger"):
            continue
        first = stage1.get(sample_id, {})
        label = first.get("majority_label")
        if not isinstance(label, str) or label not in LABELS:
            continue
        index = LABELS.index(label)
        candidates = records.get(sample_id, {}).get("candidates", [])
        if index >= len(candidates):
            continue
        candidate = candidates[index]
        old = v28[sample_id]["bbox"]
        overlap = compute_iou(old, candidate["bbox"])
        score = float(candidate.get("score", 0.0))
        area_ratio = area(candidate["bbox"]) / max(area(old), 1e-9)
        if score < 0.10 or overlap >= 0.50 or not 0.20 <= area_ratio <= 5.0:
            continue
        proposals.append({
            "sample_id": sample_id, "query": v28[sample_id]["query"],
            "before": old, "after": candidate["bbox"], "candidate_iou": overlap,
            "candidate_score": score, "candidate_rank": index + 1,
            "area_ratio": area_ratio,
            "stage1_agreement": int(first.get("agreement", 0)),
            "stage1_votes": first.get("votes", {}), "pairwise_checks": decision.get("checks", []),
        })
    proposals.sort(key=lambda x: (-x["stage1_agreement"], -x["candidate_score"], x["candidate_rank"], x["sample_id"]))
    selected = proposals[:args.limit]
    diagnostic = deepcopy(v28)
    for item in selected:
        diagnostic[item["sample_id"]]["bbox"] = item["after"]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "all_eligible.json").write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "top50_change_plan.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "prediction_diagnostic_only.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {"eligible": len(proposals), "selected": len(selected), "limit": args.limit,
               "submission_created": False}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
