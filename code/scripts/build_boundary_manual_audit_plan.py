"""Build a diagnostic competition boundary plan from a validated geometric rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.metrics import compute_iou


def load(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def area(box):
    return max(0.0, box[2]-box[0]) * max(0.0, box[3]-box[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rule-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    baseline, raw, result = load(args.baseline), load(args.candidates), load(args.rule_result)
    rule = result["selected"]["rule"]
    proposals = []
    for sample_id, base_record in baseline.items():
        base = base_record["bbox"]
        choices = []
        for rank, item in enumerate(raw.get(sample_id, {}).get("candidates", []), 1):
            box = item["bbox"]
            overlap = compute_iou(base, box)
            ratio = area(box) / max(area(base), 1e-12)
            agreement = int(item.get("agreement_count", 1))
            score = float(item.get("fusion_score", item.get("score", 0.0)))
            if (rule["min_iou"] <= overlap < rule["max_iou"] and
                    rule["min_ratio"] <= ratio <= rule["max_ratio"] and
                    agreement >= rule["min_agreement"] and score >= rule["min_score"]):
                choices.append((agreement, score, overlap, -rank, item, rank, ratio))
        if not choices:
            continue
        agreement, score, overlap, _, item, rank, ratio = max(choices)
        proposals.append({"sample_id": sample_id, "query": base_record["query"],
                          "before": base, "after": item["bbox"], "candidate_iou": overlap,
                          "area_ratio": ratio, "candidate_score": score,
                          "candidate_rank": rank, "agreement_count": agreement})
    proposals.sort(key=lambda x: (-x["agreement_count"], -x["candidate_score"],
                                  x["candidate_iou"], x["sample_id"]))
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proposals[:args.limit], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"eligible": len(proposals), "audit_batch": min(args.limit, len(proposals)),
                      "submission_created": False}, indent=2))


if __name__ == "__main__":
    main()
