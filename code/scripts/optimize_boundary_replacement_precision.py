"""Tune high-precision same-object boundary replacement rules on dev only."""

from __future__ import annotations

import argparse
import json
import math
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


def candidate_score(item):
    return float(item.get("fusion_score", item.get("score", 0.0)))


def select(base, record, rule):
    eligible = []
    for rank, item in enumerate(record.get("candidates", []), 1):
        box = item["bbox"]
        overlap = compute_iou(base, box)
        ratio = area(box) / max(area(base), 1e-12)
        agreement = int(item.get("agreement_count", 1))
        score = candidate_score(item)
        if not (rule["min_iou"] <= overlap < rule["max_iou"]):
            continue
        if not (rule["min_ratio"] <= ratio <= rule["max_ratio"]):
            continue
        if agreement < rule["min_agreement"] or score < rule["min_score"]:
            continue
        eligible.append((agreement, score, overlap, -rank, item))
    return max(eligible)[-1] if eligible else None


def evaluate(ids, truth, base_records, raw, rule):
    changed = improved = degraded = neutral = before_hits = after_hits = 0
    delta_sum = 0.0
    examples = []
    for sample_id in ids:
        if sample_id not in raw:
            continue
        base = base_records[sample_id]["bbox"]
        item = select(base, raw[sample_id], rule)
        if item is None:
            continue
        old_iou = compute_iou(base, truth[sample_id]["bbox"])
        new_iou = compute_iou(item["bbox"], truth[sample_id]["bbox"])
        delta = new_iou - old_iou
        changed += 1; delta_sum += delta
        before_hits += old_iou >= 0.5; after_hits += new_iou >= 0.5
        improved += delta > 1e-9; degraded += delta < -1e-9; neutral += abs(delta) <= 1e-9
        examples.append((delta, sample_id))
    decided = improved + degraded
    return {
        "changed": changed, "improved": improved, "degraded": degraded, "neutral": neutral,
        "precision_among_decided": improved / decided if decided else 0.0,
        "net_improved": improved - degraded,
        "mean_iou_delta_changed": delta_sum / changed if changed else 0.0,
        "hit_delta": after_hits - before_hits,
        "best_examples": [sid for _, sid in sorted(examples, reverse=True)[:10]],
        "worst_examples": [sid for _, sid in sorted(examples)[:10]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/validation/refcoco_val.json")
    parser.add_argument("--split", default="data/validation/splits/refcoco_v1/manifest.json")
    parser.add_argument("--base", default="outputs/experiments/lightweight_reranker_v7_multiprompt_full_final/reranked_candidates.json")
    parser.add_argument("--dev-candidates", default="outputs/experiments/query_decomposition_rescue_v1/run/candidate_predictions.json")
    parser.add_argument("--holdout-candidates", default="outputs/experiments/query_decomposition_rescue_v1/holdout_run/candidate_predictions.json")
    parser.add_argument("--output-dir", default="outputs/experiments/boundary_precision_v2")
    args = parser.parse_args()
    truth, split, base = load(args.annotations), load(args.split), load(args.base)
    dev_raw, holdout_raw = load(args.dev_candidates), load(args.holdout_candidates)
    dev_ids, holdout_ids = split["splits"]["dev"], split["splits"]["holdout"]

    grid = []
    for min_iou in (0.50, 0.60, 0.70, 0.80):
        for max_iou in (0.85, 0.90, 0.95, 0.999):
            if max_iou <= min_iou:
                continue
            for min_ratio, max_ratio in ((0.67, 1.0), (1.0, 1.5), (0.80, 1.25), (0.67, 1.5)):
                for min_agreement in (1, 2, 3):
                    for min_score in (0.10, 0.15, 0.20, 0.25):
                        rule = {"min_iou": min_iou, "max_iou": max_iou,
                                "min_ratio": min_ratio, "max_ratio": max_ratio,
                                "min_agreement": min_agreement, "min_score": min_score}
                        metrics = evaluate(dev_ids, truth, base, dev_raw, rule)
                        if metrics["changed"] >= 20:
                            grid.append({"rule": rule, "dev": metrics})
    # Tune only on dev: prioritize positive hit changes, precision, net gains, and support.
    viable = [x for x in grid if x["dev"]["net_improved"] > 0 and x["dev"]["hit_delta"] >= 0]
    selected = max(viable, key=lambda x: (
        x["dev"]["precision_among_decided"], x["dev"]["net_improved"],
        x["dev"]["hit_delta"], math.log1p(x["dev"]["changed"])
    )) if viable else None
    if selected:
        selected["holdout"] = evaluate(holdout_ids, truth, base, holdout_raw, selected["rule"])
    passed = bool(selected and
                  selected["dev"]["precision_among_decided"] >= 0.60 and
                  selected["holdout"]["precision_among_decided"] >= 0.60 and
                  selected["dev"]["net_improved"] > 0 and
                  selected["holdout"]["net_improved"] > 0 and
                  selected["dev"]["hit_delta"] >= 0 and selected["holdout"]["hit_delta"] >= 0)
    output = PROJECT_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "PASS" if passed else "STOP", "rules_tested": len(grid), "selected": selected,
              "competition_plan_created": False, "submission_created": False}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "dev_grid.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
