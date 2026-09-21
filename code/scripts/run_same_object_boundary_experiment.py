"""Validate conservative same-object boundary fusion, then plan V28 changes."""

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
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def fuse(base, candidate, alpha):
    return [(1 - alpha) * x + alpha * y for x, y in zip(base, candidate)]


def combined_raw(*sources):
    result = {}
    for source in sources:
        result.update(source)
    return result


def choose(base, record, config):
    eligible = []
    for index, item in enumerate(record.get("candidates", [])):
        box = item["bbox"]
        overlap = compute_iou(base, box)
        ratio = area(box) / max(area(base), 1e-9)
        agreement = int(item.get("agreement_count", 1))
        if (config["minimum_iou"] <= overlap < 0.999 and
                config["minimum_area_ratio"] <= ratio <= config["maximum_area_ratio"] and
                agreement >= config["minimum_agreement"]):
            eligible.append((
                agreement, float(item.get("fusion_score", item.get("score", 0.0))),
                overlap, -index, item,
            ))
    if not eligible:
        return None
    return max(eligible)[-1]


def evaluate(ids, annotations, base_records, raw_records, config):
    before_hits = after_hits = changed = improved = degraded = 0
    before_iou_sum = after_iou_sum = 0.0
    for sample_id in ids:
        truth = annotations[sample_id]["bbox"]
        base = base_records[sample_id]["bbox"]
        selected = choose(base, raw_records.get(sample_id, {}), config)
        after = fuse(base, selected["bbox"], config["alpha"]) if selected else base
        old_iou, new_iou = compute_iou(base, truth), compute_iou(after, truth)
        before_iou_sum += old_iou; after_iou_sum += new_iou
        before_hits += old_iou >= 0.5; after_hits += new_iou >= 0.5
        changed += selected is not None
        improved += new_iou > old_iou + 1e-9; degraded += new_iou < old_iou - 1e-9
    count = len(ids)
    return {
        "samples": count, "changed": changed,
        "acc_before": before_hits / count, "acc_after": after_hits / count,
        "acc_delta_pp": (after_hits - before_hits) * 100 / count,
        "mean_iou_before": before_iou_sum / count, "mean_iou_after": after_iou_sum / count,
        "mean_iou_delta": (after_iou_sum - before_iou_sum) / count,
        "improved": improved, "degraded": degraded,
        "new_correct": max(0, after_hits - before_hits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/validation/refcoco_val.json")
    parser.add_argument("--split", default="data/validation/splits/refcoco_v1/manifest.json")
    parser.add_argument("--base", default="outputs/experiments/lightweight_reranker_v7_multiprompt_full_final/reranked_candidates.json")
    parser.add_argument("--dev-multiprompt", default="outputs/experiments/query_decomposition_rescue_v1/run/candidate_predictions.json")
    parser.add_argument("--holdout-multiprompt", default="outputs/experiments/query_decomposition_rescue_v1/holdout_run/candidate_predictions.json")
    parser.add_argument("--competition", default="outputs/experiments/competition_v7_multiprompt/run/candidate_predictions.json")
    parser.add_argument("--v28", default="outputs/leaderboard/best_v28_02977/prediction_v28_02977.json")
    parser.add_argument("--output-dir", default="outputs/experiments/v28_same_object_boundary_v1")
    args = parser.parse_args()

    annotations, split, base = load(args.annotations), load(args.split), load(args.base)
    raw = combined_raw(load(args.dev_multiprompt), load(args.holdout_multiprompt))
    grid = []
    for minimum_iou in (0.70, 0.80, 0.90):
        for minimum_agreement in (2, 3):
            for alpha in (0.25, 0.50, 0.75, 1.0):
                config = {"minimum_iou": minimum_iou, "minimum_area_ratio": 0.67,
                          "maximum_area_ratio": 1.50, "minimum_agreement": minimum_agreement,
                          "alpha": alpha}
                metrics = evaluate(split["splits"]["dev"], annotations, base, raw, config)
                grid.append({"config": config, "dev": metrics})
    # Dev is the only tuning split. Prefer ACC, then mean IoU, then fewer changes.
    selected = max(grid, key=lambda item: (
        item["dev"]["acc_delta_pp"], item["dev"]["mean_iou_delta"], -item["dev"]["changed"]
    ))
    selected["holdout"] = evaluate(
        split["splits"]["holdout"], annotations, base, raw, selected["config"])
    dev, holdout = selected["dev"], selected["holdout"]
    passed = all((dev["acc_delta_pp"] > 0, dev["mean_iou_delta"] > 0,
                  holdout["acc_delta_pp"] > 0, holdout["mean_iou_delta"] > 0))

    output = PROJECT_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "dev_grid.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")
    competition_plan = {}
    if passed:
        v28, competition = load(args.v28), load(args.competition)
        proposals = []
        for sample_id, record in v28.items():
            base_box = record["bbox"]
            candidate = choose(base_box, competition.get(sample_id, {}), selected["config"])
            if not candidate:
                continue
            new_box = fuse(base_box, candidate["bbox"], selected["config"]["alpha"])
            confidence = (int(candidate.get("agreement_count", 1)),
                          float(candidate.get("fusion_score", candidate.get("score", 0.0))),
                          compute_iou(base_box, candidate["bbox"]))
            proposals.append((confidence, sample_id, candidate, new_box))
        proposals.sort(reverse=True, key=lambda item: item[0])
        for confidence, sample_id, candidate, new_box in proposals[:50]:
            competition_plan[sample_id] = {
                "query": v28[sample_id]["query"], "before": v28[sample_id]["bbox"],
                "after": new_box, "candidate_bbox": candidate["bbox"],
                "candidate_iou": compute_iou(v28[sample_id]["bbox"], candidate["bbox"]),
                "area_ratio": area(candidate["bbox"]) / area(v28[sample_id]["bbox"]),
                "agreement_count": candidate.get("agreement_count", 1),
                "prompt_types": candidate.get("prompt_types", []),
                "fusion_score": candidate.get("fusion_score", candidate.get("score", 0.0)),
            }
    result = {"status": "PASS" if passed else "STOP", "selected": selected,
              "competition_changes": len(competition_plan), "submission_created": False}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "competition_change_plan.json").write_text(
        json.dumps(competition_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# V28 same-object boundary experiment", "", f"- Gate: **{result['status']}**",
              f"- Selected config: `{selected['config']}`", "",
              "## Dev", "", f"- ACC delta: {dev['acc_delta_pp']:+.4f} pp",
              f"- Mean IoU delta: {dev['mean_iou_delta']:+.6f}",
              f"- Changed / improved / degraded: {dev['changed']} / {dev['improved']} / {dev['degraded']}", "",
              "## Holdout", "", f"- ACC delta: {holdout['acc_delta_pp']:+.4f} pp",
              f"- Mean IoU delta: {holdout['mean_iou_delta']:+.6f}",
              f"- Changed / improved / degraded: {holdout['changed']} / {holdout['improved']} / {holdout['degraded']}", "",
              f"- Competition plan changes: {len(competition_plan)}", "- Submission created: no"]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
