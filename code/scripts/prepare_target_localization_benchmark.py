"""Create a frozen benchmark for candidate target-selection accuracy."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_iou


def load(path):
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def stable_sample(items, count, salt):
    return sorted(items, key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest())[:count]


def main():
    annotations = load("data/validation/refcoco_val.json")
    splits = load("data/validation/splits/refcoco_v1/manifest.json")["splits"]
    cache = load("outputs/candidate_cache/refcoco_diverse_tiled_multiprompt_full_top20_v5.json")["records"]
    baseline = load("outputs/experiments/lightweight_reranker_v7_multiprompt_full_final/reranked_candidates.json")
    output = PROJECT_ROOT / "outputs/benchmarks/target_localization_v1"
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split_name, ids in splits.items():
        covered, correct, wrong = [], [], []
        labels = {}
        for sample_id in ids:
            truth = annotations[sample_id]["bbox"]
            candidates = cache[sample_id]["candidates"]
            ious = [compute_iou(item["bbox"], truth) for item in candidates]
            valid = [index for index, value in enumerate(ious) if value >= 0.5]
            if not valid:
                continue
            covered.append(sample_id)
            baseline_hit = compute_iou(baseline[sample_id]["bbox"], truth) >= 0.5
            (correct if baseline_hit else wrong).append(sample_id)
            labels[sample_id] = {
                "correct_candidate_indices": valid,
                "best_candidate_index": max(range(len(ious)), key=ious.__getitem__),
                "best_candidate_iou": max(ious),
                "baseline_correct": baseline_hit,
            }
        # Balanced pilot measures both rescue and preservation. Labels are kept
        # separate from the GPU task IDs to prevent accidental prompt leakage.
        per_class = 250 if split_name == "dev" else 150
        pilot = stable_sample(correct, per_class, f"{split_name}:correct") + stable_sample(
            wrong, per_class, f"{split_name}:wrong")
        pilot = stable_sample(pilot, len(pilot), f"{split_name}:shuffle")
        (output / f"{split_name}_pilot_ids.json").write_text(
            json.dumps(pilot, indent=2), encoding="utf-8")
        (output / f"{split_name}_labels.json").write_text(
            json.dumps({key: labels[key] for key in pilot}, indent=2), encoding="utf-8")
        summary[split_name] = {
            "samples": len(ids), "oracle_covered": len(covered),
            "baseline_correct_within_oracle": len(correct),
            "baseline_wrong_but_rescuable": len(wrong),
            "baseline_selection_accuracy_given_oracle": len(correct) / len(covered),
            "pilot_samples": len(pilot), "pilot_correct_controls": per_class,
            "pilot_rescuable_errors": per_class,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = ["# Target localization benchmark", ""]
    for name, values in summary.items():
        report += [f"## {name}", "", f"- Oracle-covered: {values['oracle_covered']}",
                   f"- Baseline correct: {values['baseline_correct_within_oracle']}",
                   f"- Rescuable selection errors: {values['baseline_wrong_but_rescuable']}",
                   f"- Selection accuracy given oracle: {values['baseline_selection_accuracy_given_oracle']:.2%}",
                   f"- Balanced pilot: {values['pilot_samples']}", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
