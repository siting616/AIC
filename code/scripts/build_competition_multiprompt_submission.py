"""Merge competition multi-prompt candidates and build frozen-V7 submissions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_cache import build_candidate_cache
from src.candidate_merge import merge_tiled_records
from src.lightweight_reranker import rerank_records, score_candidates
from src.metrics import compute_iou
from src.submit import generate_submission


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def predictions_from_control(control):
    result = {}
    for sample_id, value in control.items():
        result[sample_id] = value.get("bbox") if isinstance(value, dict) else value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--base-details", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--multiprompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dedup-iou", type=float, default=0.85)
    args = parser.parse_args()

    annotations = load(args.annotations)
    base_details = load(args.base_details)
    control = predictions_from_control(load(args.control))
    multiprompt = load(args.multiprompt)
    model = load(args.model)
    if set(annotations) != set(control):
        raise ValueError("control IDs do not exactly match competition annotations")

    # Force the actual control box into every pool. This makes comparisons and
    # margin gating valid even if the historical detail file is incomplete.
    source = {}
    for sample_id in annotations:
        record = dict(base_details.get(sample_id, {}))
        old = list(record.get("reranked_candidates") or record.get("candidates") or [])
        record["candidates"] = [{
            "bbox": control[sample_id], "score": old[0].get("score", 0.0) if old else 0.0,
            "rank": 1, "label": old[0].get("label", "") if old else "",
            "source": "control",
        }] + old
        record.pop("reranked_candidates", None)
        source[sample_id] = record

    base_cache = build_candidate_cache(annotations, source, "competition_control_and_history")
    merged = merge_tiled_records(
        base_cache, multiprompt, max_candidates=args.top_k,
        dedup_iou=args.dedup_iou, source_name="multiprompt",
        candidate_field="candidates",
    )
    full_records = rerank_records(annotations, merged["records"], model)

    full_predictions = {}
    gated_predictions = {}
    changed_full = changed_gated = 0
    margins = []
    for sample_id, annotation in annotations.items():
        candidates = merged["records"][sample_id]["candidates"]
        scores = score_candidates(annotation.get("query", ""), candidates, model)
        control_index = min(
            range(len(candidates)),
            key=lambda index: 1.0 - compute_iou(candidates[index]["bbox"], control[sample_id]),
        )
        best_bbox = full_records[sample_id]["bbox"] or control[sample_id]
        best_score = max(scores) if scores else float("-inf")
        control_score = scores[control_index] if scores else float("-inf")
        margin = best_score - control_score
        margins.append(margin)
        full_predictions[sample_id] = best_bbox
        gated_predictions[sample_id] = best_bbox if margin >= args.margin else control[sample_id]
        changed_full += compute_iou(best_bbox, control[sample_id]) < 0.999
        changed_gated += compute_iou(gated_predictions[sample_id], control[sample_id]) < 0.999

    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "merged_candidate_cache.json").write_text(
        json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    (output / "reranked_details.json").write_text(
        json.dumps(full_records, ensure_ascii=False), encoding="utf-8")
    (output / "full_predictions.json").write_text(
        json.dumps(full_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "margin_005_predictions.json").write_text(
        json.dumps(gated_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_submission(args.annotations, full_predictions,
                        output / "prediction_full.json", output / "submission_full.zip")
    generate_submission(args.annotations, gated_predictions,
                        output / "prediction_margin_005.json", output / "submission_margin_005.zip")
    report = {
        "samples": len(annotations),
        "multiprompt_records": len(multiprompt),
        "mean_candidates": merged["stats"]["mean_candidates"],
        "control_changed_full": changed_full,
        "control_changed_margin": changed_gated,
        "margin_threshold": args.margin,
        "positive_margin": sum(value > 0 for value in margins),
        "margin_at_least_threshold": sum(value >= args.margin for value in margins),
        "control": str(resolve(args.control)),
        "model": str(resolve(args.model)),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
