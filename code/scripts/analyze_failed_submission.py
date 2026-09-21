"""Audit a failed competition submission against a trusted control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_iou


PATTERNS = {
    "ordinal": r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b",
    "spatial": r"\b(left|right|top|bottom|upper|lower|middle|center|above|below|under)\b",
    "color": r"\b(red|blue|green|yellow|white|black|pink|purple|brown|gray|grey|orange)\b",
    "size": r"\b(small|tiny|large|big|largest|smallest|bigger|smaller)\b",
    "text_ocr": r"\b(sign|text|word|letter|number|printed|label|logo|screen)\b|['\"]",
    "thermal_depth": r"\b(thermal|infrared|heat|warm|cold|depth|near|far|closest|farthest)\b",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bbox_of(value):
    return value["bbox"] if isinstance(value, dict) else value


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[round((len(values) - 1) * fraction)]


def same_box(left, right, threshold=0.999):
    return compute_iou(left, right) >= threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--failed", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--multiprompt", required=True)
    parser.add_argument("--legacy", help="Older baseline used before the trusted control")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-score", type=float)
    parser.add_argument("--failed-score", type=float)
    args = parser.parse_args()

    control, failed = load(args.control), load(args.failed)
    details, cache, multiprompt = load(args.details), load(args.cache)["records"], load(args.multiprompt)
    legacy = load(args.legacy) if args.legacy else None
    if set(control) != set(failed):
        raise ValueError("control and failed IDs differ")

    rows = []
    type_total = Counter()
    type_changed = Counter()
    source_counts = Counter()
    prompt_counts = Counter()
    agreement_counts = Counter()
    old_new_ious, area_ratios, shifts = [], [], []
    legacy_reversions = 0
    for sample_id, old_record in control.items():
        query = old_record.get("query", "") if isinstance(old_record, dict) else ""
        query_types = [name for name, pattern in PATTERNS.items() if re.search(pattern, query, re.I)] or ["plain"]
        type_total.update(query_types)
        old_box, new_box = bbox_of(old_record), bbox_of(failed[sample_id])
        if same_box(old_box, new_box):
            continue
        type_changed.update(query_types)
        overlap = compute_iou(old_box, new_box)
        ratio = area(new_box) / max(area(old_box), 1e-9)
        old_center, new_center = center(old_box), center(new_box)
        shift = math.hypot(new_center[0] - old_center[0], new_center[1] - old_center[1])
        old_new_ious.append(overlap); area_ratios.append(ratio); shifts.append(shift)

        chosen = next((item for item in details[sample_id].get("candidates", [])
                       if same_box(item["bbox"], new_box)), {})
        origin = chosen.get("source_origin") or chosen.get("source") or "unknown"
        source_counts[origin] += 1
        raw_mp = next((item for item in multiprompt.get(sample_id, {}).get("candidates", [])
                       if same_box(item["bbox"], new_box)), None)
        if raw_mp:
            prompt_counts.update(raw_mp.get("prompt_types", [raw_mp.get("prompt_type", "unknown")]))
            agreement_counts[int(raw_mp.get("agreement_count", 1))] += 1
        legacy_reversion = False
        if legacy and sample_id in legacy:
            legacy_reversion = same_box(new_box, bbox_of(legacy[sample_id])) and not same_box(old_box, bbox_of(legacy[sample_id]))
            legacy_reversions += int(legacy_reversion)
        rows.append({
            "sample_id": sample_id, "query": query, "query_types": ",".join(query_types),
            "old_new_iou": overlap, "old_area": area(old_box), "new_area": area(new_box),
            "area_ratio": ratio, "center_shift": shift, "selected_source": origin,
            "detector_score": chosen.get("score"), "reranker_score": chosen.get("reranker_score"),
            "multiprompt_agreement": raw_mp.get("agreement_count") if raw_mp else None,
            "multiprompt_prompt_types": ",".join(raw_mp.get("prompt_types", [])) if raw_mp else "",
            "reverts_to_legacy": legacy_reversion,
            "old_bbox": json.dumps(old_box), "new_bbox": json.dumps(new_box),
        })

    changed = len(rows)
    summary = {
        "samples": len(control), "changed": changed, "changed_rate": changed / len(control),
        "control_score": args.control_score, "failed_score": args.failed_score,
        "score_delta": (args.failed_score - args.control_score)
            if args.control_score is not None and args.failed_score is not None else None,
        "equivalent_net_hits_lost_if_score_is_accuracy": round((args.control_score - args.failed_score) * len(control))
            if args.control_score is not None and args.failed_score is not None else None,
        "geometry": {
            "old_new_iou_median": quantile(old_new_ious, .5), "old_new_iou_p25": quantile(old_new_ious, .25),
            "near_disjoint_iou_below_0_1": sum(v < .1 for v in old_new_ious),
            "large_jump_iou_below_0_5": sum(v < .5 for v in old_new_ious),
            "center_shift_median": quantile(shifts, .5), "center_shift_above_0_25": sum(v > .25 for v in shifts),
            "area_ratio_median": quantile(area_ratios, .5), "area_shrink_below_half": sum(v < .5 for v in area_ratios),
            "area_expand_above_2x": sum(v > 2 for v in area_ratios),
        },
        "query_type_changed": dict(type_changed),
        "query_type_change_rates": {name: type_changed[name] / count for name, count in type_total.items()},
        "selected_source": dict(source_counts), "multiprompt_prompt_types": dict(prompt_counts),
        "multiprompt_agreement": dict(agreement_counts), "reversions_to_legacy": legacy_reversions,
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "changed_samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    report = [
        "# V28 → failed submission audit", "", f"- Samples: {len(control)}", f"- Changed: {changed} ({changed/len(control):.2%})",
        f"- Leaderboard: {args.control_score:.4f} → {args.failed_score:.4f} ({args.failed_score-args.control_score:+.4f})",
        f"- Equivalent net hits lost (if score is hit-rate): {summary['equivalent_net_hits_lost_if_score_is_accuracy']}", "",
        "## Geometry", "", f"- Median old/new IoU: {summary['geometry']['old_new_iou_median']:.3f}",
        f"- IoU < 0.5: {summary['geometry']['large_jump_iou_below_0_5']} / {changed}",
        f"- IoU < 0.1: {summary['geometry']['near_disjoint_iou_below_0_1']} / {changed}",
        f"- Area < 0.5x: {summary['geometry']['area_shrink_below_half']} / {changed}",
        f"- Area > 2x: {summary['geometry']['area_expand_above_2x']} / {changed}", "",
        "## Attribution", "", f"- Selected source: {dict(source_counts)}",
        f"- Multi-prompt agreement: {dict(agreement_counts)}",
        f"- Reverted a V28 change back to legacy V20: {legacy_reversions}", "",
        "## Interpretation", "",
        "The leaderboard proves the batch of changes failed overall, but without hidden labels it cannot identify each individual error.",
        "Geometry/source statistics are mechanism evidence, not per-sample correctness labels.",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
