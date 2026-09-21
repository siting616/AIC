"""Select unprocessed competition queries most likely to benefit from tiled detection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMALL_WORDS = re.compile(
    r"\b(small|tiny|little|distant|distance|far|background|corner|partially|hidden|"
    r"upper|lower|top|bottom|behind|near|remote)\b", re.I
)


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
    parser.add_argument("--v28", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--old-tiled-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    baseline, v28 = load(args.baseline), load(args.v28)
    cache, old_tiled = load(args.candidate_cache), load(args.old_tiled_records)
    records = cache.get("records", cache)
    excluded_tiled = set(old_tiled)
    changed = {qid for qid in baseline if baseline[qid]["bbox"] != v28[qid]["bbox"]}
    ranked = []
    for qid, item in baseline.items():
        if qid in excluded_tiled or qid in changed:
            continue
        box_area = area(item["bbox"])
        candidates = records.get(qid, {}).get("candidates", [])
        count = len(candidates)
        max_score = max((float(x.get("score", 0.0)) for x in candidates), default=0.0)
        keyword_hits = len(SMALL_WORDS.findall(item["query"]))
        score = 0.0
        reasons = []
        if box_area <= 0.005: score += 3; reasons.append("very_small_box")
        elif box_area <= 0.015: score += 2; reasons.append("small_box")
        elif box_area <= 0.035: score += 1; reasons.append("medium_small_box")
        if count == 0: score += 4; reasons.append("no_candidates")
        elif count <= 2: score += 2.5; reasons.append("candidate_count_le2")
        elif count <= 3: score += 1; reasons.append("candidate_count_le3")
        if max_score < 0.12: score += 2; reasons.append("very_low_score")
        elif max_score < 0.20: score += 1; reasons.append("low_score")
        if keyword_hits: score += min(2, keyword_hits); reasons.append("small_target_language")
        # Require at least one concrete weakness; tie-break toward smaller boxes and weaker scores.
        if score > 0:
            ranked.append((score, -box_area, -max_score, qid, reasons, count, max_score, box_area))
    ranked.sort(reverse=True)
    selected = ranked[:args.limit]
    subset, audit = [], []
    for score, _, _, qid, reasons, count, max_score, box_area in selected:
        item = baseline[qid]
        subset.append({"query_id": qid, "query": item["query"], "visible": item["visible"],
                       "baseline_bbox": item["bbox"]})
        audit.append({"query_id": qid, "priority_score": score, "reasons": reasons,
                      "candidate_count": count, "max_candidate_score": max_score,
                      "baseline_area": box_area, "query": item["query"]})
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    (output / "competition_subset.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "selection_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {"total_queries": len(baseline), "excluded_old_tiled": len(excluded_tiled),
               "excluded_changed_from_v28": len(changed), "eligible_scored": len(ranked),
               "selected": len(subset), "limit": args.limit,
               "minimum_selected_priority": selected[-1][0] if selected else None}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
