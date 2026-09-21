"""Deterministically rerank tiled candidates with ordinal/position language."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.metrics import compute_iou


def choose(query, candidates):
    text = query.lower()
    axis, rank, reason = None, None, "model_score"
    rules = [
        (r"leftmost|far left|on the left|left side", "x", 0, "leftmost"),
        (r"rightmost|far right|on the right|right side", "x", -1, "rightmost"),
        (r"\b(?:top|upper)\b", "y", 0, "top"),
        (r"\b(?:bottom|lower)\b", "y", -1, "bottom"),
        (r"\b(?:first|1st)\b", "x", 0, "first"),
        (r"\b(?:second|2nd)\b", "x", 1, "second"),
        (r"\b(?:third|3rd)\b", "x", 2, "third"),
        (r"\blast\b", "x", -1, "last"),
    ]
    for pattern, candidate_axis, candidate_rank, label in rules:
        if re.search(pattern, text):
            axis, rank, reason = candidate_axis, candidate_rank, label
            break
    if axis is None:
        return max(candidates, key=lambda x: float(x.get("score", 0))), reason
    ordered = sorted(
        candidates,
        key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2
        if axis == "x" else (item["bbox"][1] + item["bbox"][3]) / 2,
    )
    rank = max(-len(ordered), min(len(ordered) - 1, rank))
    return ordered[rank], reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="outputs/experiments/tiled_difficult/run/tiled_candidate_records.json")
    parser.add_argument("--output-dir", default="outputs/experiments/tiled_position_rerank")
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    records = json.loads(resolve(args.records).read_text(encoding="utf-8"))
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected, correct, no_candidate = {}, 0, 0
    for query_id, record in records.items():
        candidates = record.get("tiled_candidates", [])
        if not candidates:
            no_candidate += 1
            continue
        candidate, reason = choose(record.get("query", ""), candidates)
        gt = record.get("bbox")
        value = compute_iou(candidate["bbox"], gt) if gt else None
        correct += value is not None and value >= 0.5
        selected[query_id] = {
            "query": record.get("query", ""), "bbox": candidate["bbox"],
            "score": candidate.get("score", 0), "reason": reason, "iou": value,
        }
    total = len(records)
    metrics = {
        "samples": total, "selected": len(selected), "no_candidate": no_candidate,
        "correct": correct, "subset_acc": correct / total if total else 0,
        "refcoco_full_delta_pp": 100 * correct / 10834,
    }
    (output / "selected_candidates.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report = (
        "# Tiled Position Rerank\n\n"
        f"- Samples: {total}\n- Selected: {len(selected)}\n- No candidate: {no_candidate}\n"
        f"- Correct: {correct}\n- Difficult-subset ACC: {100*metrics['subset_acc']:.2f}%\n"
        f"- RefCOCO full ACC contribution: {metrics['refcoco_full_delta_pp']:+.2f} pp\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
