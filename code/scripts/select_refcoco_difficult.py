"""Build the frozen RefCOCO difficult subset for tiled candidate generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_iou


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def classify(query):
    text = query.lower()
    if re.search(r"\b(first|second|third|fourth|last|1st|2nd|3rd|leftmost|rightmost)\b", text):
        return "ordinal"
    if re.search(r"\b(left|right|middle|center|top|bottom|upper|lower)\b", text):
        return "position"
    if re.search(r"\b(behind|between|beside|near|nearest|next to|in front of|under|above|below)\b", text):
        return "relation"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/validation/refcoco_val.json")
    parser.add_argument("--baseline", default="outputs/experiments/refcoco_qwen3b_complex_all/reranked_predictions.json")
    parser.add_argument("--details", default="outputs/experiments/refcoco_v18_all/details.json")
    parser.add_argument("--output", default="outputs/experiments/tiled_difficult/refcoco_difficult.json")
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    annotations, baseline, details = map(load, map(resolve, (args.annotations, args.baseline, args.details)))
    selected = []
    for query_id, sample in annotations.items():
        cat = classify(sample.get("query", ""))
        if cat is None:
            continue
        gt = sample.get("bbox")
        pred = baseline.get(query_id)
        if not gt or not pred or compute_iou(pred, gt) >= 0.5:
            continue
        candidates = details.get(query_id, {}).get("reranked_candidates", [])[:20]
        old_oracle = max((compute_iou(x.get("bbox"), gt) for x in candidates), default=0.0)
        if old_oracle >= 0.5:
            continue
        selected.append({
            "query_id": query_id, "query": sample.get("query", ""),
            "source_image_id": sample.get("source_image_id"), "visible": sample.get("visible"),
            "bbox": gt, "baseline_bbox": pred,
            "baseline_iou": compute_iou(pred, gt), "old_oracle_iou": old_oracle,
            "category": cat,
        })
    selected.sort(key=lambda x: (x["category"], x["source_image_id"] or -1, x["query_id"]))
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"selected": len(selected), "categories": dict(Counter(x["category"] for x in selected))}
    (output.parent / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Subset: {output}")


if __name__ == "__main__":
    main()
