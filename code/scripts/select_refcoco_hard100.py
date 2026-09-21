"""Select a frozen 100-sample RefCOCO set for tiled-candidate experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_iou


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bbox(value):
    return value.get("bbox") if isinstance(value, dict) else value


def category(query: str) -> str:
    text = query.lower()
    if any(x in text for x in ("first", "second", "third", "last", "leftmost", "rightmost")):
        return "ordinal"
    if any(x in text for x in ("left", "right", "middle", "center", "top", "bottom")):
        return "position"
    if any(x in text for x in ("behind", "front of", "next to", "between", "near", "beside")):
        return "relation"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/validation/refcoco_val.json")
    parser.add_argument("--baseline", default="outputs/experiments/refcoco_qwen3b_complex_all/reranked_predictions.json")
    parser.add_argument("--details", default="outputs/experiments/refcoco_v18_all/details.json")
    parser.add_argument("--output-dir", default="outputs/experiments/tiled_hard100")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    resolve = lambda p: Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
    annotations = load(resolve(args.annotations))
    baseline = load(resolve(args.baseline))
    details = load(resolve(args.details))
    image_counts = Counter(str(v.get("source_image_id", v.get("visible", ""))) for v in annotations.values())
    ranked = []
    for query_id, sample in annotations.items():
        gt = bbox(sample)
        pred = bbox(baseline.get(query_id))
        if not gt or not pred:
            continue
        before_iou = compute_iou(pred, gt)
        if before_iou >= 0.5:
            continue
        candidates = details.get(query_id, {}).get("reranked_candidates", [])[:20]
        oracle_iou = max((compute_iou(bbox(item), gt) for item in candidates), default=0.0)
        # Candidate-missing examples are the purpose of the tiled experiment.
        if oracle_iou >= 0.5:
            continue
        area = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
        cat = category(sample.get("query", ""))
        crowded = image_counts[str(sample.get("source_image_id", sample.get("visible", "")))]
        priority = ({"ordinal": 4, "position": 3, "relation": 2, "other": 0}[cat]
                    + min(crowded, 8) * 0.25 + (2 if area <= 0.03 else 1 if area <= 0.08 else 0))
        ranked.append({
            "query_id": query_id,
            "query": sample.get("query", ""),
            "source_image_id": sample.get("source_image_id"),
            "visible": sample.get("visible"),
            "bbox": gt,
            "baseline_bbox": pred,
            "baseline_iou": before_iou,
            "old_oracle_iou": oracle_iou,
            "target_area": area,
            "category": cat,
            "same_image_query_count": crowded,
            "priority": priority,
        })
    ranked.sort(key=lambda x: (-x["priority"], x["target_area"], x["old_oracle_iou"], x["query_id"]))

    # Keep category coverage while filling the remainder by the frozen priority.
    selected = []
    quotas = {"ordinal": 30, "position": 30, "relation": 20, "other": 20}
    for cat, quota in quotas.items():
        selected.extend([x for x in ranked if x["category"] == cat][:quota])
    selected_ids = {x["query_id"] for x in selected}
    for item in ranked:
        if len(selected) >= args.count:
            break
        if item["query_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["query_id"])
    selected = selected[: args.count]

    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "hard100.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "hard100_ids.json").write_text(
        json.dumps([x["query_id"] for x in selected], ensure_ascii=False, indent=2), encoding="utf-8")
    subset = {x["query_id"]: annotations[x["query_id"]] for x in selected}
    (output / "hard100_annotations.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(x["category"] for x in selected)
    summary = {"selected": len(selected), "eligible_candidate_missing": len(ranked), "categories": dict(counts),
               "small_targets_area_le_003": sum(x["target_area"] <= 0.03 for x in selected)}
    (output / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Frozen IDs: {output / 'hard100_ids.json'}")


if __name__ == "__main__":
    main()
