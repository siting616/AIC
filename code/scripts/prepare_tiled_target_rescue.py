"""Convert tiled detections into a current-baseline target-rescue cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (aa+bb-inter) if aa+bb-inter else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--tiled-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.10)
    args = parser.parse_args()
    baseline, tiled = load(args.baseline), load(args.tiled_records)
    records, priority = {}, []
    for sample_id, source in tiled.items():
        if sample_id not in baseline:
            continue
        old = baseline[sample_id]["bbox"]
        candidates = [{"bbox": old, "score": 1.0, "label": "CURRENT BASELINE",
                       "source": "baseline", "rank": 1}]
        new = []
        for item in source.get("tiled_candidates", []):
            score = float(item.get("score", 0.0))
            if score < args.min_score or iou(old, item["bbox"]) >= 0.85:
                continue
            candidate = dict(item)
            candidate.update({"source": "tile", "rank": len(new)+2})
            if not any(iou(candidate["bbox"], x["bbox"]) >= 0.85 for x in new):
                new.append(candidate)
        new.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        new = new[:args.top_k-1]
        if not new:
            continue
        candidates.extend(new)
        records[sample_id] = {"query": baseline[sample_id]["query"], "candidates": candidates}
        strongest = max(float(x.get("score", 0.0)) for x in new)
        priority.append((strongest, sample_id))
    priority.sort(reverse=True)
    ids = [sample_id for _, sample_id in priority]
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate_cache.json").write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {"source_samples": len(tiled), "eligible_queries": len(ids),
             "mean_candidates_including_baseline": sum(len(x["candidates"]) for x in records.values()) / max(len(records), 1),
             "min_score": args.min_score, "top_k": args.top_k, "submission_created": False}
    (output / "metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
