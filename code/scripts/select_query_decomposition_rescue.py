"""Freeze complex-query misses that can benefit from multi-prompt detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_iou
from src.query_prompts import generate_prompts
from src.validation_split import load_manifest, subset_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    full = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    annotations = subset_annotations(full, load_manifest(resolve(args.split_manifest)), args.split)
    cache = json.loads(resolve(args.cache).read_text(encoding="utf-8"))["records"]
    selected = []
    for sample_id, item in annotations.items():
        prompts = generate_prompts(item.get("query", ""), 4)
        detector_prompts = [prompt for prompt in prompts if prompt["type"] != "reference"]
        candidates = cache[sample_id]["candidates"]
        oracle = max((compute_iou(candidate["bbox"], item["bbox"]) for candidate in candidates), default=0.0)
        if len(detector_prompts) >= 2 and oracle < 0.5:
            selected.append({
                "query_id": sample_id, "query": item["query"], "prompts": prompts,
                "old_oracle_iou": oracle,
            })
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected={len(selected)} split={args.split}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
