"""Apply the frozen conservative ordinal route on top of a base reranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_cache import validate_candidate_cache
from src.ordinal_router import apply_ordinal_router


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--base-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dedup-iou", type=float, default=0.85)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    annotations = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    cache = json.loads(resolve(args.cache).read_text(encoding="utf-8"))
    base = json.loads(resolve(args.base_candidates).read_text(encoding="utf-8"))
    validate_candidate_cache(cache, annotations)
    output_records, stats = apply_ordinal_router(
        annotations, cache["records"], base, args.dedup_iou
    )
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reranked_candidates.json").write_text(
        json.dumps(output_records, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "name": "ordinal_router_v1",
        "base": str(resolve(args.base_candidates).resolve()),
        "gate": "explicit second/2nd from left/right only",
        "dedup_iou": args.dedup_iou,
        "stats": stats,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"candidates={output / 'reranked_candidates.json'}")


if __name__ == "__main__":
    main()
