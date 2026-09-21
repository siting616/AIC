"""Merge saved tiled candidates into the canonical base candidate cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_cache import validate_candidate_cache
from src.candidate_merge import merge_tiled_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--tiled-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--dedup-iou", type=float, default=0.85)
    parser.add_argument("--source-name", default="grounding_dino_tiled")
    parser.add_argument("--candidate-field", default="new_candidates",
                        help="Source record field containing candidates; use 'candidates' for multi-prompt output")
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    base = json.loads(resolve(args.base_cache).read_text(encoding="utf-8"))
    tiled = json.loads(resolve(args.tiled_records).read_text(encoding="utf-8"))
    merged = merge_tiled_records(
        base, tiled, args.max_candidates, args.dedup_iou, args.source_name, args.candidate_field
    )
    validate_candidate_cache(merged, base["records"])
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(merged["merge_stats"], indent=2))
    print(json.dumps(merged["stats"], indent=2))
    print(f"cache={output}")


if __name__ == "__main__":
    main()
