"""Apply high-confidence cached SAM2 refinements to the current top-1 boxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.boundary_refiner import apply_cached_boundary_refinements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-candidates", required=True)
    parser.add_argument("--refinements", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-score", type=float, default=0.95)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    base = json.loads(resolve(args.base_candidates).read_text(encoding="utf-8"))
    refinements = json.loads(resolve(args.refinements).read_text(encoding="utf-8"))
    records, stats = apply_cached_boundary_refinements(base, refinements, args.minimum_score)
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reranked_candidates.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (output / "config.json").write_text(json.dumps({
        "name": "sam2_boundary_gate_v1", "minimum_score": args.minimum_score, "stats": stats,
    }, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
