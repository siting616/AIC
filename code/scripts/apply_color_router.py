"""Apply the frozen safe-color candidate route on top of the current baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.color_router import SAFE_COLORS, apply_color_router


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--color-features", required=True)
    parser.add_argument("--base-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-margin", type=float, default=0.08)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    annotations = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    cache = json.loads(resolve(args.cache).read_text(encoding="utf-8"))
    features = json.loads(resolve(args.color_features).read_text(encoding="utf-8"))
    base = json.loads(resolve(args.base_candidates).read_text(encoding="utf-8"))
    records, stats = apply_color_router(
        annotations, cache["records"], features["records"], base,
        SAFE_COLORS, args.minimum_fraction, args.minimum_margin,
    )
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reranked_candidates.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    config = {
        "name": "color_router_v1", "safe_colors": sorted(SAFE_COLORS),
        "minimum_fraction": args.minimum_fraction, "minimum_margin": args.minimum_margin,
        "stats": stats,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"candidates={output / 'reranked_candidates.json'}")


if __name__ == "__main__":
    main()
