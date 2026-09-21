"""Cache CPU HSV color fractions for each color-query candidate box."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.color_router import build_color_feature_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=32)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    annotations = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    cache = json.loads(resolve(args.cache).read_text(encoding="utf-8"))
    features = build_color_feature_cache(annotations, cache["records"], args.sample_size)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(features, ensure_ascii=False), encoding="utf-8")
    print(f"color_queries={len(features['records'])}")
    print(f"cache={output}")


if __name__ == "__main__":
    main()
