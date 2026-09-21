"""Normalize model candidate records into the canonical reusable cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_cache import build_candidate_cache, validate_candidate_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-name", default="unknown")
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    annotations = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    source = json.loads(resolve(args.source).read_text(encoding="utf-8"))
    cache = build_candidate_cache(annotations, source, args.source_name)
    validate_candidate_cache(cache, annotations)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(cache["stats"], indent=2))
    print(f"candidate_fingerprint={cache['candidate_fingerprint']}")
    print(f"cache={output}")


if __name__ == "__main__":
    main()
