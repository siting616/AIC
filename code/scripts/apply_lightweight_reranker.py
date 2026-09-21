"""Apply a frozen lightweight reranker model to a canonical candidate cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_cache import validate_candidate_cache
from src.lightweight_reranker import rerank_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    annotations = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    cache = json.loads(resolve(args.cache).read_text(encoding="utf-8"))
    model = json.loads(resolve(args.model).read_text(encoding="utf-8"))
    validate_candidate_cache(cache, annotations)
    records = rerank_records(annotations, cache["records"], model)
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reranked_candidates.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (output / "config.json").write_text(json.dumps({
        "name": "frozen_lightweight_reranker_inference",
        "model": str(resolve(args.model).resolve()),
        "cache": str(resolve(args.cache).resolve()),
    }, indent=2), encoding="utf-8")
    print(f"samples={len(records)}")
    print(f"output={output / 'reranked_candidates.json'}")


if __name__ == "__main__":
    main()
