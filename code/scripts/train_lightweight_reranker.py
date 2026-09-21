"""Train on frozen dev IDs and create CPU-only reranked candidate records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_cache import validate_candidate_cache
from src.lightweight_reranker import rerank_records, train_ridge_reranker
from src.validation_split import load_manifest, validate_split_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regularization", type=float, default=10.0)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    annotations = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    cache = json.loads(resolve(args.cache).read_text(encoding="utf-8"))
    manifest = load_manifest(resolve(args.split_manifest))
    validate_candidate_cache(cache, annotations)
    validate_split_manifest(annotations, manifest)
    model = train_ridge_reranker(
        annotations, cache["records"], manifest["splits"]["dev"], args.regularization
    )
    model["training_split"] = "dev"
    model["dataset_fingerprint"] = manifest["dataset_fingerprint"]
    model["candidate_fingerprint"] = cache["candidate_fingerprint"]
    reranked = rerank_records(annotations, cache["records"], model)
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "reranked_candidates.json").write_text(json.dumps(reranked, ensure_ascii=False), encoding="utf-8")
    print(f"training_samples={model['training_samples']}")
    print(f"training_candidates={model['training_candidates']}")
    print(f"train_rmse={model['train_rmse']:.6f}")
    print(f"model={output / 'model.json'}")
    print(f"candidates={output / 'reranked_candidates.json'}")


if __name__ == "__main__":
    main()
