"""Rerank saved Top-K candidates on CPU using all three visual modalities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import AICDataset
from src.postprocess import sanitize_bbox
from src.rerank import rerank_candidates as rerank_v17_candidates
from src.rerank_v18 import rerank_candidates as rerank_v18_candidates
from src.submit import generate_submission, save_predictions


MODE_WEIGHTS = {
    "position": {
        "model": 0.65,
        "position": 0.35,
        "depth": 0.0,
        "infrared": 0.0,
    },
    "depth": {
        "model": 0.90,
        "position": 0.0,
        "depth": 0.10,
        "infrared": 0.0,
    },
    "position_depth": {
        "model": 0.125,
        "position": 0.775,
        "depth": 0.10,
        "infrared": 0.0,
    },
    "infrared": {
        "model": 0.95,
        "position": 0.0,
        "depth": 0.0,
        "infrared": 0.05,
    },
    "full": {
        "model": 0.70,
        "position": 0.15,
        "depth": 0.10,
        "infrared": 0.05,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank saved AIC candidates.")
    parser.add_argument("--config", default="configs/preliminary.yaml")
    parser.add_argument(
        "--strategy",
        choices=("v17", "v18"),
        default="v18",
        help="Use frozen V17 scoring or V18 conservative query routing.",
    )
    parser.add_argument(
        "--route-mode",
        choices=(
            "all", "extreme", "ordinal", "ordinal_first", "ordinal_later",
            "ordinal_second", "ordinal_third_plus",
        ),
        default="all",
        help="For V18, enable all routes or one ablation family.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_WEIGHTS),
        default="position",
        help="Use one isolated signal or the full multimodal reranker.",
    )
    parser.add_argument(
        "--candidates",
        default="outputs/preliminary/predictions/candidate_predictions.json",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--details",
        default=None,
    )
    return parser.parse_args()


def resolve(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    output_path = resolve(
        args.output
        or f"outputs/experiments/v2_{args.mode}/reranked_predictions.json"
    )
    details_path = resolve(
        args.details
        or f"outputs/experiments/v2_{args.mode}/reranked_details.json"
    )
    dataset = AICDataset(
        root_dir=str(PROJECT_ROOT / cfg["data_dir"]),
        json_path=str(PROJECT_ROOT / cfg["json_file"]),
    )
    saved = json.loads(resolve(args.candidates).read_text(encoding="utf-8"))
    rerank_candidates = (
        rerank_v18_candidates if args.strategy == "v18" else rerank_v17_candidates
    )
    samples = {
        dataset[index]["sample_id"]: dataset[index] for index in range(len(dataset))
    }
    predictions = {}
    details = {}
    changed = 0
    for query_id, record in tqdm(saved.items(), desc="Reranking", unit="query"):
        sample = samples[query_id]
        candidates = record.get("candidates", [])
        if args.strategy == "v18":
            reranked = rerank_candidates(
                sample,
                candidates,
                weights=MODE_WEIGHTS[args.mode],
                route_mode=args.route_mode,
            )
        else:
            reranked = rerank_candidates(
                sample,
                candidates,
                weights=MODE_WEIGHTS[args.mode],
            )
        if reranked:
            bbox = sanitize_bbox(reranked[0]["bbox"])
            if candidates and bbox != sanitize_bbox(candidates[0]["bbox"]):
                changed += 1
        else:
            bbox = sanitize_bbox(record.get("bbox"))
        predictions[query_id] = bbox
        details[query_id] = {
            "query": sample["query"],
            "bbox": bbox,
            "reranked_candidates": reranked,
            "used_original_fallback": bool(record.get("used_fallback")),
        }

    save_predictions(predictions, str(output_path))
    save_predictions(details, str(details_path))
    submission_dir = output_path.parent
    output_json, output_zip = generate_submission(
        original_json_path=str(PROJECT_ROOT / cfg["json_file"]),
        predictions=predictions,
        output_json_path=str(submission_dir / "prediction.json"),
        output_zip_path=str(submission_dir / f"submission_v2_{args.mode}.zip"),
    )
    print(f"Queries reranked: {len(predictions)}")
    print(f"Top candidate changed: {changed}")
    print(f"Mode: {args.mode}")
    print(f"Strategy: {args.strategy}")
    print(f"Route mode: {args.route_mode}")
    print(f"Predictions: {output_path}")
    print(f"Details: {details_path}")
    print(f"Submission JSON: {output_json}")
    print(f"Submission ZIP: {output_zip}")
    print("RERANK PASSED")


if __name__ == "__main__":
    main()
