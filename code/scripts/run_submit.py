"""Generate prediction JSON and zip submission file."""

from __future__ import annotations

import argparse
import json
import sys
import site
import time
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

from src.baseline import build_predictor
from src.dataset import AICDataset
from src.postprocess import sanitize_bbox
from src.submit import generate_submission, save_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Run AIC inference and build submission.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config path relative to the project root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N queries for a smoke test; no submission zip is built.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and start inference from scratch.",
    )
    return parser.parse_args()


def load_config(config_path):
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset = AICDataset(
        root_dir=str(PROJECT_ROOT / cfg["data_dir"]),
        json_path=str(PROJECT_ROOT / cfg["json_file"]),
    )
    predictor = build_predictor(cfg)

    prediction_dir = PROJECT_ROOT / cfg["prediction_dir"]
    submission_dir = PROJECT_ROOT / cfg["submission_dir"]
    raw_pred_path = prediction_dir / "baseline_predictions.json"
    checkpoint_path = prediction_dir / "checkpoint_predictions.json"
    candidate_path = prediction_dir / "candidate_predictions.json"
    stats_path = prediction_dir / "run_stats.json"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    candidate_predictions = {}
    if checkpoint_path.exists() and not args.no_resume:
        with checkpoint_path.open("r", encoding="utf-8") as f:
            predictions = json.load(f)
        if candidate_path.exists():
            with candidate_path.open("r", encoding="utf-8") as f:
                candidate_predictions = json.load(f)
        print(f"Resuming from {len(predictions)} completed predictions.")

    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    checkpoint_every = int(cfg.get("runtime", {}).get("checkpoint_every", 100))
    completed_since_save = 0
    new_prediction_count = 0
    started_at = time.time()

    for idx in tqdm(range(total), desc="Predicting", unit="query"):
        sample = dataset[idx]
        if sample["sample_id"] in predictions:
            continue
        try:
            detail = predictor.predict_detailed(sample)
            final_bbox = sanitize_bbox(detail["bbox"])
            detail["bbox"] = final_bbox
            for candidate in detail.get("candidates", []):
                candidate["bbox"] = sanitize_bbox(candidate.get("bbox"))
            detail["query"] = sample["query"]
            predictions[sample["sample_id"]] = final_bbox
            candidate_predictions[sample["sample_id"]] = detail
        except Exception as exc:
            fallback = sanitize_bbox(None)
            predictions[sample["sample_id"]] = fallback
            candidate_predictions[sample["sample_id"]] = {
                "bbox": fallback,
                "candidates": [],
                "used_fallback": True,
                "fallback_reason": f"inference_error:{type(exc).__name__}",
                "error": str(exc),
                "query": sample["query"],
            }
        completed_since_save += 1
        new_prediction_count += 1
        if completed_since_save >= checkpoint_every:
            save_predictions(predictions, str(checkpoint_path))
            save_predictions(candidate_predictions, str(candidate_path))
            completed_since_save = 0

    save_predictions(predictions, str(checkpoint_path))
    save_predictions(predictions, str(raw_pred_path))
    save_predictions(candidate_predictions, str(candidate_path))
    elapsed = time.time() - started_at
    fallback_count = sum(
        bool(item.get("used_fallback")) for item in candidate_predictions.values()
    )
    rescue_count = sum(
        item.get("selection_mode") == "rescue_threshold"
        for item in candidate_predictions.values()
    )
    inference_error_count = sum(
        str(item.get("fallback_reason", "")).startswith("inference_error:")
        for item in candidate_predictions.values()
    )
    stats = {
        "requested_queries": total,
        "saved_predictions": len(predictions),
        "saved_candidate_records": len(candidate_predictions),
        "fallback_count": fallback_count,
        "fallback_rate": (
            fallback_count / len(candidate_predictions)
            if candidate_predictions
            else 0.0
        ),
        "rescue_count": rescue_count,
        "rescue_rate": (
            rescue_count / len(candidate_predictions)
            if candidate_predictions
            else 0.0
        ),
        "inference_error_count": inference_error_count,
        "elapsed_seconds_this_run": elapsed,
        "average_seconds_per_new_query": (
            elapsed / new_prediction_count if new_prediction_count else None
        ),
        "model": cfg.get("model", {}).get("grounding_dino", {}),
    }
    save_predictions(stats, str(stats_path))

    if args.limit is not None:
        print(f"Smoke test complete: {len(predictions)} predictions saved.")
        print(f"Candidate details saved: {candidate_path}")
        print(f"Run statistics saved: {stats_path}")
        print("Submission generation skipped because --limit was used.")
        return

    if len(predictions) != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} predictions, found {len(predictions)}. "
            "Resume inference before building a submission."
        )

    output_json, output_zip = generate_submission(
        original_json_path=str(PROJECT_ROOT / cfg["json_file"]),
        predictions=predictions,
        output_json_path=str(submission_dir / "prediction.json"),
        output_zip_path=str(submission_dir / "submission.zip"),
    )
    print(f"Saved raw predictions: {raw_pred_path}")
    print(f"Saved submission JSON: {output_json}")
    print(f"Saved submission ZIP: {output_zip}")


if __name__ == "__main__":
    main()
