"""Generate GroundingDINO candidates from multiple query decompositions."""

from __future__ import annotations

import argparse
import json
import site
import sys
import time
from collections import Counter
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
from src.query_prompts import generate_prompts, merge_prompt_candidates
from src.submit import generate_submission, save_predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="run_multiprompt_candidates v35c-pathfix")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--per-prompt", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dedup-iou", type=float, default=0.85)
    parser.add_argument("--ids-json", help="Optional frozen JSON list/dict of query IDs")
    parser.add_argument("--images-dir", help="Cloud image directory used to remap Windows annotation paths")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection and prompt decomposition without loading the model")
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    dataset = AICDataset(
        str(PROJECT_ROOT / cfg["data_dir"]),
        str(PROJECT_ROOT / cfg["json_file"]),
        str(resolve(args.images_dir)) if args.images_dir else None,
    )
    selected_ids = None
    if args.ids_json:
        selected_payload = json.loads(resolve(args.ids_json).read_text(encoding="utf-8"))
        if isinstance(selected_payload, list):
            selected_ids = {
                str(item.get("query_id", item.get("sample_id"))) if isinstance(item, dict) else str(item)
                for item in selected_payload
            }
        elif isinstance(selected_payload, dict):
            selected_ids = set(selected_payload)
        else:
            raise SystemExit("--ids-json must contain a JSON list or object")
    indices = [i for i, sample_id in enumerate(dataset.sample_ids) if selected_ids is None or sample_id in selected_ids]
    if args.limit is not None:
        indices = indices[:args.limit]
    if selected_ids is not None and len(indices) != len(selected_ids):
        missing = selected_ids - set(dataset.sample_ids)
        raise SystemExit(f"{len(missing)} selected query IDs are missing from the dataset")
    if args.dry_run:
        prompt_counts = Counter()
        for index in indices:
            prompts = generate_prompts(dataset[index]["query"], args.max_prompts)
            prompt_counts[len([p for p in prompts if p["type"] != "reference"])] += 1
        print(f"Dry run OK: {len(indices)} queries; detector passes/query={dict(sorted(prompt_counts.items()))}")
        return

    predictor = build_predictor(cfg)
    predictor.top_k = max(predictor.top_k, args.per_prompt)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_predictions.json"
    prediction_path = output_dir / "baseline_predictions.json"
    stats_path = output_dir / "candidate_stats.json"
    checkpoint_path = output_dir / "checkpoint.json"

    records = {}
    if checkpoint_path.exists() and not args.no_resume:
        records = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        print(f"Resuming from {len(records)} completed records.")

    total = len(indices)
    checkpoint_every = int(cfg.get("runtime", {}).get("checkpoint_every", 100))
    started_at = time.time()
    new_records = 0

    for index in tqdm(indices, desc="Multi-prompt candidates", unit="query"):
        sample = dataset[index]
        query_id = sample["sample_id"]
        if query_id in records:
            continue
        prompts = generate_prompts(sample["query"], args.max_prompts)
        prompt_results = []
        prompt_errors = []
        for prompt in prompts:
            # Keep the parsed reference in diagnostics, but do not spend a
            # detector forward pass on a box that is not a valid target answer.
            if prompt["type"] == "reference":
                continue
            prompt_sample = dict(sample)
            prompt_sample["query"] = prompt["text"]
            try:
                detail = predictor.predict_detailed(prompt_sample)
                prompt_results.append(
                    {
                        "prompt_type": prompt["type"],
                        "prompt": prompt["text"],
                        "selection_mode": detail.get("selection_mode"),
                        "used_fallback": bool(detail.get("used_fallback")),
                        "candidates": detail.get("candidates", []),
                    }
                )
            except Exception as exc:
                prompt_errors.append(
                    {
                        "prompt_type": prompt["type"],
                        "prompt": prompt["text"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        candidates = merge_prompt_candidates(
            prompt_results,
            top_k=args.top_k,
            per_prompt=args.per_prompt,
            dedup_iou=args.dedup_iou,
        )
        for candidate in candidates:
            candidate["bbox"] = sanitize_bbox(candidate.get("bbox"))
        fallback = not candidates
        bbox = candidates[0]["bbox"] if candidates else sanitize_bbox(None)
        records[query_id] = {
            "query": sample["query"],
            "bbox": bbox,
            "candidates": candidates,
            "prompts": prompts,
            "prompt_results": prompt_results,
            "prompt_errors": prompt_errors,
            "used_fallback": fallback,
            "selection_mode": "multiprompt_fusion" if candidates else "center_fallback",
        }
        new_records += 1
        if new_records % checkpoint_every == 0:
            save_predictions(records, str(checkpoint_path))

    save_predictions(records, str(checkpoint_path))
    save_predictions(records, str(candidate_path))
    predictions = {query_id: record["bbox"] for query_id, record in records.items()}
    save_predictions(predictions, str(prediction_path))

    candidate_counts = Counter(len(record["candidates"]) for record in records.values())
    prompt_type_counts = Counter(
        candidate.get("prompt_type")
        for record in records.values()
        for candidate in record["candidates"]
    )
    stats = {
        "requested_queries": total,
        "saved_records": len(records),
        "fallback_count": sum(bool(record["used_fallback"]) for record in records.values()),
        "prompt_error_count": sum(len(record["prompt_errors"]) for record in records.values()),
        "candidate_count_distribution": dict(sorted(candidate_counts.items())),
        "selected_primary_prompt_types": dict(sorted(prompt_type_counts.items())),
        "average_candidates": (
            sum(count * samples for count, samples in candidate_counts.items()) / len(records)
            if records else 0.0
        ),
        "elapsed_seconds_this_run": time.time() - started_at,
        "new_records_this_run": new_records,
        "settings": {
            "max_prompts": args.max_prompts,
            "per_prompt": args.per_prompt,
            "top_k": args.top_k,
            "dedup_iou": args.dedup_iou,
        },
    }
    save_predictions(stats, str(stats_path))

    if args.limit is None and selected_ids is None and len(records) == len(dataset):
        generate_submission(
            str(PROJECT_ROOT / cfg["json_file"]),
            predictions,
            str(output_dir / "prediction.json"),
            str(output_dir / "submission_multiprompt_raw.zip"),
        )

    print(f"Saved records: {len(records)}")
    print(f"Candidates: {candidate_path}")
    print(f"Statistics: {stats_path}")


if __name__ == "__main__":
    main()
