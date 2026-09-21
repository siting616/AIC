"""Generate tiled candidates for a frozen unlabeled competition subset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.baseline import build_predictor
from src.metrics import compute_iou
from src.postprocess import sanitize_bbox


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def bounds(width, height, overlap):
    tw, th = round(width / (2 - overlap)), round(height / (2 - overlap))
    return [(x, y, x + tw, y + th) for y in (0, height - th) for x in (0, width - tw)]


def remap(box, crop, width, height):
    left, top, right, bottom = crop
    return sanitize_bbox([
        (left + box[0] * (right-left)) / width, (top + box[1] * (bottom-top)) / height,
        (left + box[2] * (right-left)) / width, (top + box[3] * (bottom-top)) / height,
    ])


def dedupe(candidates, threshold=.85):
    result = []
    for candidate in sorted(candidates, key=lambda x: x.get("score", 0), reverse=True):
        if not any(compute_iou(candidate["bbox"], old["bbox"]) >= threshold for old in result):
            result.append(candidate)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/refcoco_tiled.yaml")
    parser.add_argument("--subset", default="outputs/experiments/tiled_competition/competition_subset.json")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/tiled_competition/run")
    parser.add_argument("--overlap", type=float, default=.20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    subset = load(resolve(args.subset))
    images_dir = resolve(args.images_dir)

    def image_path(item):
        original = Path(item["visible"])
        options = [original, images_dir / original.name, images_dir / original]
        for option in options:
            if option.exists():
                return option
        return options[1]

    missing = [x["query_id"] for x in subset if not image_path(x).exists()]
    if missing:
        raise SystemExit(f"Missing images for {len(missing)} samples; first: {missing[0]}")
    if args.dry_run:
        print(f"Dry run OK: {len(subset)} competition queries; 4 tiles/query")
        return
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    predictor = build_predictor(cfg)
    output = resolve(args.output_dir)
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.json"
    records = {} if args.no_resume or not checkpoint.exists() else load(checkpoint)
    done, started = 0, time.time()
    for item in tqdm(subset, desc="Tiled competition", unit="query"):
        qid = item["query_id"]
        if qid in records:
            continue
        image = Image.open(image_path(item)).convert("RGB")
        width, height = image.size
        candidates, errors = [], []
        for index, crop in enumerate(bounds(width, height, args.overlap)):
            path = crops / f"{qid}_tile{index}.jpg"
            image.crop(crop).save(path, quality=95)
            try:
                detail = predictor.predict_detailed({"visible_path": str(path), "query": item["query"]})
                for candidate in detail.get("candidates", []):
                    candidates.append({
                        "bbox": remap(candidate["bbox"], crop, width, height),
                        "score": float(candidate.get("score", 0)), "label": candidate.get("label", ""),
                        "source": "tile", "tile_index": index,
                    })
            except Exception as exc:
                errors.append({"tile_index": index, "error": f"{type(exc).__name__}: {exc}"})
        candidates = dedupe(candidates)
        genuinely_new = [x for x in candidates if compute_iou(x["bbox"], item["baseline_bbox"]) < .85]
        records[qid] = {
            "query": item["query"], "baseline_bbox": item["baseline_bbox"],
            "tiled_candidates": candidates, "new_candidates": genuinely_new,
            "new_candidate_count": len(genuinely_new), "errors": errors,
        }
        done += 1
        if done % 10 == 0:
            save(checkpoint, records)
    save(checkpoint, records)
    save(output / "tiled_candidate_records.json", records)
    stats = {
        "samples": len(records), "with_candidates": sum(bool(x["tiled_candidates"]) for x in records.values()),
        "with_genuinely_new_candidates": sum(bool(x["new_candidates"]) for x in records.values()),
        "mean_new_candidates": sum(x["new_candidate_count"] for x in records.values()) / len(records),
        "elapsed_seconds_this_run": time.time() - started,
    }
    save(output / "metrics.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Details: {output}")


if __name__ == "__main__":
    main()
