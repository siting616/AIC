"""Generate 2x2 overlapping tiled candidates for the frozen RefCOCO hard100.

The experiment reuses the existing GroundingDINO checkpoint.  It evaluates
candidate coverage only and never creates a competition submission.
"""

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


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def tile_bounds(width: int, height: int, overlap: float, grid_size: int = 2):
    # grid*c - (grid-1)*overlap*c = full size.
    denominator = grid_size - (grid_size - 1) * overlap
    tile_w = min(width, round(width / denominator))
    tile_h = min(height, round(height / denominator))
    xs = [round(i * (width - tile_w) / max(1, grid_size - 1)) for i in range(grid_size)]
    ys = [round(i * (height - tile_h) / max(1, grid_size - 1)) for i in range(grid_size)]
    return [(x, y, x + tile_w, y + tile_h) for y in ys for x in xs]


def remap_box(box, crop, width: int, height: int):
    left, top, right, bottom = crop
    crop_w, crop_h = right - left, bottom - top
    x1, y1, x2, y2 = box
    return sanitize_bbox([
        (left + x1 * crop_w) / width,
        (top + y1 * crop_h) / height,
        (left + x2 * crop_w) / width,
        (top + y2 * crop_h) / height,
    ])


def dedupe(candidates, threshold: float):
    result = []
    for candidate in sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        if any(compute_iou(candidate["bbox"], old["bbox"]) >= threshold for old in result):
            continue
        result.append(candidate)
    return result


def old_candidates(details, query_id):
    result = []
    for item in details.get(query_id, {}).get("reranked_candidates", [])[:20]:
        box = item.get("bbox")
        if not box:
            continue
        result.append({
            "bbox": sanitize_bbox(box),
            "score": float(item.get("rerank_score", item.get("score", 0.0))),
            "source": "old_pool",
            "label": item.get("label", ""),
        })
    return result


def report(records):
    count = len(records)
    old_covered = sum(x["old_oracle_iou"] >= 0.5 for x in records.values())
    tiled_covered = sum(x["tiled_oracle_iou"] >= 0.5 for x in records.values())
    combined_covered = sum(x["combined_oracle_iou"] >= 0.5 for x in records.values())
    newly_covered = sum(x["old_oracle_iou"] < 0.5 <= x["combined_oracle_iou"] for x in records.values())
    values = {
        "samples": count,
        "old_candidate_coverage": old_covered / count if count else 0.0,
        "tiled_candidate_coverage": tiled_covered / count if count else 0.0,
        "combined_candidate_coverage": combined_covered / count if count else 0.0,
        "newly_covered_samples": newly_covered,
        "new_coverage_pp": 100.0 * newly_covered / count if count else 0.0,
        "mean_new_candidates": sum(x["new_candidate_count"] for x in records.values()) / count if count else 0.0,
        "gate_pass": newly_covered > 0,
    }
    lines = [
        "# Tiled Hard100 Candidate Report", "",
        f"- Samples: {count}",
        f"- Old candidate coverage: {100*values['old_candidate_coverage']:.2f}%",
        f"- Tiled-only candidate coverage: {100*values['tiled_candidate_coverage']:.2f}%",
        f"- Combined candidate coverage: {100*values['combined_candidate_coverage']:.2f}%",
        f"- Newly covered samples: {newly_covered}",
        f"- New coverage: {values['new_coverage_pp']:+.2f} pp",
        f"- Mean genuinely new candidates: {values['mean_new_candidates']:.2f}",
        "", "PASS: run the RefCOCO difficult subset." if values["gate_pass"] else
        "STOP: tiled inference produced no new correct candidate coverage.", "",
    ]
    return values, "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/refcoco_tiled.yaml")
    parser.add_argument("--hard100", default="outputs/experiments/tiled_hard100/hard100.json")
    parser.add_argument(
        "--expected-count", type=int, default=100,
        help="Safety check for sample count; use 0 for a prebuilt variable-size subset",
    )
    parser.add_argument("--old-details", default="outputs/experiments/refcoco_v18_all/details.json")
    parser.add_argument("--output-dir", default="outputs/experiments/tiled_hard100/run")
    parser.add_argument(
        "--images-dir",
        help="Cloud COCO image directory; remaps Windows paths using the filename",
    )
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--grid-size", type=int, choices=(2, 3), default=2)
    parser.add_argument("--dedup-iou", type=float, default=0.85)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.overlap < 1.0:
        raise SystemExit("--overlap must be in [0, 1).")
    hard100 = load(resolve(args.hard100))
    if args.expected_count > 0 and len(hard100) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} frozen samples, found {len(hard100)} records."
        )
    images_dir = resolve(args.images_dir) if args.images_dir else None

    def image_path(item):
        original = Path(item["visible"])
        if original.exists():
            return original
        if images_dir:
            mapped = images_dir / original.name
            if mapped.exists():
                return mapped
            image_id = item.get("source_image_id")
            if image_id is not None:
                canonical = images_dir / f"COCO_train2014_{int(image_id):012d}.jpg"
                if canonical.exists():
                    return canonical
        return original

    missing = [x["query_id"] for x in hard100 if not image_path(x).exists()]
    if missing:
        raise SystemExit(f"Missing images for {len(missing)} samples; first: {missing[0]}")
    if args.dry_run:
        print(f"Dry run OK: {len(hard100)} samples; {args.grid_size ** 2} tiles/sample; overlap={args.overlap:.0%}")
        return

    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    predictor = build_predictor(cfg)
    details = load(resolve(args.old_details))
    output_dir = resolve(args.output_dir)
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.json"
    records = {} if args.no_resume or not checkpoint.exists() else load(checkpoint)
    checkpoint_every = int(cfg.get("runtime", {}).get("checkpoint_every", 10))
    completed = 0
    started = time.time()

    for item in tqdm(hard100, desc="Tiled hard100", unit="query"):
        query_id = item["query_id"]
        if query_id in records:
            continue
        image = Image.open(image_path(item)).convert("RGB")
        width, height = image.size
        tiled = []
        errors = []
        for tile_index, crop in enumerate(tile_bounds(width, height, args.overlap, args.grid_size)):
            crop_path = crop_dir / f"{query_id}_tile{tile_index}.jpg"
            image.crop(crop).save(crop_path, quality=95)
            sample = {"visible_path": str(crop_path), "query": item["query"]}
            try:
                detail = predictor.predict_detailed(sample)
                for candidate in detail.get("candidates", []):
                    tiled.append({
                        "bbox": remap_box(candidate["bbox"], crop, width, height),
                        "score": float(candidate.get("score", 0.0)),
                        "label": candidate.get("label", ""),
                        "source": "tile",
                        "tile_index": tile_index,
                        "tile_bounds_px": list(crop),
                    })
            except Exception as exc:
                errors.append({"tile_index": tile_index, "error": f"{type(exc).__name__}: {exc}"})
        tiled = dedupe(tiled, args.dedup_iou)
        old = old_candidates(details, query_id)
        genuinely_new = [x for x in tiled if not any(compute_iou(x["bbox"], y["bbox"]) >= args.dedup_iou for y in old)]
        combined = dedupe(old + tiled, args.dedup_iou)
        gt = item["bbox"]
        oracle = lambda xs: max((compute_iou(x["bbox"], gt) for x in xs), default=0.0)
        records[query_id] = {
            "query": item["query"], "bbox": gt,
            "old_oracle_iou": oracle(old), "tiled_oracle_iou": oracle(tiled),
            "combined_oracle_iou": oracle(combined),
            "old_candidate_count": len(old), "tiled_candidate_count": len(tiled),
            "new_candidate_count": len(genuinely_new),
            "new_candidates": genuinely_new, "tiled_candidates": tiled, "errors": errors,
        }
        completed += 1
        if completed % checkpoint_every == 0:
            save(checkpoint, records)

    save(checkpoint, records)
    save(output_dir / "tiled_candidate_records.json", records)
    metrics, markdown = report(records)
    metrics.update({"elapsed_seconds": time.time() - started, "overlap": args.overlap,
                    "dedup_iou": args.dedup_iou, "grid_size": args.grid_size})
    save(output_dir / "metrics.json", metrics)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Details: {output_dir}")


if __name__ == "__main__":
    main()
