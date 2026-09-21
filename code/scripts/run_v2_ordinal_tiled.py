"""GPU tiled recovery for V2 ordinal families.

Runs one canonical target prompt per family rather than repeating the original
ordinal query.  Each scene uses four overlapping 2x2 crops plus three vertical
strips.  Results are proposal data only and never touch the frozen baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def tile_bounds(width: int, height: int, overlap: float) -> list[tuple[str, tuple[int, int, int, int]]]:
    tw, th = round(width / (2 - overlap)), round(height / (2 - overlap))
    tiles = [
        (f"grid_{row}_{col}", (x, y, x + tw, y + th))
        for row, y in enumerate((0, height - th))
        for col, x in enumerate((0, width - tw))
    ]
    strip_width = round(width / (3 - 2 * overlap))
    starts = (0, round((width - strip_width) / 2), width - strip_width)
    tiles.extend((f"strip_{index}", (x, 0, x + strip_width, height)) for index, x in enumerate(starts))
    return tiles


def remap(box, crop, width: int, height: int) -> list[float]:
    left, top, right, bottom = crop
    mapped = [
        (left + float(box[0]) * (right - left)) / width,
        (top + float(box[1]) * (bottom - top)) / height,
        (left + float(box[2]) * (right - left)) / width,
        (top + float(box[3]) * (bottom - top)) / height,
    ]
    return [min(1.0, max(0.0, value)) for value in mapped]


def image_path(images_dir: Path, scene_id: str) -> Path:
    options = [
        images_dir / "visible" / f"{scene_id}{suffix}"
        for suffix in (".png", ".jpg", ".jpeg")
    ] + [images_dir / f"{scene_id}{suffix}" for suffix in (".png", ".jpg", ".jpeg")]
    return next((path for path in options if path.exists()), options[0])


def dedupe(candidates: list[dict], threshold: float = 0.85) -> list[dict]:
    from src.metrics import compute_iou

    kept = []
    for candidate in sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True):
        if all(compute_iou(candidate["bbox"], other["bbox"]) < threshold for other in kept):
            kept.append(candidate)
    return kept


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/competition_multiprompt.yaml")
    parser.add_argument("--manifest", default="outputs/v2/ordinal/tiled_recovery_manifest.json")
    parser.add_argument("--images-dir", default="初赛数据集-基于大模型的多模态视觉理解与推理/Images")
    parser.add_argument("--output-dir", default="outputs/v2/ordinal/tiled_gpu")
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load(resolve(args.manifest))
    if args.limit is not None:
        manifest = manifest[: args.limit]
    images_dir = resolve(args.images_dir)
    missing = [item["scene_id"] for item in manifest if not image_path(images_dir, item["scene_id"]).exists()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} visible images; first scene: {missing[0]}")
    if args.dry_run:
        print(json.dumps({
            "families": len(manifest),
            "detector_passes_per_family": 7,
            "total_detector_passes": 7 * len(manifest),
            "images_dir": str(images_dir),
            "proposal_only": True,
        }, ensure_ascii=False, indent=2))
        return

    # Heavy dependencies stay below the dry-run gate so local planning does not
    # require or mutate the previously configured GPU environment.
    import torch
    import yaml
    from PIL import Image
    from tqdm import tqdm
    from src.baseline import build_predictor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; activate the previously configured GPU environment.")
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    predictor = build_predictor(cfg)
    output_dir = resolve(args.output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    records = {} if args.no_resume or not checkpoint_path.exists() else load(checkpoint_path)
    new_count = 0
    started = time.time()
    for family in tqdm(manifest, desc="Ordinal tiled recovery", unit="family"):
        family_id = f"{family['scene_id']}|{family['target']}|{family['direction']}"
        if family_id in records:
            continue
        source = Image.open(image_path(images_dir, family["scene_id"])).convert("RGB")
        width, height = source.size
        candidates, errors = [], []
        for tile_name, crop in tile_bounds(width, height, args.overlap):
            crop_path = crops_dir / f"{family['scene_id']}_{tile_name}.jpg"
            source.crop(crop).save(crop_path, quality=95)
            try:
                detail = predictor.predict_detailed({
                    "visible_path": str(crop_path),
                    "query": family["target"],
                })
                for candidate in detail.get("candidates", []):
                    candidates.append({
                        "bbox": remap(candidate["bbox"], crop, width, height),
                        "score": float(candidate.get("score", 0.0)),
                        "label": candidate.get("label", ""),
                        "source": "ordinal_tiled_recovery",
                        "tile": tile_name,
                    })
            except Exception as exc:
                errors.append({"tile": tile_name, "error": f"{type(exc).__name__}: {exc}"})
        records[family_id] = {
            "scene_id": family["scene_id"],
            "target": family["target"],
            "direction": family["direction"],
            "query_ids": family["query_ids"],
            "required_instances": family["required_instances"],
            "candidates": dedupe(candidates),
            "errors": errors,
        }
        new_count += 1
        if new_count % 10 == 0:
            save(checkpoint_path, records)
    save(checkpoint_path, records)
    save(output_dir / "tiled_family_candidates.json", records)
    metrics = {
        "requested_families": len(manifest),
        "saved_families": len(records),
        "families_with_enough_candidates": sum(
            len(item["candidates"]) >= item["required_instances"] for item in records.values()
        ),
        "errors": sum(len(item["errors"]) for item in records.values()),
        "new_families_this_run": new_count,
        "elapsed_seconds_this_run": time.time() - started,
        "gpu": torch.cuda.get_device_name(0),
        "proposal_only": True,
    }
    save(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
