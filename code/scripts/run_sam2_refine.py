"""Refine selected RefCOCO boxes with SAM2 and audit IoU 0.3--0.6 changes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_iou


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape[-2:]
    # Pixel upper bounds are exclusive in the normalized xyxy convention.
    return [
        float(xs.min()) / width,
        float(ys.min()) / height,
        float(xs.max() + 1) / width,
        float(ys.max() + 1) / height,
    ]


def report(records):
    count = len(records)
    before_correct = sum(x["input_iou"] >= 0.5 for x in records.values())
    after_correct = sum(x["refined_iou"] >= 0.5 for x in records.values())
    new_correct = sum(x["input_iou"] < 0.5 <= x["refined_iou"] for x in records.values())
    new_wrong = sum(x["refined_iou"] < 0.5 <= x["input_iou"] for x in records.values())
    improved = sum(x["refined_iou"] > x["input_iou"] + 1e-9 for x in records.values())
    degraded = sum(x["refined_iou"] + 1e-9 < x["input_iou"] for x in records.values())
    empty = sum(x.get("empty_mask", False) for x in records.values())
    metrics = {
        "samples": count,
        "before_correct": before_correct,
        "after_correct": after_correct,
        "new_correct": new_correct,
        "new_wrong": new_wrong,
        "net_correct": new_correct - new_wrong,
        "improved": improved,
        "degraded": degraded,
        "empty_masks": empty,
        "mean_iou_before": sum(x["input_iou"] for x in records.values()) / count if count else 0,
        "mean_iou_after": sum(x["refined_iou"] for x in records.values()) / count if count else 0,
    }
    lines = [
        "# SAM2 Boundary Refinement Report", "",
        f"- Samples: {count}",
        f"- Correct before / after: {before_correct} / {after_correct}",
        f"- New correct / new wrong: {new_correct} / {new_wrong}",
        f"- Net correct: {metrics['net_correct']:+d}",
        f"- IoU improved / degraded: {improved} / {degraded}",
        f"- Mean IoU before / after: {metrics['mean_iou_before']:.4f} / {metrics['mean_iou_after']:.4f}",
        f"- Empty masks: {empty}", "",
        "PASS" if metrics["net_correct"] > 0 else "STOP: SAM2 produced no net ACC gain.", "",
    ]
    return metrics, "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default="outputs/experiments/tiled_difficult/sam2_iou_03_06.json")
    parser.add_argument("--difficult", default="outputs/experiments/tiled_difficult/refcoco_difficult.json")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--sam2-repo", default="/root/autodl-tmp/sam2")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--output-dir", default="outputs/experiments/sam2_refine_03_06")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    selection = load(resolve(args.selection))
    difficult = {x["query_id"]: x for x in load(resolve(args.difficult))}
    images_dir = resolve(args.images_dir)
    checkpoint_path = Path(args.checkpoint)
    missing_meta = [qid for qid in selection if qid not in difficult]
    if missing_meta:
        raise SystemExit(f"Missing difficult metadata for {len(missing_meta)} records; first: {missing_meta[0]}")

    def image_path(qid):
        item = difficult[qid]
        original = Path(item["visible"])
        if original.exists():
            return original
        mapped = images_dir / original.name
        if mapped.exists():
            return mapped
        return images_dir / f"COCO_train2014_{int(item['source_image_id']):012d}.jpg"

    missing_images = [qid for qid in selection if not image_path(qid).exists()]
    if missing_images:
        raise SystemExit(f"Missing images for {len(missing_images)} records; first: {missing_images[0]}")
    if not checkpoint_path.exists():
        raise SystemExit(f"SAM2 checkpoint not found: {checkpoint_path}")
    if args.dry_run:
        print(f"Dry run OK: {len(selection)} samples; checkpoint={checkpoint_path}")
        return

    sam2_repo = Path(args.sam2_repo)
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(args.model_config, str(checkpoint_path), device="cuda")
    predictor = SAM2ImagePredictor(model)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_json = output_dir / "checkpoint.json"
    records = {} if args.no_resume or not checkpoint_json.exists() else load(checkpoint_json)
    started = time.time()
    completed = 0

    for qid, item in tqdm(selection.items(), desc="SAM2 refine", unit="query"):
        if qid in records:
            continue
        image = np.asarray(Image.open(image_path(qid)).convert("RGB"))
        height, width = image.shape[:2]
        box = item["input_bbox"]
        pixel_box = np.asarray([box[0] * width, box[1] * height, box[2] * width, box[3] * height], dtype=np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(box=pixel_box, multimask_output=True)
        best_index = int(np.argmax(scores))
        refined = mask_bbox(np.asarray(masks[best_index], dtype=bool))
        empty = refined is None
        if empty:
            refined = box
        gt = item["gt_bbox"]
        records[qid] = {
            "query": item.get("query", ""), "input_bbox": box, "refined_bbox": refined,
            "gt_bbox": gt, "input_iou": compute_iou(box, gt),
            "refined_iou": compute_iou(refined, gt), "sam_score": float(scores[best_index]),
            "empty_mask": empty,
        }
        completed += 1
        if completed % 10 == 0:
            save(checkpoint_json, records)

    save(checkpoint_json, records)
    save(output_dir / "refined_records.json", records)
    metrics, markdown = report(records)
    metrics["elapsed_seconds_this_run"] = time.time() - started
    save(output_dir / "metrics.json", metrics)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Details: {output_dir}")


if __name__ == "__main__":
    main()
