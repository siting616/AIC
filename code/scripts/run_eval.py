"""Run baseline evaluation."""

from __future__ import annotations

import sys
import site
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

from src.baseline import build_predictor
from src.dataset import AICDataset
from src.metrics import compute_acc_at_05, per_sample_iou
from src.postprocess import sanitize_bbox


def load_config():
    with (PROJECT_ROOT / "configs" / "default.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    dataset = AICDataset(
        root_dir=str(PROJECT_ROOT / cfg["data_dir"]),
        json_path=str(PROJECT_ROOT / cfg["json_file"]),
    )
    predictor = build_predictor(cfg)

    predictions = {}
    gt = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        predictions[sample["sample_id"]] = sanitize_bbox(predictor.predict(sample))
        gt[sample["sample_id"]] = sample["bbox"]

    ious = per_sample_iou(predictions, gt)
    for sample_id, iou in ious.items():
        print(f"{sample_id}: IoU = {iou:.6f}")

    threshold = float(cfg["eval"].get("iou_threshold", 0.5))
    acc = compute_acc_at_05(predictions, gt, threshold=threshold)
    print(f"ACC@0.5 = {acc:.1f}")


if __name__ == "__main__":
    main()
