"""Visualize ground-truth bboxes on visible images."""

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

from src.dataset import AICDataset
from src.visualize import draw_bbox_on_visible


def load_config():
    with (PROJECT_ROOT / "configs" / "default.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    dataset = AICDataset(
        root_dir=str(PROJECT_ROOT / cfg["data_dir"]),
        json_path=str(PROJECT_ROOT / cfg["json_file"]),
    )

    output_dir = PROJECT_ROOT / cfg["visualization_dir"]
    for idx in range(len(dataset)):
        save_path = draw_bbox_on_visible(dataset[idx], str(output_dir))
        print(f"Saved visualization: {save_path}")


if __name__ == "__main__":
    main()
