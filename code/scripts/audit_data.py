"""Audit the formal AIC dataset before cloud inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Audit AIC JSON and image files.")
    parser.add_argument("--config", default="configs/preliminary.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_root = (PROJECT_ROOT / cfg["data_dir"]).resolve()
    json_path = (PROJECT_ROOT / cfg["json_file"]).resolve()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Query JSON must be a non-empty object.")

    required = ("visible", "infrared", "depth", "query")
    missing_fields = []
    missing_files = []
    modality_files = {name: set() for name in required[:3]}
    image_props = {name: Counter() for name in required[:3]}

    for query_id, item in data.items():
        for field in required:
            if field not in item:
                missing_fields.append((query_id, field))
        if missing_fields:
            continue
        for modality in required[:3]:
            path = data_root / item[modality]
            modality_files[modality].add(path)
            if not path.is_file():
                missing_files.append((query_id, modality, str(path)))

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields[:10]}")
    if missing_files:
        raise FileNotFoundError(f"Missing image files: {missing_files[:10]}")

    for modality, paths in modality_files.items():
        for path in paths:
            with Image.open(path) as image:
                image_props[modality][(image.size, image.mode, path.suffix.lower())] += 1

    visible_by_query = Counter(item["visible"] for item in data.values())
    print(f"Queries: {len(data)}")
    print(f"Unique image groups: {len(visible_by_query)}")
    print(
        "Queries per image: "
        f"min={min(visible_by_query.values())}, "
        f"max={max(visible_by_query.values())}, "
        f"avg={len(data) / len(visible_by_query):.4f}"
    )
    print(f"Records containing bbox: {sum('bbox' in item for item in data.values())}")
    for modality in required[:3]:
        print(f"{modality}: {len(modality_files[modality])} referenced files")
        for props, count in image_props[modality].most_common():
            print(f"  {props}: {count}")
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
