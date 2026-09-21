"""Create a frozen image-grouped dev/holdout validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation_split import create_grouped_split, validate_split_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", default="data/validation/splits/refcoco_v1")
    parser.add_argument("--holdout-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--force", action="store_true", help="Replace existing split files")
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    annotations = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    output_dir = resolve(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"Split already exists: {manifest_path}. Use --force to replace it.")

    manifest = create_grouped_split(annotations, args.holdout_fraction, args.seed)
    validate_split_manifest(annotations, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, ids in manifest["splits"].items():
        subset = {sample_id: annotations[sample_id] for sample_id in ids}
        (output_dir / f"{split}.json").write_text(
            json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    counts = manifest["counts"]
    print(f"Created frozen split at {output_dir}")
    print(f"dev={counts['dev_samples']} samples / {counts['dev_image_groups']} images")
    print(f"holdout={counts['holdout_samples']} samples / {counts['holdout_image_groups']} images")
    print(f"dataset_fingerprint={manifest['dataset_fingerprint']}")


if __name__ == "__main__":
    main()
