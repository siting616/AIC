"""Convert REFER/RefCOCO annotations to the AIC competition JSON shape.

REFER datasets are RGB-only. The RGB path is repeated in infrared/depth for
loader compatibility and records are marked ``source_modalities: [visible]``.
Those placeholders must not be used to evaluate multimodal fusion.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path


def load_refs(path: Path):
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("rb") as stream:
            try:
                payload = pickle.load(stream)
            except UnicodeDecodeError:
                stream.seek(0)
                payload = pickle.load(stream, encoding="latin1")
    if isinstance(payload, dict) and "refs" in payload:
        payload = payload["refs"]
    if not isinstance(payload, list):
        raise ValueError("REFER refs file must contain a list")
    return payload


def normalize_xywh(bbox, width, height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid COCO bbox: {bbox!r}")
    x, y, w, h = map(float, bbox)
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size or bbox: size=({width}, {height}), bbox={bbox}")
    result = [x / width, y / height, (x + w) / width, (y + h) / height]
    result = [max(0.0, min(1.0, value)) for value in result]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f"BBox becomes empty after normalization: {bbox}")
    return result


def _sentence_text(sentence):
    if isinstance(sentence, str):
        return sentence.strip()
    if isinstance(sentence, dict):
        return str(sentence.get("sent") or sentence.get("raw") or "").strip()
    return ""


def _path_value(image_path, output_path, absolute_paths):
    if absolute_paths:
        return str(image_path.resolve())
    return os.path.relpath(image_path.resolve(), output_path.parent.resolve()).replace("\\", "/")


def convert(instances, refs, images_dir, output_path, dataset_name, splits, absolute_paths=False):
    images = {item["id"]: item for item in instances.get("images", [])}
    annotations = {item["id"]: item for item in instances.get("annotations", [])}
    requested = set(splits)
    converted = {}
    skipped_missing = 0
    for ref in refs:
        if "all" not in requested and str(ref.get("split", "")) not in requested:
            continue
        image = images.get(ref.get("image_id"))
        annotation = annotations.get(ref.get("ann_id"))
        if image is None or annotation is None:
            skipped_missing += 1
            continue
        image_path = Path(images_dir) / image["file_name"]
        path_value = _path_value(image_path, Path(output_path), absolute_paths)
        bbox = normalize_xywh(annotation["bbox"], float(image["width"]), float(image["height"]))
        for index, sentence in enumerate(ref.get("sentences") or [], 1):
            query = _sentence_text(sentence)
            if not query:
                continue
            sent_id = sentence.get("sent_id", sentence.get("id", index)) if isinstance(sentence, dict) else index
            sample_id = f"{dataset_name}_{ref.get('ref_id', ref.get('ann_id'))}_{sent_id}"
            converted[sample_id] = {
                "visible": path_value, "infrared": path_value, "depth": path_value,
                "query": query, "bbox": bbox,
                "source_dataset": dataset_name,
                "source_split": ref.get("split"),
                "source_image_id": ref.get("image_id"),
                "source_ref_id": ref.get("ref_id"),
                "source_modalities": ["visible"],
            }
    return converted, skipped_missing


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--split", nargs="+", default=["val"])
    parser.add_argument("--absolute-paths", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    instances = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    converted, skipped = convert(
        instances, load_refs(Path(args.refs)), Path(args.images_dir), output,
        args.dataset_name, args.split, args.absolute_paths,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = len({item["source_image_id"] for item in converted.values()})
    print(f"Converted samples: {len(converted)}")
    print(f"Unique scenes: {scenes}")
    print(f"Skipped missing refs: {skipped}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
