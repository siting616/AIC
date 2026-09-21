"""CPU-only candidate color features and conservative color routing."""

from __future__ import annotations

import colorsys
import re
from pathlib import Path

from PIL import Image


COLOR_PATTERN = re.compile(
    r"\b(red|blue|green|yellow|white|black|pink|purple|brown|gray|grey|orange)\b",
    re.I,
)
SAFE_COLORS = {"black", "blue", "yellow", "purple", "brown", "gray", "orange"}


def query_color(query):
    match = COLOR_PATTERN.search(str(query))
    if not match:
        return None
    color = match.group(1).lower()
    return "gray" if color == "grey" else color


def pixel_matches_color(color, red, green, blue):
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
    hue *= 360
    if color == "black":
        return value < 0.22
    if color == "white":
        return saturation < 0.18 and value > 0.78
    if color == "gray":
        return saturation < 0.20 and 0.22 <= value <= 0.78
    if color == "red":
        return (hue < 15 or hue >= 345) and saturation > 0.35 and value > 0.22
    if color == "orange":
        return 15 <= hue < 42 and saturation > 0.35 and value > 0.35
    if color == "yellow":
        return 42 <= hue < 75 and saturation > 0.30 and value > 0.35
    if color == "green":
        return 70 <= hue < 175 and saturation > 0.25 and value > 0.18
    if color == "blue":
        return 175 <= hue < 260 and saturation > 0.25 and value > 0.18
    if color == "purple":
        return 260 <= hue < 315 and saturation > 0.22 and value > 0.18
    if color == "pink":
        return 315 <= hue < 345 and saturation > 0.18 and value > 0.50
    if color == "brown":
        return 15 <= hue < 50 and saturation > 0.25 and 0.15 < value < 0.68
    return False


def candidate_color_fraction(image, bbox, color, sample_size=32):
    width, height = image.size
    x1, y1, x2, y2 = bbox
    bounds = (
        max(0, min(width - 1, int(x1 * width))),
        max(0, min(height - 1, int(y1 * height))),
        max(1, min(width, int(x2 * width))),
        max(1, min(height, int(y2 * height))),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return 0.0
    crop = image.crop(bounds).resize((sample_size, sample_size)).convert("RGB")
    pixels = list(crop.getdata())
    return sum(pixel_matches_color(color, *pixel) for pixel in pixels) / len(pixels)


def build_color_feature_cache(annotations, candidate_records, sample_size=32):
    records = {}
    images = {}
    for sample_id, annotation in annotations.items():
        color = query_color(annotation.get("query", ""))
        candidates = candidate_records.get(sample_id, {}).get("candidates", [])
        if not color:
            continue
        image_path = str(Path(annotation["visible"]))
        image = images.get(image_path)
        if image is None:
            image = Image.open(image_path).convert("RGB")
            images[image_path] = image
        records[sample_id] = {
            "color": color,
            "fractions": [
                candidate_color_fraction(image, item["bbox"], color, sample_size)
                for item in candidates
            ],
        }
    return {
        "schema_version": 1,
        "feature": "hsv_query_color_fraction",
        "sample_size": sample_size,
        "records": records,
    }


def apply_color_router(annotations, cache_records, color_features, base_records,
                       safe_colors=SAFE_COLORS, minimum_fraction=0.10, minimum_margin=0.08):
    output = {}
    stats = {"samples": len(annotations), "color_queries": 0, "safe_color_queries": 0, "changed": 0}
    for sample_id, annotation in annotations.items():
        base = base_records[sample_id]
        ranked = [dict(item) for item in base.get("candidates", [])]
        feature = color_features.get(sample_id)
        color = feature.get("color") if feature else None
        if color:
            stats["color_queries"] += 1
        if color in safe_colors and ranked:
            stats["safe_color_queries"] += 1
            raw = cache_records.get(sample_id, {}).get("candidates", [])
            fractions = feature["fractions"]
            best_index = max(range(len(fractions)), key=lambda index: fractions[index])
            base_bbox = ranked[0]["bbox"]
            base_index = next((i for i, item in enumerate(raw) if item["bbox"] == base_bbox), None)
            base_fraction = fractions[base_index] if base_index is not None else 0.0
            if (fractions[best_index] >= minimum_fraction and
                    fractions[best_index] - base_fraction >= minimum_margin):
                selected_bbox = raw[best_index]["bbox"]
                selected_index = next((i for i, item in enumerate(ranked) if item["bbox"] == selected_bbox), None)
                if selected_index is not None and selected_index != 0:
                    selected = ranked.pop(selected_index)
                    selected["color_route"] = color
                    selected["color_fraction"] = fractions[best_index]
                    ranked.insert(0, selected)
                    stats["changed"] += 1
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        output[sample_id] = {"bbox": ranked[0]["bbox"] if ranked else base.get("bbox"), "candidates": ranked}
    return output, stats
