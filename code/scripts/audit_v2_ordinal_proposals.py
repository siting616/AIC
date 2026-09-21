"""Rank and visualize proposal-only V2 ordinal changes for human review."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_iou


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def image_path(images_dir: Path, scene: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = images_dir / "visible" / f"{scene}{suffix}"
        if path.exists():
            return path
    return None


def analyze(report: dict) -> list[dict]:
    rows = []
    for family in report["families"]:
        if not family.get("proposals"):
            continue
        proposals = family["proposals"]
        old_boxes = [item["v1_bbox"] for item in proposals if item.get("v1_bbox")]
        old_conflicts = sum(
            compute_iou(old_boxes[i], old_boxes[j]) >= 0.85
            for i in range(len(old_boxes)) for j in range(i + 1, len(old_boxes))
        )
        for item in proposals:
            if not item.get("changed"):
                continue
            old, new = item["v1_bbox"], item["v2_bbox"]
            old_area, new_area = area(old), area(new)
            ratio = new_area / max(old_area, 1e-12)
            shift = math.dist(center(old), center(new)) / math.sqrt(2)
            overlap = compute_iou(old, new)
            scale_consistency = family.get("selected_area_ratio")
            flags = []
            if ratio < 0.10 or ratio > 4.0:
                flags.append("extreme_area_change")
            elif ratio < 0.25 or ratio > 2.5:
                flags.append("large_area_change")
            if overlap < 0.02:
                flags.append("near_disjoint")
            if shift > 0.40:
                flags.append("large_center_shift")
            if scale_consistency is not None and scale_consistency > 4.0:
                flags.append("family_scale_spread")
            touches = sum(value <= .005 for value in (new[0], new[1])) + sum(value >= .995 for value in (new[2], new[3]))
            if touches:
                flags.append("touches_image_edge")

            # Priority rewards families that repair duplicate V1 answers and
            # have coherent new instance scales.  Risk never implies acceptance.
            review_score = 3.0 * old_conflicts
            review_score += 2.0 if scale_consistency is not None and scale_consistency <= 2.5 else 0.0
            review_score += 1.0 if 0.10 <= ratio <= 2.5 else 0.0
            review_score -= 1.5 * len(flags)
            risk = "high" if any(x in flags for x in ("extreme_area_change", "large_center_shift", "family_scale_spread")) else ("medium" if flags else "low")
            rows.append({
                **item,
                "scene_id": family["scene_id"],
                "target": family["target"],
                "direction": family["direction"],
                "family_query_count": len(family["query_ids"]),
                "family_old_conflict_pairs": old_conflicts,
                "family_area_ratio": scale_consistency,
                "old_new_iou": overlap,
                "new_old_area_ratio": ratio,
                "center_shift": shift,
                "risk": risk,
                "risk_flags": flags,
                "review_score": review_score,
            })
    return sorted(rows, key=lambda item: (-item["review_score"], item["risk"], item["query_id"]))


def draw_record(record: dict, source: Path, destination: Path) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    def pixels(box):
        return (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
    draw.rectangle(pixels(record["v1_bbox"]), outline=(255, 50, 50), width=max(3, width // 500))
    draw.rectangle(pixels(record["v2_bbox"]), outline=(50, 255, 80), width=max(3, width // 500))
    title = f"{record['query_id']} | {record['risk']} | old=RED new=GREEN | {record['query']}"
    draw.rectangle((0, 0, width, 32), fill=(0, 0, 0))
    draw.text((8, 8), title[:180], fill=(255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((960, 540))
    image.save(destination, quality=90)


def contact_sheets(paths: list[Path], output_dir: Path, batch_size: int = 20) -> None:
    for batch_index in range(0, len(paths), batch_size):
        batch = paths[batch_index:batch_index + batch_size]
        thumbs = []
        for path in batch:
            image = Image.open(path).convert("RGB")
            image.thumbnail((480, 270))
            thumbs.append(image.copy())
        if not thumbs:
            continue
        sheet = Image.new("RGB", (960, math.ceil(len(thumbs) / 2) * 270), "white")
        for index, image in enumerate(thumbs):
            sheet.paste(image, ((index % 2) * 480, (index // 2) * 270))
        sheet.save(output_dir / f"batch_{batch_index // batch_size + 1:02d}.jpg", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="outputs/v2/ordinal/merged/ordinal_merged_report.json")
    parser.add_argument("--images-dir", default="初赛数据集-基于大模型的多模态视觉理解与推理/Images")
    parser.add_argument("--output-dir", default="outputs/v2/ordinal/audit")
    parser.add_argument("--visualize", type=int, default=100)
    parser.add_argument("--risk", choices=("low", "medium", "high"), help="Visualize only one risk tier")
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    report = load(resolve(args.report))
    rows = analyze(report)
    visual_rows = [item for item in rows if not args.risk or item["risk"] == args.risk]
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "ranked_proposals.json", rows)
    fieldnames = [
        "query_id", "scene_id", "target", "ordinal_index", "risk", "review_score",
        "family_old_conflict_pairs", "family_area_ratio", "old_new_iou",
        "new_old_area_ratio", "center_shift", "risk_flags", "query",
    ]
    with (output / "ranked_proposals.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "risk_flags": "|".join(row["risk_flags"])})

    images_dir = resolve(args.images_dir)
    rendered = []
    missing = 0
    for rank, row in enumerate(visual_rows[:args.visualize], 1):
        source = image_path(images_dir, row["scene_id"])
        if source is None:
            missing += 1
            continue
        destination = output / "items" / f"{rank:04d}_{row['query_id']}.jpg"
        draw_record(row, source, destination)
        rendered.append(destination)
    contact_sheets(rendered, output)
    summary = {
        "proposals": len(rows),
        "risk_counts": dict(Counter(item["risk"] for item in rows)),
        "visualized": len(rendered),
        "missing_images": missing,
        "automatic_overrides": 0,
    }
    save(output / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
