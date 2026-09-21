"""Prepare an incremental target-localization batch and visual contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def get_font(size: int):
    for name in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def draw_box(draw, box, width, height, color):
    xy = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
    draw.rectangle(xy, outline=color, width=max(3, round(min(width, height) / 180)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--eligible", required=True)
    parser.add_argument("--used-plan", required=True, nargs="+")
    parser.add_argument("--start", type=int, default=50)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline, eligible = load(args.baseline), load(args.eligible)
    used = [item for path in args.used_plan for item in load(path)]
    used_ids = {item["sample_id"] for item in used}
    remaining = [item for item in eligible if item["sample_id"] not in used_ids]
    batch = remaining[args.start - len(used): args.start - len(used) + args.count]
    # If start refers to the original ranked list, this evaluates ranks 51 onward.
    if args.start == len(used):
        batch = remaining[:args.count]

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    if not images_dir.is_absolute():
        images_dir = PROJECT_ROOT / images_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_batch.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    thumb_w, thumb_h, caption_h = 560, 360, 92
    sheets = []
    for offset in range(0, len(batch), 10):
        subset = batch[offset:offset + 10]
        sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + caption_h) * 5), "white")
        for local_index, item in enumerate(subset):
            sample_id = item["sample_id"]
            image_rel = baseline[sample_id]["visible"]
            image = Image.open(images_dir / Path(image_rel).name).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            panel = Image.new("RGB", (thumb_w, thumb_h), (238, 238, 238))
            panel.paste(image, ((thumb_w-image.width)//2, (thumb_h-image.height)//2))
            draw = ImageDraw.Draw(panel)
            xoff, yoff = (thumb_w-image.width)//2, (thumb_h-image.height)//2
            def shifted(box):
                return [(box[0]*image.width+xoff)/thumb_w, (box[1]*image.height+yoff)/thumb_h,
                        (box[2]*image.width+xoff)/thumb_w, (box[3]*image.height+yoff)/thumb_h]
            draw_box(draw, shifted(baseline[sample_id]["bbox"]), thumb_w, thumb_h, "red")
            draw_box(draw, shifted(item["after"]), thumb_w, thumb_h, "lime")
            x = (local_index % 2) * thumb_w
            y = (local_index // 2) * (thumb_h + caption_h)
            sheet.paste(panel, (x, y))
            text = f"#{args.start + offset + local_index + 1} {sample_id} rank={item['candidate_rank']} score={item['candidate_score']:.3f}\n{item['query']}"
            ImageDraw.Draw(sheet).multiline_text((x+7, y+thumb_h+5), text, fill="black", font=get_font(16), spacing=3)
        name = f"batch_{offset//10+1}.jpg"
        sheet.save(output_dir / name, quality=93)
        sheets.append(name)
    manifest = {"baseline": args.baseline, "candidate_count": len(batch), "rank_start": args.start + 1,
                "rank_end": args.start + len(batch), "red": "current baseline", "green": "proposed", "sheets": sheets}
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
