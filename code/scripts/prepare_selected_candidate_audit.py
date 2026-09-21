"""Prepare contact sheets for a selected ordered list of candidate ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path: str):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


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
    parser.add_argument("--ids-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    baseline = load(args.baseline)
    eligible = {item["sample_id"]: item for item in load(args.eligible)}
    ids = [sample_id for sample_id in load(args.ids_json) if sample_id in eligible]
    batch = [eligible[sample_id] for sample_id in ids[: args.count]]
    images_dir = resolve(args.images_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_batch.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    thumb_w, thumb_h, caption_h = 560, 360, 92
    sheets = []
    for offset in range(0, len(batch), 10):
        subset = batch[offset : offset + 10]
        sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + caption_h) * 5), "white")
        for local_index, item in enumerate(subset):
            sample_id = item["sample_id"]
            image_rel = baseline[sample_id]["visible"]
            image = Image.open(images_dir / Path(image_rel).name).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            panel = Image.new("RGB", (thumb_w, thumb_h), (238, 238, 238))
            panel.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
            draw = ImageDraw.Draw(panel)
            xoff, yoff = (thumb_w - image.width) // 2, (thumb_h - image.height) // 2

            def shifted(box):
                return [
                    (box[0] * image.width + xoff) / thumb_w,
                    (box[1] * image.height + yoff) / thumb_h,
                    (box[2] * image.width + xoff) / thumb_w,
                    (box[3] * image.height + yoff) / thumb_h,
                ]

            draw_box(draw, shifted(baseline[sample_id]["bbox"]), thumb_w, thumb_h, "red")
            draw_box(draw, shifted(item["after"]), thumb_w, thumb_h, "lime")
            x = (local_index % 2) * thumb_w
            y = (local_index // 2) * (thumb_h + caption_h)
            sheet.paste(panel, (x, y))
            text = (
                f"#{offset + local_index + 1} {sample_id} "
                f"rank={item['candidate_rank']} score={item['candidate_score']:.3f}\n"
                f"{item['query']}"
            )
            ImageDraw.Draw(sheet).multiline_text(
                (x + 7, y + thumb_h + 5),
                text,
                fill="black",
                font=get_font(16),
                spacing=3,
            )
        name = f"batch_{offset // 10 + 1}.jpg"
        sheet.save(output_dir / name, quality=93)
        sheets.append(name)

    manifest = {
        "baseline": args.baseline,
        "ids_json": args.ids_json,
        "candidate_count": len(batch),
        "red": "current baseline",
        "green": "proposed",
        "sheets": sheets,
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
