"""Render side-by-side V28 versus V31 group audit images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = [(239, 68, 68), (37, 99, 235), (22, 163, 74), (217, 119, 6), (147, 51, 234)]


def font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_panel(image, ids, boxes, queries, title):
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    width, height = panel.size
    line_width = max(3, round(min(width, height) / 220))
    for index, qid in enumerate(ids):
        box = boxes[qid]
        color = COLORS[index % len(COLORS)]
        xy = [round(box[0]*width), round(box[1]*height), round(box[2]*width), round(box[3]*height)]
        draw.rectangle(xy, outline=color, width=line_width)
        draw.rectangle([xy[0], max(0, xy[1]-25), xy[0]+72, xy[1]], fill=color)
        draw.text((xy[0]+4, max(1, xy[1]-23)), qid[-3:], fill="white", font=font(17))
    header = Image.new("RGB", (width, 42 + 25*len(ids)), "white")
    hdraw = ImageDraw.Draw(header)
    hdraw.text((8, 6), title, fill="black", font=font(24))
    for index, qid in enumerate(ids):
        color = COLORS[index % len(COLORS)]
        y = 40 + index*25
        hdraw.rectangle([8, y, 24, y+16], fill=color)
        hdraw.text((30, y-3), f"{qid}: {queries[qid][:95]}", fill="black", font=font(16))
    canvas = Image.new("RGB", (width, header.height + height), "white")
    canvas.paste(header, (0, 0)); canvas.paste(panel, (0, header.height))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="outputs/experiments/v31_change_plan/accepted_groups.json")
    parser.add_argument("--changes", default="outputs/experiments/v31_change_plan/changes.json")
    parser.add_argument("--v28", default="outputs/experiments/joint_assignment_v31/v28/prediction.json")
    parser.add_argument("--images-dir", default="初赛数据集-基于大模型的多模态视觉理解与推理/Images/visible")
    parser.add_argument("--output-dir", default="outputs/experiments/v31_change_plan/audit")
    args = parser.parse_args()
    resolve = lambda x: Path(x) if Path(x).is_absolute() else PROJECT_ROOT / x
    load = lambda x: json.loads(resolve(x).read_text(encoding="utf-8"))
    groups, changes, v28 = load(args.groups), load(args.changes), load(args.v28)
    images_dir, output = resolve(args.images_dir), resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    html = ["<html><meta charset='utf-8'><body><h1>V31 visual audit</h1>"]
    for index, group in enumerate(groups, 1):
        ids = group["query_ids"]
        queries = {qid: v28[qid]["query"] for qid in ids}
        old_boxes = {qid: v28[qid]["bbox"] for qid in ids}
        new_boxes = {qid: changes.get(qid, {}).get("after", old_boxes[qid]) for qid in ids}
        image_path = images_dir / Path(group["image_id"]).name
        image = Image.open(image_path).convert("RGB")
        if image.width > 900:
            ratio = 900 / image.width
            image = image.resize((900, round(image.height*ratio)))
        left = draw_panel(image, ids, old_boxes, queries, "V28 BEFORE")
        right = draw_panel(image, ids, new_boxes, queries, "V31 PROPOSED")
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (230, 230, 230))
        canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
        name = f"{index:03d}_{Path(group['image_id']).stem}_{group['object_class']}.jpg"
        canvas.save(output / name, quality=92)
        manifest.append({"index": index, "image": name, "image_id": group["image_id"],
                         "object_class": group["object_class"], "query_ids": ids,
                         "changed_query_ids": group["changed_query_ids"], "decision": "pending",
                         "notes": ""})
        html.append(f"<h2>{index:03d} {group['image_id']} ({group['object_class']})</h2><img src='{name}' style='max-width:100%'>")
    html.append("</body></html>")
    (output / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"Rendered {len(manifest)} audit groups: {output}")


if __name__ == "__main__":
    main()
