#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug repeated-instance recall on scene 000023.

Goal:
Test whether Grounding DINO can recall the individual round stone objects when
"stone pier" is expanded to more detector-friendly visual synonyms.

This script does NOT modify any submission.
It only writes diagnostic overlays/crops/JSON.
"""

import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROJECT_ROOT = Path("/root/autodl-tmp/aic_v1_project")
DATA_ROOT = PROJECT_ROOT / "data/official/初赛数据集-基于大模型的多模态视觉理解与推理"
RGB_PATH = DATA_ROOT / "Images/visible/000023.png"
HF_HUB = PROJECT_ROOT / "models/hf_cache/hub"
OUT_DIR = PROJECT_ROOT / "outputs/debug_000023_alias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "stone pier",
    "stone bollard",
    "round stone bollard",
    "spherical stone bollard",
    "stone sphere",
    "stone ball",
    "round concrete bollard",
    "concrete sphere",
    "concrete bollard",
    "stone post",
    "bollard",
]

THRESHOLD = 0.04
TEXT_THRESHOLD = 0.04
BATCH = 2
STRIP_OVERLAP = 0.18


def find_snapshot(repo_dir_name: str) -> str:
    root = HF_HUB / repo_dir_name / "snapshots"
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
    raise RuntimeError(f"No valid snapshot under {root}")


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    x = inter(a, b)
    u = area(a) + area(b) - x
    return x / u if u > 0 else 0.0


def containment(a, b):
    x = inter(a, b)
    m = min(area(a), area(b))
    return x / m if m > 0 else 0.0


def clamp(b, W, H):
    x1, y1, x2, y2 = map(float, b)
    return [
        max(0.0, min(W, x1)),
        max(0.0, min(H, y1)),
        max(0.0, min(W, x2)),
        max(0.0, min(H, y2)),
    ]


DINO_PATH = find_snapshot("models--IDEA-Research--grounding-dino-base")
print("Loading DINO:", DINO_PATH)

processor = AutoProcessor.from_pretrained(DINO_PATH, local_files_only=True)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_PATH, local_files_only=True
).to("cuda")
model.eval()

img = Image.open(RGB_PATH).convert("RGB")
W, H = img.size


def detect(image, prompts, offset=(0, 0), source="full"):
    out = []
    ox, oy = offset
    w, h = image.size

    for s in range(0, len(prompts), BATCH):
        batch = prompts[s:s+BATCH]
        texts = [p + "." for p in batch]
        images = [image] * len(batch)

        inputs = processor(
            images=images, text=texts, return_tensors="pt", padding=True
        )
        inputs = {
            k: v.to("cuda") if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(h, w)] * len(batch),
        )

        for prompt, result in zip(batch, results):
            boxes = result["boxes"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()
            for b, score in zip(boxes, scores):
                gb = clamp(
                    [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy],
                    W, H
                )
                if area(gb) < 16:
                    continue
                out.append({
                    "bbox": [round(x, 2) for x in gb],
                    "score": round(float(score), 6),
                    "prompt": prompt,
                    "source": source,
                })

    return out


all_candidates = []

# Full image.
all_candidates += detect(img, PROMPTS, source="full")

# 5 vertical strips for better horizontal repeated-instance recall.
n = 5
base = W / n
pad = base * STRIP_OVERLAP

for i in range(n):
    x1 = int(max(0, i * base - pad))
    x2 = int(min(W, (i + 1) * base + pad))
    crop = img.crop((x1, 0, x2, H))
    all_candidates += detect(
        crop, PROMPTS, offset=(x1, 0), source=f"vstrip{i}"
    )

# 2x2 tiles.
half_w, half_h = W / 2, H / 2
pw = half_w * 0.15
ph = half_h * 0.15
tiles = [
    (0, 0, int(half_w + pw), int(half_h + ph)),
    (int(half_w - pw), 0, W, int(half_h + ph)),
    (0, int(half_h - ph), int(half_w + pw), H),
    (int(half_w - pw), int(half_h - ph), W, H),
]
for i, (x1, y1, x2, y2) in enumerate(tiles):
    crop = img.crop((x1, y1, x2, y2))
    all_candidates += detect(
        crop, PROMPTS, offset=(x1, y1), source=f"tile{i}"
    )

# Sort then conservative duplicate merge.
all_candidates.sort(key=lambda x: x["score"], reverse=True)
kept = []

for c in all_candidates:
    merged = None
    for k in kept:
        if iou(c["bbox"], k["bbox"]) >= 0.68 or containment(c["bbox"], k["bbox"]) >= 0.90:
            merged = k
            break
    if merged is None:
        c["matched_prompts"] = [c["prompt"]]
        c["sources"] = [c["source"]]
        kept.append(c)
    else:
        if c["prompt"] not in merged["matched_prompts"]:
            merged["matched_prompts"].append(c["prompt"])
        if c["source"] not in merged["sources"]:
            merged["sources"].append(c["source"])

# Prefer candidate boxes that could plausibly be one repeated object.
# Keep broad boxes too in JSON, but make a "small" diagnostic sheet.
for i, c in enumerate(kept):
    c["candidate_id"] = f"C{i}"
    c["area_ratio"] = round(area(c["bbox"]) / (W * H), 5)

json_path = OUT_DIR / "candidates.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(kept, f, ensure_ascii=False, indent=2)

# Overlay of top 40.
overlay = img.copy()
draw = ImageDraw.Draw(overlay)

for c in kept[:40]:
    x1, y1, x2, y2 = c["bbox"]
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    label = f'{c["candidate_id"]} {c["score"]:.2f}'
    draw.rectangle([x1, max(0, y1-22), x1+92, y1], fill="white")
    draw.text((x1+2, max(0, y1-19)), label, fill="black")

overlay_path = OUT_DIR / "overlay_top40.jpg"
overlay.save(overlay_path, quality=92)

# Crop sheet: prioritize single-object-sized boxes.
small = [
    c for c in kept
    if 0.0003 <= c["area_ratio"] <= 0.08
][:32]

cols = 4
tw, th = 270, 210
rows = max(1, math.ceil(len(small) / cols))
sheet = Image.new("RGB", (cols * tw, rows * th), "white")

for i, c in enumerate(small):
    x1, y1, x2, y2 = map(int, c["bbox"])
    crop = img.crop((x1, y1, x2, y2))
    crop.thumbnail((tw-18, th-55))
    tile = Image.new("RGB", (tw, th), "white")
    tile.paste(crop, ((tw-crop.width)//2, 48))
    d = ImageDraw.Draw(tile)
    d.text((7, 5), f'{c["candidate_id"]} score={c["score"]:.3f}', fill="black")
    d.text((7, 22), f'area={c["area_ratio"]:.4f} {c["prompt"]}', fill="black")
    sheet.paste(tile, ((i % cols)*tw, (i // cols)*th))

sheet_path = OUT_DIR / "small_candidate_crops.jpg"
sheet.save(sheet_path, quality=92)

print("=" * 80)
print("DONE")
print("Raw detections :", len(all_candidates))
print("Merged boxes   :", len(kept))
print("Small boxes    :", len(small))
print("JSON           :", json_path)
print("Overlay        :", overlay_path)
print("Crop sheet     :", sheet_path)

print("\nTop detector-friendly prompts among small boxes:")
counts = {}
for c in small:
    counts[c["prompt"]] = counts.get(c["prompt"], 0) + 1
for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
    print(f"{k:28s} {v}")
