import os
import gc
import json
import math
import re
import itertools
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/root/autodl-tmp/aic_v1_project")

DATA_ROOT = (
    PROJECT_ROOT
    / "data/official/初赛数据集-基于大模型的多模态视觉理解与推理"
)

RGB_PATH = DATA_ROOT / "Images/visible/000002.png"
IR_PATH = DATA_ROOT / "Images/infrared/000002.png"
DEPTH_PATH = DATA_ROOT / "Images/depth/000002.png"

INPUT_JSON = (
    PROJECT_ROOT
    / "outputs/stage1b_000002/stage1b_candidates.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs/stage2_000002"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "stage2_predictions.json"

HF_HUB = PROJECT_ROOT / "models/hf_cache/hub"

MAX_TARGETS_FOR_QWEN = 10
MAX_REFS_FOR_QWEN = 5
MAX_GROUP_BASE = 6


# ============================================================
# MODEL PATH
# ============================================================

def find_snapshot(repo):
    root = HF_HUB / repo / "snapshots"

    for p in root.iterdir():
        if p.is_dir() and (p / "config.json").exists():
            return str(p)

    raise RuntimeError(repo)


QWEN_PATH = find_snapshot(
    "models--Qwen--Qwen3-VL-8B-Instruct"
)


# ============================================================
# LOAD DATA
# ============================================================

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

rgb_cv = cv2.imread(str(RGB_PATH), cv2.IMREAD_COLOR)
ir_cv = cv2.imread(str(IR_PATH), cv2.IMREAD_UNCHANGED)
depth = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)

if rgb_cv is None or ir_cv is None or depth is None:
    raise RuntimeError("Failed to load multimodal images")

H, W = depth.shape[:2]

rgb_pil = Image.open(RGB_PATH).convert("RGB")
ir_pil = Image.open(IR_PATH).convert("RGB")


# ============================================================
# DEPTH VISUALIZATION
# ============================================================

valid = depth > 0

depth_vis = np.zeros((H, W), dtype=np.uint8)

if valid.any():
    vals = depth[valid].astype(np.float32)

    lo = np.percentile(vals, 2)
    hi = np.percentile(vals, 98)

    norm = (
        (depth.astype(np.float32) - lo)
        / max(hi - lo, 1.0)
    )

    norm = np.clip(norm, 0, 1)

    # Near = bright, far = dark for Qwen visualization
    depth_vis = ((1.0 - norm) * 255).astype(np.uint8)
    depth_vis[~valid] = 0

depth_pil = Image.fromarray(depth_vis).convert("RGB")


# ============================================================
# BASIC GEOMETRY
# ============================================================

def box_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def center(b):
    return (
        (b[0]+b[2])/2,
        (b[1]+b[3])/2,
    )


def intersect_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    return max(0, x2-x1) * max(0, y2-y1)


def iou(a, b):
    inter = intersect_area(a, b)

    union = box_area(a) + box_area(b) - inter

    return inter / union if union > 0 else 0.0


def horizontal_overlap(a, b):
    inter = max(
        0,
        min(a[2], b[2]) - max(a[0], b[0])
    )

    width = max(
        1,
        min(a[2]-a[0], b[2]-b[0])
    )

    return inter / width


def vertical_overlap(a, b):
    inter = max(
        0,
        min(a[3], b[3]) - max(a[1], b[1])
    )

    height = max(
        1,
        min(a[3]-a[1], b[3]-b[1])
    )

    return inter / height


def union_boxes(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


# ============================================================
# ROBUST ROI
# ============================================================

def inner_box(box, ratio=0.65):

    x1, y1, x2, y2 = map(float, box)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    w = (x2 - x1) * ratio
    h = (y2 - y1) * ratio

    return [
        max(0, cx-w/2),
        max(0, cy-h/2),
        min(W, cx+w/2),
        min(H, cy+h/2),
    ]


def roi_slice(arr, box):

    x1, y1, x2, y2 = inner_box(box)

    x1 = int(max(0, math.floor(x1)))
    y1 = int(max(0, math.floor(y1)))
    x2 = int(min(W, math.ceil(x2)))
    y2 = int(min(H, math.ceil(y2)))

    return arr[y1:y2, x1:x2]


def depth_stats(box):

    roi = roi_slice(depth, box)

    if roi.size == 0:
        return {
            "valid_ratio": 0.0,
            "median_mm": None,
            "p25_mm": None,
            "p75_mm": None,
        }

    mask = (roi > 300) & (roi <= 19999)

    vals = roi[mask]

    vr = float(vals.size / roi.size)

    if vals.size == 0:
        return {
            "valid_ratio": round(vr, 4),
            "median_mm": None,
            "p25_mm": None,
            "p75_mm": None,
        }

    return {
        "valid_ratio": round(vr, 4),
        "median_mm": int(np.median(vals)),
        "p25_mm": int(np.percentile(vals, 25)),
        "p75_mm": int(np.percentile(vals, 75)),
    }


def thermal_stats(box):

    roi = roi_slice(ir_cv, box)

    if roi.size == 0:
        return {
            "mean": None,
            "median": None,
        }

    if roi.ndim == 3:
        thermal = roi.astype(np.float32).mean(axis=2)
    else:
        thermal = roi.astype(np.float32)

    return {
        "mean": round(float(np.mean(thermal)), 2),
        "median": round(float(np.median(thermal)), 2),
    }


# ============================================================
# RELATION SCORES
# ============================================================

def relation_features(target_box, ref_box):

    tcx, tcy = center(target_box)
    rcx, rcy = center(ref_box)

    td = depth_stats(target_box)
    rd = depth_stats(ref_box)

    return {
        "target_left_of_ref": tcx < rcx,
        "target_right_of_ref": tcx > rcx,
        "target_above_ref": tcy < rcy,
        "target_below_ref": tcy > rcy,

        "horizontal_overlap": round(
            horizontal_overlap(target_box, ref_box),
            3,
        ),

        "vertical_overlap": round(
            vertical_overlap(target_box, ref_box),
            3,
        ),

        "center_distance_px": round(
            math.hypot(tcx-rcx, tcy-rcy),
            1,
        ),

        "target_depth_mm": td["median_mm"],
        "reference_depth_mm": rd["median_mm"],

        "target_closer_than_reference": (
            td["median_mm"] is not None
            and rd["median_mm"] is not None
            and td["median_mm"] < rd["median_mm"]
        ),
    }


# ============================================================
# CANDIDATE EVIDENCE
# ============================================================

def enrich_candidate(c):

    b = c["bbox"]

    cx, cy = center(b)

    result = dict(c)

    result["center_norm"] = [
        round(cx/W, 4),
        round(cy/H, 4),
    ]

    result["depth"] = depth_stats(b)
    result["thermal"] = thermal_stats(b)

    return result


# ============================================================
# COUNT / GROUP HYPOTHESES
# ============================================================

def build_groups(candidates, count):

    if count <= 1:
        return []

    base = candidates[:MAX_GROUP_BASE]

    groups = []

    # Avoid exploding combinations.
    if count > 3:
        return []

    for combo in itertools.combinations(base, count):

        # Reject almost-identical overlapping detections.
        bad = False

        for a, b in itertools.combinations(combo, 2):
            if iou(a["bbox"], b["bbox"]) > 0.35:
                bad = True
                break

        if bad:
            continue

        boxes = [x["bbox"] for x in combo]

        ub = union_boxes(boxes)

        score = sum(x["score"] for x in combo) / count

        groups.append({
            "candidate_id": "",
            "members": [
                x["candidate_id"]
                for x in combo
            ],
            "bbox": [
                round(x, 2)
                for x in ub
            ],
            "score": round(score, 6),
            "depth": depth_stats(ub),
            "thermal": thermal_stats(ub),
        })

    groups.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    groups = groups[:10]

    for i, g in enumerate(groups):
        g["candidate_id"] = f"G{i}"

    return groups


# ============================================================
# VISUALIZATION
# ============================================================

def draw_candidates(
    base_img,
    candidates,
    group_candidates=None,
):

    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    for c in candidates:

        x1, y1, x2, y2 = c["bbox"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=5,
        )

        draw.rectangle(
            [x1, max(0, y1-25), x1+70, y1],
            fill="white",
        )

        draw.text(
            (x1+3, max(0, y1-22)),
            c["candidate_id"],
            fill="black",
        )

    if group_candidates:

        for g in group_candidates[:4]:

            x1, y1, x2, y2 = g["bbox"]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="blue",
                width=5,
            )

            draw.text(
                (x1+5, y1+5),
                g["candidate_id"],
                fill="blue",
            )

    return img


def draw_references(base_img, references):

    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    for ri, ref in enumerate(references):

        for c in ref["candidates"][:MAX_REFS_FOR_QWEN]:

            x1, y1, x2, y2 = c["bbox"]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="green",
                width=4,
            )

            draw.text(
                (x1+3, y1+3),
                c["candidate_id"],
                fill="green",
            )

    return img


# ============================================================
# QWEN
# ============================================================

print("Loading Qwen3-VL:", QWEN_PATH)

processor = AutoProcessor.from_pretrained(
    QWEN_PATH,
    local_files_only=True,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

model.eval()


def extract_json(text):

    s = text.find("{")
    e = text.rfind("}")

    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            pass

    return None


def judge_query(
    qid,
    query,
    parsed,
    targets,
    groups,
    references,
):

    target_vis = draw_candidates(
        rgb_pil,
        targets,
        groups,
    )

    ref_vis = draw_references(
        rgb_pil,
        references,
    )

    target_path = OUTPUT_DIR / f"{qid}_targets.jpg"
    ref_path = OUTPUT_DIR / f"{qid}_references.jpg"

    target_vis.save(target_path, quality=95)
    ref_vis.save(ref_path, quality=95)

    evidence = {
        "query": query,
        "parsed": parsed,
        "target_candidates": targets,
        "group_candidates": groups,
        "reference_candidates": references,
    }

    prompt = f"""
You are the final candidate judge in a multimodal visual grounding system.

The required target is described by:

QUERY:
{query}

Structured parse:
{json.dumps(parsed, ensure_ascii=False)}

Candidate evidence:
{json.dumps(evidence, ensure_ascii=False)}

Images are provided in this order:

1. RGB scene with TARGET candidates in red.
   Group candidates, when present, are blue boxes.

2. RGB scene with REFERENCE-object candidates in green.

3. Original infrared image.

4. Depth visualization.
   IMPORTANT: in the numerical evidence, depth is in millimeters.
   Smaller depth means closer to camera.
   Use NUMERICAL depth evidence rather than guessing from visualization.

Your job:

- Select the candidate that best matches the full referring expression.
- Use appearance from RGB.
- Use infrared only when useful.
- Use numerical depth for front/behind/near/far reasoning.
- Use spatial constraints and reference objects.
- Apply ordinal constraints such as leftmost only AFTER semantic and relation filtering.
- If count > 1, prefer a GROUP candidate G* representing all requested objects.
- Do not prefer the highest DINO score automatically.
- Do not output coordinates yourself.
- Select only an existing T* or G* candidate.
- If none of the candidates plausibly localizes the target, return "NONE".

Return ONLY:

{{
  "selected_candidate_id": "T0 or G0 or NONE",
  "confidence": 0.0,
  "reason": "short explanation"
}}
""".strip()

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": str(target_path),
            },
            {
                "type": "image",
                "image": str(ref_path),
            },
            {
                "type": "image",
                "image": str(IR_PATH),
            },
            {
                "type": "image",
                "image": str(
                    OUTPUT_DIR / "depth_visualization.png"
                ),
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
        )

    n = inputs.input_ids.shape[1]

    answer = processor.batch_decode(
        generated[:, n:],
        skip_special_tokens=True,
    )[0]

    parsed_answer = extract_json(answer)

    return parsed_answer, answer


# Save depth visualization once
depth_pil.save(
    OUTPUT_DIR / "depth_visualization.png"
)


# ============================================================
# RUN
# ============================================================

results = {}


for qid, item in data.items():

    print("\n" + "="*80)
    print(qid)
    print(item["query"])

    parsed = item["parsed"]

    targets = [
        enrich_candidate(x)
        for x in item["target"]["candidates"][
            :MAX_TARGETS_FOR_QWEN
        ]
    ]

    count = int(parsed.get("count", 1) or 1)

    groups = build_groups(
        targets,
        count,
    )

    references = []

    for ref in item.get("references", []):

        enriched = []

        for c in ref["candidates"][:MAX_REFS_FOR_QWEN]:
            enriched.append(
                enrich_candidate(c)
            )

        # Add target/reference numerical relations
        for t in targets:
            t.setdefault(
                "relation_evidence",
                {}
            )

            t["relation_evidence"][
                ref["reference_object"]
            ] = []

            for r in enriched:

                t["relation_evidence"][
                    ref["reference_object"]
                ].append({
                    "reference_id":
                        r["candidate_id"],
                    "features":
                        relation_features(
                            t["bbox"],
                            r["bbox"],
                        )
                })

        references.append({
            "relation": ref["relation"],
            "reference_object":
                ref["reference_object"],
            "candidates": enriched,
        })

    print("Targets:", len(targets))
    print("Groups :", len(groups))
    print("Refs   :", len(references))

    selected, raw = judge_query(
        qid,
        item["query"],
        parsed,
        targets,
        groups,
        references,
    )

    print("Qwen:", raw)

    if selected is None:
        selected = {
            "selected_candidate_id": "NONE",
            "confidence": 0.0,
            "reason": "Failed to parse Qwen output",
        }

    candidate_id = selected.get(
        "selected_candidate_id",
        "NONE",
    )

    candidate_map = {
        c["candidate_id"]: c
        for c in targets
    }

    candidate_map.update({
        g["candidate_id"]: g
        for g in groups
    })

    chosen = candidate_map.get(
        candidate_id
    )

    if chosen is not None:

        bbox = chosen["bbox"]

        normalized = [
            round(bbox[0] / W, 6),
            round(bbox[1] / H, 6),
            round(bbox[2] / W, 6),
            round(bbox[3] / H, 6),
        ]

    else:
        bbox = None
        normalized = None

    results[qid] = {
        "query": item["query"],
        "parsed": parsed,
        "decision": selected,
        "bbox": bbox,
        "bbox_normalized": normalized,
        "targets": targets,
        "groups": groups,
        "references": references,
        "qwen_raw": raw,
    }

    print("SELECTED:", candidate_id)
    print("BBOX:", bbox)


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n" + "="*80)
print("STAGE 2 COMPLETE")
print("="*80)
print(OUTPUT_JSON)

print(
    "Peak CUDA allocated:",
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        2,
    ),
    "GB"
)
