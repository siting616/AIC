import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import (
    GroundingDinoProcessor,
    GroundingDinoForObjectDetection,
)

PROJECT_ROOT = Path("/root/autodl-tmp/aic_v1_project")

DATA_ROOT = (
    PROJECT_ROOT
    / "data/official/初赛数据集-基于大模型的多模态视觉理解与推理"
)

RGB_PATH = DATA_ROOT / "Images/visible/000002.png"

PARSED_PATH = (
    PROJECT_ROOT
    / "outputs/stage1_000002/parsed_queries_v2.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/stage1b_000002"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_HUB = PROJECT_ROOT / "models/hf_cache/hub"

BOX_THRESHOLD = 0.12
TEXT_THRESHOLD = 0.12

TARGET_TOPK = 15
REFERENCE_TOPK = 10


# ============================================================
# Model
# ============================================================

def find_snapshot(repo_folder):
    root = HF_HUB / repo_folder / "snapshots"

    for p in root.iterdir():
        if p.is_dir() and (p / "config.json").exists():
            return str(p)

    raise RuntimeError(f"No snapshot: {repo_folder}")


DINO_PATH = find_snapshot(
    "models--IDEA-Research--grounding-dino-base"
)

print("DINO:", DINO_PATH)

processor = GroundingDinoProcessor.from_pretrained(
    DINO_PATH,
    local_files_only=True,
)

model = GroundingDinoForObjectDetection.from_pretrained(
    DINO_PATH,
    local_files_only=True,
).cuda()

model.eval()


# ============================================================
# Data
# ============================================================

rgb = Image.open(RGB_PATH).convert("RGB")
W, H = rgb.size

with open(PARSED_PATH, "r", encoding="utf-8") as f:
    parsed_queries = json.load(f)


# ============================================================
# Utilities
# ============================================================

def prompt_clean(x):
    x = str(x).strip()

    if not x:
        return None

    if not x.endswith("."):
        x += "."

    return x


def unique_prompts(prompts):
    result = []
    seen = set()

    for p in prompts:
        p = prompt_clean(p)

        if not p:
            continue

        key = p.lower()

        if key not in seen:
            result.append(p)
            seen.add(key)

    return result


def expand_target_prompts(parsed, query):
    """
    Generic fallback prompt expansion.

    This does NOT perform spatial reasoning.
    It only gives DINO more visually-groundable noun phrases.
    """

    prompts = list(parsed.get("target_dino_prompts", []))

    target = parsed.get("target_class", "").lower()
    q = query.lower()

    # Abstract architectural target fallback
    if target == "passage" or "passage" in target:
        prompts += [
            "passageway",
            "entrance",
            "doorway",
            "corridor",
        ]

        if "upper level" in q:
            prompts += [
                "stairway",
                "stairs",
                "staircase",
                "upper level entrance",
            ]

    if target == "menu":
        prompts += [
            "menu board",
            "sign board",
        ]

    if "sign" in target:
        prompts += [
            "advertisement sign",
            "poster",
        ]

    if "camera" in target:
        prompts += [
            "surveillance camera",
            "security camera",
        ]

    return unique_prompts(prompts)


def expand_reference_prompts(constraint):
    prompts = list(
        constraint.get("reference_dino_prompts", [])
    )

    ref = constraint.get(
        "reference_object",
        "",
    ).lower()

    if "street lamp" in ref:
        prompts += [
            "lamp post",
            "street light",
        ]

    if "pillar" in ref:
        prompts += [
            "column",
        ]

    if "awning" in ref:
        prompts += [
            "canopy",
        ]

    if "dining barrier" in ref:
        prompts += [
            "barrier",
            "fence",
        ]

    if "entrance door" in ref:
        prompts += [
            "door",
            "glass door",
        ]

    return unique_prompts(prompts)


def clip_box(box):
    x1, y1, x2, y2 = box

    x1 = max(0.0, min(float(W), float(x1)))
    y1 = max(0.0, min(float(H), float(y1)))
    x2 = max(0.0, min(float(W), float(x2)))
    y2 = max(0.0, min(float(H), float(y2)))

    return [x1, y1, x2, y2]


def area(box):
    x1, y1, x2, y2 = box

    return (
        max(0.0, x2 - x1)
        *
        max(0.0, y2 - y1)
    )


def intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    return (
        max(0.0, x2 - x1)
        *
        max(0.0, y2 - y1)
    )


def iou(a, b):
    inter = intersection(a, b)

    union = area(a) + area(b) - inter

    if union <= 0:
        return 0.0

    return inter / union


def containment(a, b):
    """
    Intersection / area of smaller box.

    Useful for removing nested duplicate DINO boxes.
    """

    inter = intersection(a, b)

    smaller = min(
        area(a),
        area(b),
    )

    if smaller <= 0:
        return 0.0

    return inter / smaller


def merge_candidates(
    candidates,
    topk,
):
    """
    Merge detections from multiple prompts.

    Duplicate if:
      IoU >= 0.65
    OR
      one box is almost contained in another.
    """

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True,
    )

    kept = []

    for cand in candidates:

        duplicate = None

        for old in kept:

            if (
                iou(cand["bbox"], old["bbox"]) >= 0.65
                or
                containment(
                    cand["bbox"],
                    old["bbox"],
                ) >= 0.92
            ):
                duplicate = old
                break

        if duplicate is not None:

            if (
                cand["prompt"]
                not in duplicate["matched_prompts"]
            ):
                duplicate["matched_prompts"].append(
                    cand["prompt"]
                )

            continue

        cand["matched_prompts"] = [
            cand["prompt"]
        ]

        kept.append(cand)

        if len(kept) >= topk:
            break

    return kept


# ============================================================
# DINO
# ============================================================

def run_dino(prompt):

    inputs = processor(
        images=rgb,
        text=prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(H, W)],
    )[0]

    boxes = (
        result["boxes"]
        .detach()
        .cpu()
        .tolist()
    )

    scores = (
        result["scores"]
        .detach()
        .cpu()
        .tolist()
    )

    candidates = []

    for box, score in zip(boxes, scores):

        box = clip_box(box)

        x1, y1, x2, y2 = box

        if x2 <= x1 or y2 <= y1:
            continue

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        candidate = {
            "bbox": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
            ],

            "bbox_normalized": [
                round(x1 / W, 6),
                round(y1 / H, 6),
                round(x2 / W, 6),
                round(y2 / H, 6),
            ],

            "center": [
                round(cx, 2),
                round(cy, 2),
            ],

            "area_ratio": round(
                area(box) / (W * H),
                6,
            ),

            "score": round(
                float(score),
                6,
            ),

            "prompt": prompt,
        }

        candidates.append(candidate)

    return candidates


def detect_prompt_set(
    prompts,
    topk,
):

    raw = []

    for prompt in prompts:

        found = run_dino(prompt)

        print(
            f'    "{prompt}" -> '
            f'{len(found)} raw'
        )

        raw.extend(found)

    return merge_candidates(
        raw,
        topk,
    )


# ============================================================
# Run
# ============================================================

all_results = {}


for qid, info in parsed_queries.items():

    query = info["query"]
    parsed = info["parsed"]

    print("\n" + "=" * 80)
    print(qid)
    print(query)

    # --------------------------------------------------------
    # Target detection
    # --------------------------------------------------------

    target_prompts = expand_target_prompts(
        parsed,
        query,
    )

    print("\nTARGET PROMPTS:")

    for p in target_prompts:
        print(" ", p)

    print("\nTARGET DETECTION:")

    target_candidates = detect_prompt_set(
        target_prompts,
        TARGET_TOPK,
    )

    for i, c in enumerate(target_candidates):
        c["candidate_id"] = f"T{i}"

    # --------------------------------------------------------
    # Reference object detection
    # --------------------------------------------------------

    reference_results = []

    constraints = parsed.get(
        "constraints",
        [],
    )

    for ci, constraint in enumerate(constraints):

        ref_prompts = expand_reference_prompts(
            constraint
        )

        print(
            f"\nREFERENCE {ci}:",
            constraint["reference_object"],
        )

        print(
            "RELATION:",
            constraint["relation"],
        )

        for p in ref_prompts:
            print(" ", p)

        ref_candidates = detect_prompt_set(
            ref_prompts,
            REFERENCE_TOPK,
        )

        for i, c in enumerate(ref_candidates):
            c["candidate_id"] = f"R{ci}_{i}"

        reference_results.append({
            "constraint_index": ci,
            "relation": constraint["relation"],
            "reference_object": constraint[
                "reference_object"
            ],
            "prompts": ref_prompts,
            "candidates": ref_candidates,
        })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = {
        "query": query,
        "parsed": parsed,

        "target": {
            "prompts": target_prompts,
            "candidates": target_candidates,
        },

        "references": reference_results,
    }

    all_results[qid] = result

    print(
        "\nTARGET CANDIDATES:",
        len(target_candidates),
    )

    for c in target_candidates:
        print(
            c["candidate_id"],
            "score=",
            c["score"],
            "bbox=",
            c["bbox"],
            "area=",
            c["area_ratio"],
            "prompts=",
            c["matched_prompts"],
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    vis = rgb.copy()
    draw = ImageDraw.Draw(vis)

    # Targets
    for c in target_candidates:

        x1, y1, x2, y2 = c["bbox"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0),
            width=4,
        )

        draw.text(
            (x1 + 3, max(0, y1 - 16)),
            c["candidate_id"],
            fill=(255, 0, 0),
        )

    # References
    for ref in reference_results:

        for c in ref["candidates"]:

            x1, y1, x2, y2 = c["bbox"]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(0, 255, 0),
                width=3,
            )

            draw.text(
                (x1 + 3, y1 + 3),
                c["candidate_id"],
                fill=(0, 255, 0),
            )

    vis.save(
        OUTPUT_DIR / f"{qid}_target_reference.jpg",
        quality=95,
    )


# ============================================================
# JSON
# ============================================================

out_json = OUTPUT_DIR / "stage1b_candidates.json"

with open(
    out_json,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_results,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n" + "=" * 80)
print("STAGE 1B COMPLETE")
print("=" * 80)

print("JSON:", out_json)

for p in sorted(OUTPUT_DIR.glob("*.jpg")):
    print("IMAGE:", p.name)
