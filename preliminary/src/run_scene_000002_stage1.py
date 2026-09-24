import os
import re
import gc
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    GroundingDinoProcessor,
    GroundingDinoForObjectDetection,
)

# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path("/root/autodl-tmp/aic_v1_project")

DATA_ROOT = (
    PROJECT_ROOT
    / "data/official/初赛数据集-基于大模型的多模态视觉理解与推理"
)

QUERY_JSON = DATA_ROOT / "queries/queries.json"
VISIBLE_PATH = DATA_ROOT / "Images/visible/000002.png"

OUTPUT_DIR = PROJECT_ROOT / "outputs/stage1_000002"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_HUB = PROJECT_ROOT / "models/hf_cache/hub"

SCENE_ID = "000002"

DINO_THRESHOLD = 0.15
DINO_TEXT_THRESHOLD = 0.15
TOP_K = 12


# ============================================================
# Find local HF snapshots
# ============================================================

def find_snapshot(repo_folder):
    base = HF_HUB / repo_folder / "snapshots"

    if not base.exists():
        raise FileNotFoundError(f"Model cache not found: {base}")

    candidates = [
        p for p in base.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    ]

    if not candidates:
        raise FileNotFoundError(f"No valid snapshot under: {base}")

    return str(candidates[0])


QWEN_PATH = find_snapshot(
    "models--Qwen--Qwen3-VL-8B-Instruct"
)

DINO_PATH = find_snapshot(
    "models--IDEA-Research--grounding-dino-base"
)

print("Qwen:", QWEN_PATH)
print("DINO:", DINO_PATH)


# ============================================================
# Load queries
# ============================================================

with open(QUERY_JSON, "r", encoding="utf-8") as f:
    all_queries = json.load(f)

scene_queries = {
    k: v
    for k, v in all_queries.items()
    if k.startswith(SCENE_ID + "_")
}

scene_queries = dict(sorted(scene_queries.items()))

print("\nQueries:")
for qid, item in scene_queries.items():
    print(qid, "->", item["query"])


# ============================================================
# Utilities
# ============================================================

def extract_json(text):
    """
    Try to extract the first JSON object from Qwen output.
    """
    text = text.strip()

    text = re.sub(r"^```json", "", text, flags=re.I).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def clean_prompt(x):
    if not isinstance(x, str):
        return None

    x = x.strip()

    if not x:
        return None

    if not x.endswith("."):
        x += "."

    return x


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def deduplicate_candidates(candidates, iou_threshold=0.85):
    """
    Merge almost-identical boxes coming from different DINO prompts.
    Keep higher-confidence candidate.
    """

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True,
    )

    kept = []

    for cand in candidates:
        duplicate = False

        for old in kept:
            if box_iou(cand["bbox"], old["bbox"]) >= iou_threshold:
                duplicate = True

                # Record extra prompt that also found this box.
                old.setdefault("matched_prompts", [])

                if cand["prompt"] not in old["matched_prompts"]:
                    old["matched_prompts"].append(cand["prompt"])

                break

        if not duplicate:
            cand["matched_prompts"] = [cand["prompt"]]
            kept.append(cand)

    return kept[:TOP_K]


# ============================================================
# Stage A
# Qwen3-VL query parsing
# ============================================================

print("\n" + "=" * 70)
print("Loading Qwen3-VL for query parsing...")
print("=" * 70)

qwen_processor = AutoProcessor.from_pretrained(
    QWEN_PATH,
    local_files_only=True,
)

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

qwen_model.eval()


def parse_query_with_qwen(query):
    prompt = f"""
You are the query parser of a visual grounding system.

The user gives an English referring-expression query.
Convert it into compact structured information for an object detector.

Important:
1. Do NOT solve the image.
2. Extract what object should ultimately be localized.
3. Grounding DINO works better with short visual noun phrases.
4. dino_prompts should contain 1 to 3 SHORT detector phrases, from specific to generic.
5. Remove spatial reasoning such as "leftmost", "beside", "below", "in front of"
   from detector prompts. Those relations will be solved later.
6. Keep useful visual attributes such as color when appropriate.

Return ONLY valid JSON in exactly this structure:

{{
  "target_class": "generic object category",
  "attributes": ["attribute1", "attribute2"],
  "relation": "none or short spatial relation",
  "reference_object": "none or referenced object",
  "ordinal": "none or leftmost/rightmost/first/second/etc",
  "dino_prompts": ["short phrase 1", "short phrase 2"]
}}

Query:
{query}
""".strip()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    text = qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = qwen_processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(qwen_model.device)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated = qwen_model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]

    generated_text = qwen_processor.batch_decode(
        generated[:, input_len:],
        skip_special_tokens=True,
    )[0]

    parsed = extract_json(generated_text)

    if parsed is None:
        print("WARNING: Qwen JSON parse failed.")
        print("Raw output:", generated_text)

        parsed = {
            "target_class": query,
            "attributes": [],
            "relation": "none",
            "reference_object": "none",
            "ordinal": "none",
            "dino_prompts": [query],
        }

    prompts = parsed.get("dino_prompts", [])

    if isinstance(prompts, str):
        prompts = [prompts]

    prompts = [
        p for p in (clean_prompt(x) for x in prompts)
        if p
    ]

    # Extra fallback: generic target class
    target_class = clean_prompt(
        parsed.get("target_class", "")
    )

    if target_class and target_class not in prompts:
        prompts.append(target_class)

    parsed["dino_prompts"] = prompts[:3]

    return parsed, generated_text


parsed_queries = {}

for qid, item in scene_queries.items():
    query = item["query"]

    print("\n" + "-" * 70)
    print(qid)
    print("QUERY:", query)

    parsed, raw = parse_query_with_qwen(query)

    parsed_queries[qid] = {
        "query": query,
        "parsed": parsed,
        "qwen_raw": raw,
    }

    print(
        json.dumps(
            parsed,
            ensure_ascii=False,
            indent=2,
        )
    )


with open(
    OUTPUT_DIR / "parsed_queries.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        parsed_queries,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Release Qwen GPU memory before DINO
# ============================================================

print("\nReleasing Qwen...")

del qwen_model
del qwen_processor

gc.collect()
torch.cuda.empty_cache()

print(
    "CUDA allocated after release:",
    round(torch.cuda.memory_allocated() / 1024**3, 3),
    "GB",
)


# ============================================================
# Stage B
# Grounding DINO candidate generation
# ============================================================

print("\n" + "=" * 70)
print("Loading Grounding DINO...")
print("=" * 70)

dino_processor = GroundingDinoProcessor.from_pretrained(
    DINO_PATH,
    local_files_only=True,
)

dino_model = GroundingDinoForObjectDetection.from_pretrained(
    DINO_PATH,
    local_files_only=True,
).cuda()

dino_model.eval()

rgb = Image.open(VISIBLE_PATH).convert("RGB")
width, height = rgb.size

print("Image size:", width, "x", height)


def run_dino(prompt):
    inputs = dino_processor(
        images=rgb,
        text=prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=DINO_THRESHOLD,
        text_threshold=DINO_TEXT_THRESHOLD,
        target_sizes=[(height, width)],
    )[0]

    boxes = results["boxes"].detach().cpu().tolist()
    scores = results["scores"].detach().cpu().tolist()

    labels = results.get("labels", [])

    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    candidates = []

    for idx, (box, score) in enumerate(zip(boxes, scores)):
        label = (
            str(labels[idx])
            if idx < len(labels)
            else prompt
        )

        x1, y1, x2, y2 = box

        candidates.append(
            {
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                ],
                "bbox_normalized": [
                    round(float(x1) / width, 6),
                    round(float(y1) / height, 6),
                    round(float(x2) / width, 6),
                    round(float(y2) / height, 6),
                ],
                "score": round(float(score), 6),
                "label": label,
                "prompt": prompt,
            }
        )

    return candidates


all_results = {}

for qid, info in parsed_queries.items():
    print("\n" + "=" * 70)
    print(qid)
    print(info["query"])

    prompts = info["parsed"]["dino_prompts"]

    print("DINO prompts:", prompts)

    raw_candidates = []

    for prompt in prompts:
        candidates = run_dino(prompt)

        print(
            f'  "{prompt}" -> {len(candidates)} candidates'
        )

        raw_candidates.extend(candidates)

    candidates = deduplicate_candidates(raw_candidates)

    # Assign stable candidate IDs
    for i, cand in enumerate(candidates):
        cand["candidate_id"] = f"C{i}"

    print("Merged Top-K:", len(candidates))

    for cand in candidates:
        print(
            cand["candidate_id"],
            "score=",
            cand["score"],
            "bbox=",
            cand["bbox"],
            "prompts=",
            cand["matched_prompts"],
        )

    all_results[qid] = {
        "query": info["query"],
        "parsed": info["parsed"],
        "candidates": candidates,
    }

    # --------------------------------------------------------
    # Draw candidate visualization
    # --------------------------------------------------------

    vis = rgb.copy()
    draw = ImageDraw.Draw(vis)

    for cand in candidates:
        x1, y1, x2, y2 = cand["bbox"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0),
            width=4,
        )

        text = (
            f'{cand["candidate_id"]} '
            f'{cand["score"]:.3f}'
        )

        tx = int(x1)
        ty = max(0, int(y1) - 20)

        draw.rectangle(
            [tx, ty, tx + 125, ty + 20],
            fill=(255, 255, 255),
        )

        draw.text(
            (tx + 2, ty + 2),
            text,
            fill=(0, 0, 0),
        )

    out_path = OUTPUT_DIR / f"{qid}_candidates.jpg"

    vis.save(
        out_path,
        quality=95,
    )

    print("Saved:", out_path)


with open(
    OUTPUT_DIR / "stage1_candidates.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        all_results,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n" + "=" * 70)
print("STAGE 1 COMPLETE")
print("=" * 70)

print("Output directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for p in sorted(OUTPUT_DIR.iterdir()):
    print(" -", p.name)

print(
    "\nPeak CUDA memory:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB",
)
