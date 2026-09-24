#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V1 Full Runner
==============
Qwen3-VL parser + Grounding DINO high-recall candidates + Stage 2.2 Hybrid judge.

Designed for the user's AutoDL layout:
  /root/autodl-tmp/aic_v1_project/

Key properties:
- Load Qwen3-VL once.
- Load Grounding DINO once.
- Parse all queries in one scene together (fallback to per-query parse on failure).
- Process all 2000 scenes / 9555 queries.
- Save checkpoint after every scene.
- --resume supported.
- Keep detailed debug JSON and generate submission.json preserving original query records.
- Stage 2.2 philosophy: Qwen is the final semantic judge; ordinary geometry is a soft hint;
  depth is routed only to depth-related queries; infrared only to thermal/IR queries;
  conservative loose-box suppression handles obvious duplicate oversized boxes.

This script intentionally avoids saving thousands of debug images permanently.
Per-query overlays/crops are written to a runtime cache and removed after judging.
"""

import argparse
import gc
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# CLI
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description="Run V1 on the full competition dataset")
    p.add_argument(
        "--project-root",
        default="/root/autodl-tmp/aic_v1_project",
        help="Project root",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Dataset root; default is <project-root>/data/official/初赛数据集-基于大模型的多模态视觉理解与推理",
    )
    p.add_argument(
        "--queries",
        default=None,
        help="queries.json path; default is <data-root>/queries/queries.json",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; default is <project-root>/outputs/v1_full",
    )
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint_predictions.json")
    p.add_argument("--scene", default=None, help="Run only one scene id, e.g. 000002")
    p.add_argument("--start-scene-index", type=int, default=0, help="Start index in sorted scene list")
    p.add_argument("--max-scenes", type=int, default=None, help="Optional number of scenes to run")
    p.add_argument("--dino-threshold", type=float, default=0.12)
    p.add_argument("--dino-text-threshold", type=float, default=0.10)
    p.add_argument("--dino-batch", type=int, default=2)
    p.add_argument("--max-targets", type=int, default=12)
    p.add_argument("--max-refs", type=int, default=5)
    p.add_argument("--max-shortlist", type=int, default=8)
    p.add_argument("--max-group-base", type=int, default=7)
    p.add_argument("--keep-runtime-images", action="store_true", help="Keep temporary overlays/crops")
    p.add_argument("--no-ir", action="store_true", help="Disable IR routing even for thermal queries")
    p.add_argument("--no-depth", action="store_true", help="Disable depth routing even for depth queries")
    p.add_argument("--checkpoint-every-scenes", type=int, default=20, help="Rewrite full checkpoint/submission every N completed scenes; per-scene result files are saved every scene")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


ARGS = parse_args()
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

PROJECT_ROOT = Path(ARGS.project_root)
DATA_ROOT = Path(ARGS.data_root) if ARGS.data_root else (
    PROJECT_ROOT / "data/official/初赛数据集-基于大模型的多模态视觉理解与推理"
)
QUERIES_PATH = Path(ARGS.queries) if ARGS.queries else DATA_ROOT / "queries/queries.json"
OUTPUT_DIR = Path(ARGS.output_dir) if ARGS.output_dir else PROJECT_ROOT / "outputs/v1_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = OUTPUT_DIR / "runtime_cache"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_predictions.json"
SUBMISSION_PATH = OUTPUT_DIR / "submission.json"
SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"
FAILURES_PATH = OUTPUT_DIR / "failures.json"
SCENE_RESULTS_DIR = OUTPUT_DIR / "scene_results"
SCENE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB = PROJECT_ROOT / "models/hf_cache/hub"

MAX_TARGETS = ARGS.max_targets
MAX_REFS = ARGS.max_refs
MAX_SHORTLIST = ARGS.max_shortlist
MAX_GROUP_BASE = ARGS.max_group_base


# ============================================================
# Utilities
# ============================================================


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def find_snapshot(repo_dir_name: str) -> str:
    root = HF_HUB / repo_dir_name / "snapshots"
    if not root.exists():
        raise RuntimeError(f"Model cache not found: {root}")
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
    raise RuntimeError(f"No valid snapshot found under: {root}")


def atomic_json_dump(obj: Any, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    # Strip common code fences first.
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        x = json.loads(text)
        return x if isinstance(x, dict) else None
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            x = json.loads(text[s:e + 1])
            return x if isinstance(x, dict) else None
        except Exception:
            return None
    return None


def clamp_bbox(b: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = map(float, b)
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 < 1:
        x2 = min(float(w), x1 + 1)
    if y2 - y1 < 1:
        y2 = min(float(h), y1 + 1)
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def normalize_bbox(b: List[float], w: int, h: int) -> List[float]:
    b = clamp_bbox(b, w, h)
    return [
        round(max(0.0, min(1.0, b[0] / w)), 6),
        round(max(0.0, min(1.0, b[1] / h)), 6),
        round(max(0.0, min(1.0, b[2] / w)), 6),
        round(max(0.0, min(1.0, b[3] / h)), 6),
    ]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def intersection(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a, b):
    inter = intersection(a, b)
    smaller = min(area(a), area(b))
    return inter / smaller if smaller > 0 else 0.0


def horizontal_overlap(a, b):
    ov = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    denom = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return ov / denom


def vertical_overlap(a, b):
    ov = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    denom = max(1.0, min(a[3] - a[1], b[3] - b[1]))
    return ov / denom


def union_boxes(boxes):
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def human_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ============================================================
# Load queries / scene groups
# ============================================================

if not QUERIES_PATH.exists():
    raise FileNotFoundError(f"queries.json not found: {QUERIES_PATH}")

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    ORIGINAL_QUERIES: Dict[str, dict] = json.load(f)

if not isinstance(ORIGINAL_QUERIES, dict):
    raise RuntimeError("Expected queries.json to be a JSON object keyed by query id")

SCENES: Dict[str, List[str]] = defaultdict(list)
for qid in ORIGINAL_QUERIES:
    scene_id = qid.split("_")[0]
    SCENES[scene_id].append(qid)

for sid in SCENES:
    SCENES[sid] = sorted(SCENES[sid])

scene_ids = sorted(SCENES)
if ARGS.scene:
    if ARGS.scene not in SCENES:
        raise ValueError(f"Scene {ARGS.scene} not found")
    scene_ids = [ARGS.scene]
else:
    scene_ids = scene_ids[ARGS.start_scene_index:]
    if ARGS.max_scenes is not None:
        scene_ids = scene_ids[:ARGS.max_scenes]

print(f"[{now_str()}] Dataset queries: {len(ORIGINAL_QUERIES)}")
print(f"[{now_str()}] Dataset scenes : {len(SCENES)}")
print(f"[{now_str()}] This run scenes: {len(scene_ids)}")
print(f"[{now_str()}] queries.json   : {QUERIES_PATH}")
print(f"[{now_str()}] output dir     : {OUTPUT_DIR}")


# ============================================================
# Models
# ============================================================

QWEN_PATH = find_snapshot("models--Qwen--Qwen3-VL-8B-Instruct")
DINO_PATH = find_snapshot("models--IDEA-Research--grounding-dino-base")

print(f"[{now_str()}] Loading Qwen3-VL: {QWEN_PATH}")
qwen_processor = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True)
qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
qwen_model.eval()

print(f"[{now_str()}] Loading Grounding DINO: {DINO_PATH}")
dino_processor = AutoProcessor.from_pretrained(DINO_PATH, local_files_only=True)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_PATH,
    local_files_only=True,
).to("cuda")
dino_model.eval()


# ============================================================
# Qwen text generation helpers
# ============================================================


def qwen_text_only(prompt: str, max_new_tokens: int = 1200) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], padding=True, return_tensors="pt").to(qwen_model.device)
    with torch.inference_mode():
        out = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    n = inputs.input_ids.shape[1]
    return qwen_processor.batch_decode(out[:, n:], skip_special_tokens=True)[0]


CANONICAL_RELATIONS = {
    "above", "over", "below", "under", "beside", "next to", "on",
    "left of", "right of", "in front of", "behind", "inside", "near"
}


def normalize_relation(x: str) -> str:
    x = (x or "").strip().lower()
    aliases = {
        "underneath": "below",
        "beneath": "below",
        "under": "below",
        "over": "over",
        "above": "above",
        "next to": "beside",
        "near to": "near",
        "infront of": "in front of",
        "front of": "in front of",
        "mounted on": "on",
        "standing on": "on",
        "within": "inside",
    }
    return aliases.get(x, x)


def sanitize_parse(qid: str, query: str, p: Optional[dict]) -> dict:
    if not isinstance(p, dict):
        p = {}
    target_class = str(p.get("target_class") or query).strip()
    attrs = p.get("target_attributes") or []
    if not isinstance(attrs, list):
        attrs = [str(attrs)]
    attrs = [str(x).strip() for x in attrs if str(x).strip()]

    try:
        count = int(p.get("count", 1) or 1)
    except Exception:
        count = 1
    count = max(1, min(count, 20))

    ordinal = str(p.get("ordinal") or "none").strip().lower()
    prompts = p.get("target_dino_prompts") or []
    if not isinstance(prompts, list):
        prompts = [str(prompts)]
    prompts = [str(x).strip() for x in prompts if str(x).strip()]
    # Always preserve the clean target class as a recall prompt.
    if target_class and target_class.lower() not in {x.lower() for x in prompts}:
        prompts.append(target_class)
    prompts = prompts[:6]

    constraints = []
    raw_constraints = p.get("constraints") or []
    if isinstance(raw_constraints, dict):
        raw_constraints = [raw_constraints]
    for c in raw_constraints:
        if not isinstance(c, dict):
            continue
        rel = normalize_relation(str(c.get("relation") or ""))
        ref = str(c.get("reference_object") or "").strip()
        if not rel or not ref:
            continue
        rps = c.get("reference_dino_prompts") or []
        if not isinstance(rps, list):
            rps = [str(rps)]
        rps = [str(x).strip() for x in rps if str(x).strip()]
        if ref.lower() not in {x.lower() for x in rps}:
            rps.append(ref)
        constraints.append({
            "relation": rel,
            "reference_object": ref,
            "reference_dino_prompts": rps[:5],
        })

    return {
        "target_class": target_class,
        "target_attributes": attrs,
        "count": count,
        "ordinal": ordinal,
        "target_dino_prompts": prompts,
        "constraints": constraints,
    }


PARSER_INSTRUCTIONS = r"""
You are a parser for a visual grounding system. Parse each English referring expression into structured JSON.

The TARGET is the object/region that the final bounding box must cover. Reference objects are only used to resolve relations and must NOT be mixed into target_class.

For every query output:
- target_class: concise core target noun phrase.
- target_attributes: visible semantic attributes of the target, e.g. color, clothing, material, text/imagery.
- count: explicit requested target count; default 1.
- ordinal: normalize spatial/ordinal wording, e.g. none, leftmost, rightmost, topmost, bottommost, second from left, third from right.
- target_dino_prompts: 2-5 short Grounding-DINO-friendly noun phrases. Include the core noun alone when useful, plus one or two attribute/synonym variants. Do not include reference-object phrases as target prompts.
- constraints: each relation to a reference object as {relation, reference_object, reference_dino_prompts}.

Use concise canonical relations when possible: above, over, below, beside, on, left of, right of, in front of, behind, inside, near.
Examples:
"man in a checkered shirt beside the white street lamp" -> target=man; attribute=checkered shirt; relation=beside; reference=white street lamp.
"leftmost security camera mounted on the stone pillar, directly below the black awning" -> target=security camera; ordinal=leftmost; constraints on stone pillar + below black awning.
"golden menu standing in front of the black dining barrier" -> target=menu; attribute=golden; relation=in front of; reference=black dining barrier.
"two white umbrellas above the outdoor dining area" -> target=umbrella; count=2; attribute=white; relation=above; reference=outdoor dining area.

Return ONLY a JSON object keyed by the supplied query IDs. No markdown, no prose.
"""


def parse_scene_queries(qids: List[str]) -> Dict[str, dict]:
    payload = {qid: ORIGINAL_QUERIES[qid]["query"] for qid in qids}
    prompt = PARSER_INSTRUCTIONS + "\nQUERIES:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    raw = qwen_text_only(prompt, max_new_tokens=max(900, 260 * len(qids)))
    obj = extract_json_object(raw)

    parsed: Dict[str, dict] = {}
    if isinstance(obj, dict):
        for qid in qids:
            if isinstance(obj.get(qid), dict):
                parsed[qid] = sanitize_parse(qid, payload[qid], obj.get(qid))

    # If batched output missed anything, fall back to one query at a time.
    for qid in qids:
        if qid in parsed:
            continue
        one_prompt = PARSER_INSTRUCTIONS + "\nQUERIES:\n" + json.dumps({qid: payload[qid]}, ensure_ascii=False)
        try:
            one_raw = qwen_text_only(one_prompt, max_new_tokens=500)
            one_obj = extract_json_object(one_raw)
            parsed[qid] = sanitize_parse(qid, payload[qid], one_obj.get(qid) if isinstance(one_obj, dict) else None)
        except Exception:
            parsed[qid] = sanitize_parse(qid, payload[qid], None)
    return parsed


# ============================================================
# Grounding DINO candidate generation
# ============================================================


def dedupe_prompts(prompts: List[str]) -> List[str]:
    out, seen = [], set()
    for p in prompts:
        p = re.sub(r"\s+", " ", str(p).strip().strip("."))
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def dino_detect_prompts(
    image: Image.Image,
    prompts: List[str],
    threshold: Optional[float] = None,
    text_threshold: Optional[float] = None,
) -> List[dict]:
    prompts = dedupe_prompts(prompts)
    if not prompts:
        return []
    threshold = ARGS.dino_threshold if threshold is None else threshold
    text_threshold = ARGS.dino_text_threshold if text_threshold is None else text_threshold
    w, h = image.size
    candidates = []

    for start in range(0, len(prompts), max(1, ARGS.dino_batch)):
        batch_prompts = prompts[start:start + max(1, ARGS.dino_batch)]
        texts = [p + "." for p in batch_prompts]
        images = [image] * len(texts)
        inputs = dino_processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = dino_model(**inputs)

        # Current Transformers Grounding DINO API uses threshold and text_threshold.
        results = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)] * len(texts),
        )

        for prompt, result in zip(batch_prompts, results):
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            if hasattr(boxes, "detach"):
                boxes = boxes.detach().cpu().tolist()
            if hasattr(scores, "detach"):
                scores = scores.detach().cpu().tolist()
            for b, s in zip(boxes, scores):
                b = clamp_bbox([float(v) for v in b], w, h)
                if area(b) < 4:
                    continue
                candidates.append({
                    "bbox": b,
                    "score": round(float(s), 6),
                    "prompt": prompt + ".",
                    "matched_prompts": [prompt + "."],
                })
    return candidates


def merge_candidates(candidates: List[dict], max_keep: int) -> List[dict]:
    # Conservative duplicate merge. Adjacent instances should remain separate.
    ordered = sorted(candidates, key=lambda x: float(x["score"]), reverse=True)
    kept: List[dict] = []
    for c in ordered:
        merged = False
        for k in kept:
            ov = iou(c["bbox"], k["bbox"])
            cont = containment(c["bbox"], k["bbox"])
            if ov >= 0.72 or cont >= 0.93:
                # Keep the higher-score geometry, but record prompt support.
                k.setdefault("matched_prompts", [])
                for p in c.get("matched_prompts", [c.get("prompt")]):
                    if p and p not in k["matched_prompts"]:
                        k["matched_prompts"].append(p)
                merged = True
                break
        if not merged:
            kept.append(dict(c))
        if len(kept) >= max_keep * 2:
            # Enough recall before final truncation.
            pass
    kept = sorted(kept, key=lambda x: float(x["score"]), reverse=True)[:max_keep]
    for i, c in enumerate(kept):
        c["candidate_id"] = f"T{i}"
    return kept


def make_target_candidates(image: Image.Image, parsed: dict, query: str) -> List[dict]:
    prompts = parsed.get("target_dino_prompts", [])
    candidates = dino_detect_prompts(image, prompts)
    merged = merge_candidates(candidates, MAX_TARGETS)

    # Recall fallback chain from V1 plan.
    if not merged:
        fallback_prompts = [query, parsed.get("target_class", "")]
        candidates = dino_detect_prompts(image, fallback_prompts, threshold=max(0.07, ARGS.dino_threshold - 0.04))
        merged = merge_candidates(candidates, MAX_TARGETS)

    if not merged:
        # Last DINO retry at low threshold using the clean target noun.
        core = parsed.get("target_class") or query
        candidates = dino_detect_prompts(image, [core], threshold=0.05, text_threshold=0.05)
        merged = merge_candidates(candidates, MAX_TARGETS)
    return merged


def make_reference_candidates(image: Image.Image, parsed: dict) -> List[dict]:
    refs = []
    for ridx, c in enumerate(parsed.get("constraints", [])):
        prompts = c.get("reference_dino_prompts", []) or [c.get("reference_object", "")]
        cand = dino_detect_prompts(image, prompts)
        merged = merge_candidates(cand, MAX_REFS)
        # Rename target-style IDs to reference IDs.
        for j, x in enumerate(merged):
            x["candidate_id"] = f"R{ridx}_{j}"
        refs.append({
            "relation": normalize_relation(c.get("relation", "")),
            "reference_object": c.get("reference_object", ""),
            "candidates": merged,
        })
    return refs


# ============================================================
# Stage 2.2 scene context / routing
# ============================================================

DEPTH_QUERY_WORDS = {
    "in front of", "behind", "nearest", "closest", "farthest", "farther",
    "closer", "frontmost", "rearmost", "nearer"
}
IR_WORDS = {
    "hot", "hottest", "warm", "warmer", "warmest", "heat", "thermal",
    "cold", "colder", "coldest", "infrared"
}


def get_routing(parsed: dict, query: str) -> dict:
    relations = [normalize_relation(str(x.get("relation", ""))) for x in parsed.get("constraints", [])]
    q = query.lower()
    use_depth = any(r in {"in front of", "behind"} for r in relations) or any(w in q for w in DEPTH_QUERY_WORDS)
    use_ir = any(w in q for w in IR_WORDS)
    if ARGS.no_depth:
        use_depth = False
    if ARGS.no_ir:
        use_ir = False
    return {
        "RGB": True,
        "geometry": bool(parsed.get("constraints")) or parsed.get("ordinal", "none") != "none",
        "depth": bool(use_depth),
        "infrared": bool(use_ir),
    }


class SceneContext:
    def __init__(self, scene_id: str, visible_path: Path, ir_path: Path, depth_path: Path):
        self.scene_id = scene_id
        self.visible_path = visible_path
        self.ir_path = ir_path
        self.depth_path = depth_path
        self.rgb_pil = Image.open(visible_path).convert("RGB")
        self.W, self.H = self.rgb_pil.size
        self.depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if self.depth is None:
            self.depth = np.zeros((self.H, self.W), dtype=np.uint16)
        if self.depth.ndim == 3:
            self.depth = self.depth[..., 0]
        self.depth_vis_path = RUNTIME_DIR / f"{scene_id}_depth_vis.png"
        self._build_depth_vis()

    def _build_depth_vis(self):
        depth = self.depth
        valid = depth > 0
        vis = np.zeros((self.H, self.W), dtype=np.uint8)
        if valid.any():
            vals = depth[valid].astype(np.float32)
            lo, hi = np.percentile(vals, [2, 98])
            norm = (depth.astype(np.float32) - lo) / max(float(hi - lo), 1.0)
            norm = np.clip(norm, 0, 1)
            vis = ((1.0 - norm) * 255).astype(np.uint8)
            vis[~valid] = 0
        Image.fromarray(vis).convert("RGB").save(self.depth_vis_path)

    def depth_stats(self, box, ratio=0.65):
        x1, y1, x2, y2 = map(float, box)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ww, hh = (x2 - x1) * ratio, (y2 - y1) * ratio
        x1 = int(max(0, cx - ww / 2))
        x2 = int(min(self.W, cx + ww / 2))
        y1 = int(max(0, cy - hh / 2))
        y2 = int(min(self.H, cy + hh / 2))
        roi = self.depth[y1:y2, x1:x2]
        if roi.size == 0:
            return {"valid_ratio": 0.0, "median_mm": None}
        mask = (roi > 300) & (roi <= 19999)
        vals = roi[mask]
        vr = float(vals.size / roi.size)
        return {
            "valid_ratio": round(vr, 4),
            "median_mm": int(np.median(vals)) if vals.size else None,
        }

    def cleanup(self):
        if not ARGS.keep_runtime_images:
            try:
                self.depth_vis_path.unlink(missing_ok=True)
            except Exception:
                pass


# ============================================================
# Stage 2.2 Hybrid scoring
# ============================================================


def tightness_filter(candidates: List[dict], count: int):
    if count != 1:
        return [dict(x) for x in candidates], []
    candidates = [dict(x) for x in candidates]
    suppressed = set()
    reasons = []
    for i in range(len(candidates)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(candidates)):
            if j in suppressed:
                continue
            a, b = candidates[i], candidates[j]
            ov = iou(a["bbox"], b["bbox"])
            if ov < 0.40:
                continue
            aa, ab = area(a["bbox"]), area(b["bbox"])
            small_idx, large_idx = (i, j) if aa <= ab else (j, i)
            small, large = candidates[small_idx], candidates[large_idx]
            sa, la = area(small["bbox"]), area(large["bbox"])
            if sa <= 0:
                continue
            area_ratio = la / sa
            small_score = float(small["score"])
            large_score = float(large["score"])
            score_ratio = small_score / max(large_score, 1e-6)
            score_gap = small_score - large_score
            if area_ratio >= 1.20 and score_ratio >= 1.50 and score_gap >= 0.15:
                suppressed.add(large_idx)
                reasons.append({
                    "suppressed": large["candidate_id"],
                    "kept": small["candidate_id"],
                    "iou": round(ov, 3),
                    "area_ratio": round(area_ratio, 3),
                    "small_score": round(small_score, 6),
                    "large_score": round(large_score, 6),
                    "score_ratio": round(score_ratio, 3),
                    "score_gap": round(score_gap, 3),
                })
    kept = [x for idx, x in enumerate(candidates) if idx not in suppressed]
    return kept, reasons


def relation_score(ctx: SceneContext, target, ref, relation, allow_depth=False):
    tb, rb = target["bbox"], ref["bbox"]
    tcx, tcy = center(tb)
    rcx, rcy = center(rb)
    hov, vov = horizontal_overlap(tb, rb), vertical_overlap(tb, rb)
    dx, dy = abs(tcx - rcx) / ctx.W, abs(tcy - rcy) / ctx.H
    near_y = max(0.0, 1.0 - dy / 0.35)
    near_center = max(0.0, 1.0 - math.hypot(dx, dy) / 0.45)
    inside_ref = rb[0] <= tcx <= rb[2] and rb[1] <= tcy <= rb[3]
    relation = normalize_relation(relation)

    details = {
        "horizontal_overlap": round(hov, 3),
        "vertical_overlap": round(vov, 3),
    }

    if relation in {"above", "over"}:
        score = 0.50 * float(tcy < rcy) + 0.30 * hov + 0.20 * near_y
    elif relation in {"below", "under"}:
        score = 0.50 * float(tcy > rcy) + 0.30 * hov + 0.20 * near_y
    elif relation in {"beside", "next to", "near"}:
        score = 0.55 * vov + 0.45 * near_center
    elif relation == "on":
        score = 0.60 * float(inside_ref) + 0.20 * hov + 0.20 * vov
        details["target_center_inside_reference"] = bool(inside_ref)
    elif relation == "inside":
        score = 0.75 * float(inside_ref) + 0.25 * min(1.0, max(hov, vov))
        details["target_center_inside_reference"] = bool(inside_ref)
    elif relation == "left of":
        score = 0.65 * float(tcx < rcx) + 0.35 * near_center
    elif relation == "right of":
        score = 0.65 * float(tcx > rcx) + 0.35 * near_center
    elif relation in {"in front of", "behind"} and allow_depth:
        td, rd = ctx.depth_stats(tb), ctx.depth_stats(rb)
        tdepth, rdepth = td["median_mm"], rd["median_mm"]
        depth_ok = False
        depth_delta = None
        if tdepth is not None and rdepth is not None:
            depth_delta = rdepth - tdepth
            depth_ok = (tdepth < rdepth) if relation == "in front of" else (tdepth > rdepth)
        score = 0.65 * float(depth_ok) + 0.20 * max(hov, vov) + 0.15 * near_center
        details.update({
            "target_depth_mm": tdepth,
            "reference_depth_mm": rdepth,
            "depth_delta_mm": depth_delta,
            "depth_relation_ok": bool(depth_ok),
        })
    else:
        score = near_center
    return float(score), details


def score_constraints(ctx: SceneContext, target, references, routing):
    if not references:
        return 1.0, []
    out = []
    for ref_info in references:
        relation = ref_info["relation"]
        best = None
        for ref in ref_info.get("candidates", [])[:MAX_REFS]:
            rs, details = relation_score(ctx, target, ref, relation, routing["depth"])
            pair_score = 0.85 * rs + 0.15 * float(ref["score"])
            item = {
                "reference_id": ref["candidate_id"],
                "reference_detector_score": ref["score"],
                "relation_score": round(rs, 4),
                "pair_score": round(pair_score, 4),
                "details": details,
            }
            if best is None or item["pair_score"] > best["pair_score"]:
                best = item
        out.append({
            "relation": relation,
            "reference_object": ref_info["reference_object"],
            "best_pair": best,
        })
    vals = [x["best_pair"]["pair_score"] for x in out if x["best_pair"] is not None]
    return (min(vals) if vals else 0.0), out


def prepare_targets(ctx: SceneContext, parsed: dict, targets: List[dict], references: List[dict], routing: dict):
    count = int(parsed.get("count", 1) or 1)
    targets, suppressed = tightness_filter(targets, count)
    scored = []
    for c in targets:
        c = dict(c)
        rel, info = score_constraints(ctx, c, references, routing)
        c["relation_score"] = round(rel, 4)
        c["relation_info"] = info
        if not references:
            priority = float(c["score"])
        elif routing["depth"]:
            priority = 0.40 * float(c["score"]) + 0.60 * rel
        else:
            priority = 0.75 * float(c["score"]) + 0.25 * rel
        c["hybrid_priority"] = round(priority, 4)
        if routing["depth"]:
            c["depth"] = ctx.depth_stats(c["bbox"])
        scored.append(c)
    return scored, suppressed


def annotate_ordinal(candidates: List[dict], ordinal: str):
    ordinal = str(ordinal or "none").lower()
    for c in candidates:
        c["ordinal_rank"] = None
    if ordinal == "none" or not candidates:
        return candidates

    if "from left" in ordinal or ordinal == "leftmost":
        ordered = sorted(candidates, key=lambda c: center(c["bbox"])[0])
    elif "from right" in ordinal or ordinal == "rightmost":
        ordered = sorted(candidates, key=lambda c: center(c["bbox"])[0], reverse=True)
    elif ordinal in {"topmost", "uppermost"} or "from top" in ordinal:
        ordered = sorted(candidates, key=lambda c: center(c["bbox"])[1])
    elif ordinal in {"bottommost", "lowermost"} or "from bottom" in ordinal:
        ordered = sorted(candidates, key=lambda c: center(c["bbox"])[1], reverse=True)
    else:
        ordered = candidates
    for i, c in enumerate(ordered):
        c["ordinal_rank"] = i
    return candidates


def desired_ordinal_rank(ordinal: str) -> int:
    o = str(ordinal or "none").lower()
    if any(x in o for x in ["second", "2nd"]):
        return 1
    if any(x in o for x in ["third", "3rd"]):
        return 2
    if any(x in o for x in ["fourth", "4th"]):
        return 3
    if any(x in o for x in ["fifth", "5th"]):
        return 4
    return 0


def make_shortlist(candidates: List[dict], routing: dict, ordinal: str):
    candidates = annotate_ordinal(candidates, ordinal)
    if routing["depth"]:
        ordered = sorted(candidates, key=lambda c: c["hybrid_priority"], reverse=True)
    else:
        ordered = sorted(candidates, key=lambda c: float(c["score"]), reverse=True)
    shortlist = ordered[:MAX_SHORTLIST]

    # Ensure the requested ordinal rank survives shortlist truncation, while keeping geometry soft.
    if ordinal and ordinal != "none":
        wanted_rank = desired_ordinal_rank(ordinal)
        ord_candidates = [c for c in candidates if c.get("ordinal_rank") == wanted_rank]
        if ord_candidates and all(c["candidate_id"] != ord_candidates[0]["candidate_id"] for c in shortlist):
            if shortlist:
                shortlist[-1] = ord_candidates[0]
            else:
                shortlist = [ord_candidates[0]]
    return shortlist


def build_groups(ctx: SceneContext, targets: List[dict], count: int, references: List[dict], routing: dict):
    if count <= 1 or count > 5:
        return []
    base = sorted(targets, key=lambda x: float(x["score"]), reverse=True)[:MAX_GROUP_BASE]
    groups = []
    for combo in itertools.combinations(base, count):
        if any(iou(a["bbox"], b["bbox"]) > 0.35 for a, b in itertools.combinations(combo, 2)):
            continue
        box = union_boxes([x["bbox"] for x in combo])
        det_score = sum(float(x["score"]) for x in combo) / count
        fake = {"bbox": box, "score": det_score}
        rel, info = score_constraints(ctx, fake, references, routing)
        if not references:
            priority = det_score
        elif routing["depth"]:
            priority = 0.40 * det_score + 0.60 * rel
        else:
            priority = 0.75 * det_score + 0.25 * rel
        g = {
            "candidate_id": "",
            "members": [x["candidate_id"] for x in combo],
            "bbox": [round(v, 2) for v in box],
            "score": round(det_score, 4),
            "relation_score": round(rel, 4),
            "hybrid_priority": round(priority, 4),
            "relation_info": info,
            "ordinal_rank": None,
        }
        if routing["depth"]:
            g["depth"] = ctx.depth_stats(box)
        groups.append(g)
    groups.sort(key=lambda x: x["hybrid_priority"] if routing["depth"] else x["score"], reverse=True)
    groups = groups[:MAX_SHORTLIST]
    for i, g in enumerate(groups):
        g["candidate_id"] = f"G{i}"
    return groups


# ============================================================
# Qwen visual judge
# ============================================================


def draw_overlay(ctx: SceneContext, candidates: List[dict], path: Path, color="red"):
    img = ctx.rgb_pil.copy()
    draw = ImageDraw.Draw(img)
    for c in candidates:
        x1, y1, x2, y2 = c["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.rectangle([x1, max(0, y1 - 24), x1 + 72, y1], fill="white")
        draw.text((x1 + 2, max(0, y1 - 21)), c["candidate_id"], fill="black")
    img.save(path, quality=90)


def draw_reference_overlay(ctx: SceneContext, references: List[dict], path: Path):
    img = ctx.rgb_pil.copy()
    draw = ImageDraw.Draw(img)
    for ref_info in references:
        for ref in ref_info.get("candidates", [])[:MAX_REFS]:
            x1, y1, x2, y2 = ref["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
            draw.rectangle([x1, max(0, y1 - 22), x1 + 90, y1], fill="white")
            draw.text((x1 + 2, max(0, y1 - 19)), ref["candidate_id"], fill="green")
    img.save(path, quality=90)


def make_crop_sheet(ctx: SceneContext, candidates: List[dict], path: Path):
    if not candidates:
        return None
    tw, th = 220, 170
    sheet = Image.new("RGB", (tw * len(candidates), th), "white")
    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = map(int, c["bbox"])
        crop = ctx.rgb_pil.crop((max(0, x1), max(0, y1), min(ctx.W, x2), min(ctx.H, y2)))
        crop.thumbnail((tw - 16, th - 35))
        tile = Image.new("RGB", (tw, th), "white")
        tile.paste(crop, ((tw - crop.width) // 2, 30))
        ImageDraw.Draw(tile).text((8, 8), c["candidate_id"], fill="black")
        sheet.paste(tile, (i * tw, 0))
    sheet.save(path, quality=90)
    return path


def qwen_judge(ctx: SceneContext, qid: str, query: str, parsed: dict, routing: dict, candidates: List[dict], references: List[dict], light=False):
    qdir = RUNTIME_DIR / ctx.scene_id
    qdir.mkdir(parents=True, exist_ok=True)
    overlay = qdir / f"{qid}_targets.jpg"
    crops = qdir / f"{qid}_crops.jpg"
    refs_overlay = qdir / f"{qid}_refs.jpg"

    cand_for_images = candidates[:5] if light else candidates
    draw_overlay(ctx, cand_for_images, overlay)
    make_crop_sheet(ctx, cand_for_images, crops)

    content = [
        {"type": "image", "image": str(ctx.visible_path)},
        {"type": "image", "image": str(overlay)},
        {"type": "image", "image": str(crops)},
    ]
    if references and not light:
        draw_reference_overlay(ctx, references, refs_overlay)
        content.append({"type": "image", "image": str(refs_overlay)})
    if routing["depth"]:
        content.append({"type": "image", "image": str(ctx.depth_vis_path)})
    if routing["infrared"] and ctx.ir_path.exists():
        content.append({"type": "image", "image": str(ctx.ir_path)})

    evidence = []
    for c in candidates:
        e = {
            "candidate_id": c["candidate_id"],
            "bbox": c["bbox"],
            "detector_score": c.get("score"),
            "relation_score_hint": c.get("relation_score"),
            "hybrid_priority_for_shortlist_only": c.get("hybrid_priority"),
            "ordinal_rank_hint": c.get("ordinal_rank"),
            "members": c.get("members"),
            "relation_info": c.get("relation_info"),
        }
        if routing["depth"]:
            e["depth"] = c.get("depth")
        evidence.append(e)

    allowed = [k for k, v in routing.items() if v]
    prompt = f"""You are the FINAL visual grounding candidate judge.

QUERY: {query}
Structured parse: {json.dumps(parsed, ensure_ascii=False)}
ALLOWED EVIDENCE: {allowed}
Candidates: {json.dumps(evidence, ensure_ascii=False)}

Image order:
1) original RGB image
2) target candidates (red boxes)
3) target candidate crops
4) reference candidates (green boxes), if supplied
5) depth visualization, only for depth-routed queries
6) infrared image, only for infrared-routed queries

Rules:
- Qwen remains the final semantic judge. Do NOT blindly follow a numeric score.
- RGB is primary for identity, color, clothing, texture, material, text and semantic appearance.
- Geometry scores are SOFT HINTS because reference detection may be noisy.
- For relation queries, jointly inspect target and correct reference object.
- Depth may be used ONLY if depth is allowed. Smaller millimeter depth means closer to camera.
- For in-front-of / behind / nearest / farthest queries, depth is important, but tiny depth differences may be sensor noise; combine depth with visual/spatial evidence.
- Infrared may be used ONLY if infrared is allowed.
- Never invent an irrelevant reason. If depth is not allowed, do not use depth.
- ordinal_rank is only a hint among semantically/relation-valid targets. Do not choose a wrong object just because it is geometrically extreme.
- For count=1 prefer a tight box around exactly one requested instance, but do not reject a valid candidate merely because another is smaller.
- For count>1 choose a G* group candidate covering all requested objects.
- Select ONLY an existing candidate ID. Do not generate coordinates.

Return ONLY JSON:
{{"selected_candidate_id":"T0 or G0 or NONE","used_evidence":["RGB"],"reason":"brief grounded reason"}}"""
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(qwen_model.device)
    with torch.inference_mode():
        out = qwen_model.generate(**inputs, max_new_tokens=180, do_sample=False)
    n = inputs.input_ids.shape[1]
    raw = qwen_processor.batch_decode(out[:, n:], skip_special_tokens=True)[0]
    return extract_json_object(raw), raw


def cleanup_query_runtime(scene_id: str, qid: str):
    if ARGS.keep_runtime_images:
        return
    qdir = RUNTIME_DIR / scene_id
    for suffix in ("_targets.jpg", "_crops.jpg", "_refs.jpg"):
        try:
            (qdir / f"{qid}{suffix}").unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================
# Checkpoint / submission state
# ============================================================

predictions: Dict[str, dict] = {}
failures: Dict[str, dict] = {}
run_meta = {
    "version": "V1-Stage2.2-Hybrid-full",
    "started_at": now_str(),
    "queries_path": str(QUERIES_PATH),
    "total_dataset_queries": len(ORIGINAL_QUERIES),
    "total_dataset_scenes": len(SCENES),
    "run_scene_ids": scene_ids,
    "args": vars(ARGS),
}

if ARGS.resume:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        if isinstance(ckpt, dict) and "predictions" in ckpt:
            predictions = ckpt.get("predictions", {})
            failures = ckpt.get("failures", {})
            print(f"[{now_str()}] Resume checkpoint: loaded {len(predictions)} predictions")

    # Per-scene files are the most durable / recent state and avoid rewriting a huge JSON every scene.
    scene_files = sorted(SCENE_RESULTS_DIR.glob("*.json"))
    loaded_scene_files = 0
    for sf in scene_files:
        try:
            with open(sf, "r", encoding="utf-8") as f:
                sd = json.load(f)
            if isinstance(sd, dict):
                predictions.update(sd.get("predictions", {}))
                failures.update(sd.get("failures", {}))
                for q in sd.get("predictions", {}):
                    failures.pop(q, None)
                loaded_scene_files += 1
        except Exception as e:
            print(f"WARNING: failed to load scene result {sf}: {e}")
    if loaded_scene_files:
        print(f"[{now_str()}] Resume scene files: {loaded_scene_files}; total predictions now {len(predictions)}")


def build_submission(preds: Dict[str, dict]) -> dict:
    out = {}
    for qid, original in ORIGINAL_QUERIES.items():
        if qid not in preds or preds[qid].get("bbox_normalized") is None:
            continue
        item = dict(original)
        item["bbox"] = preds[qid]["bbox_normalized"]
        out[qid] = item
    return out


def save_scene_result(scene_id: str):
    qids = SCENES[scene_id]
    obj = {
        "scene_id": scene_id,
        "saved_at": now_str(),
        "predictions": {q: predictions[q] for q in qids if q in predictions},
        "failures": {q: failures[q] for q in qids if q in failures},
    }
    atomic_json_dump(obj, SCENE_RESULTS_DIR / f"{scene_id}.json")


def save_state():
    ckpt = {
        "meta": run_meta,
        "predictions": predictions,
        "failures": failures,
        "saved_at": now_str(),
    }
    atomic_json_dump(ckpt, CHECKPOINT_PATH)
    atomic_json_dump(build_submission(predictions), SUBMISSION_PATH)
    atomic_json_dump(failures, FAILURES_PATH)


# ============================================================
# Main loop
# ============================================================

start_time = time.time()
initial_done = len(predictions)
processed_this_run = 0
scene_done_this_run = 0

for scene_pos, scene_id in enumerate(scene_ids, start=1):
    qids = SCENES[scene_id]
    # If all qids are already predicted, skip whole scene.
    if all(qid in predictions and predictions[qid].get("bbox_normalized") is not None for qid in qids):
        print(f"[{now_str()}] [{scene_pos}/{len(scene_ids)}] scene {scene_id}: already complete, skip")
        continue

    scene_t0 = time.time()
    first_item = ORIGINAL_QUERIES[qids[0]]
    visible_path = DATA_ROOT / first_item["visible"]
    ir_path = DATA_ROOT / first_item["infrared"]
    depth_path = DATA_ROOT / first_item["depth"]

    print("\n" + "=" * 100)
    print(f"[{now_str()}] [{scene_pos}/{len(scene_ids)}] SCENE {scene_id} | {len(qids)} queries")
    print(f"RGB: {visible_path}")

    try:
        ctx = SceneContext(scene_id, visible_path, ir_path, depth_path)
    except Exception as e:
        print(f"SCENE LOAD ERROR {scene_id}: {e}")
        for qid in qids:
            failures[qid] = {"stage": "scene_load", "error": repr(e), "traceback": traceback.format_exc()[-4000:]}
        save_scene_result(scene_id)
        if scene_done_this_run % max(1, ARGS.checkpoint_every_scenes) == 0:
            save_state()
        continue

    # Parse all unfinished queries in the scene at once.
    unfinished = [qid for qid in qids if qid not in predictions or predictions[qid].get("bbox_normalized") is None]
    try:
        parsed_map = parse_scene_queries(unfinished)
    except Exception as e:
        print(f"PARSER BATCH ERROR scene {scene_id}: {e}")
        parsed_map = {qid: sanitize_parse(qid, ORIGINAL_QUERIES[qid]["query"], None) for qid in unfinished}

    for qi, qid in enumerate(unfinished, start=1):
        q_t0 = time.time()
        record = ORIGINAL_QUERIES[qid]
        query = record["query"]
        parsed = parsed_map[qid]
        routing = get_routing(parsed, query)
        fallback_used = None

        print("-" * 100)
        print(f"[{now_str()}] {scene_id} [{qi}/{len(unfinished)}] {qid}: {query}")
        print("PARSED:", json.dumps(parsed, ensure_ascii=False))
        print("ROUTING:", routing)

        try:
            # Candidate generation.
            targets = make_target_candidates(ctx.rgb_pil, parsed, query)
            references = make_reference_candidates(ctx.rgb_pil, parsed)

            if not targets:
                # Emergency candidate: keep pipeline valid and make failure explicit.
                targets = [{
                    "bbox": [0.0, 0.0, float(ctx.W), float(ctx.H)],
                    "score": 0.0,
                    "prompt": "emergency full-frame",
                    "matched_prompts": ["emergency full-frame"],
                    "candidate_id": "T0",
                }]
                fallback_used = "no_dino_target_full_frame"

            scored_targets, suppressed = prepare_targets(ctx, parsed, targets, references, routing)
            count = int(parsed.get("count", 1) or 1)
            if count == 1:
                shortlist = make_shortlist(scored_targets, routing, parsed.get("ordinal", "none"))
            else:
                shortlist = build_groups(ctx, scored_targets, count, references, routing)
                if not shortlist:
                    # If grouping fails, keep individual candidates so Qwen can still choose something.
                    shortlist = make_shortlist(scored_targets, routing, parsed.get("ordinal", "none"))
                    fallback_used = (fallback_used + "+" if fallback_used else "") + "group_fallback_to_individuals"

            if not shortlist:
                shortlist = scored_targets[:1]

            print("TARGETS:", len(targets), "REF GROUPS:", [len(x.get("candidates", [])) for x in references])
            print("SHORTLIST:", [(c["candidate_id"], c.get("score"), c.get("hybrid_priority")) for c in shortlist])
            if suppressed:
                print("SUPPRESSED:", suppressed)

            # Qwen final judge. OOM retry uses a lighter visual payload.
            try:
                decision, raw = qwen_judge(ctx, qid, query, parsed, routing, shortlist, references, light=False)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("QWEN OOM -> retry light mode")
                    torch.cuda.empty_cache()
                    gc.collect()
                    decision, raw = qwen_judge(ctx, qid, query, parsed, routing, shortlist[:5], references, light=True)
                    fallback_used = (fallback_used + "+" if fallback_used else "") + "qwen_light_retry"
                else:
                    raise

            cmap = {c["candidate_id"]: c for c in shortlist}
            selected_id = decision.get("selected_candidate_id") if isinstance(decision, dict) else None
            chosen = cmap.get(selected_id)

            if chosen is None:
                # Always emit a valid bbox for platform submission.
                chosen = max(
                    shortlist,
                    key=lambda c: float(c.get("hybrid_priority", c.get("score", 0.0))),
                )
                fallback_used = (fallback_used + "+" if fallback_used else "") + "judge_invalid_fallback_top"
                if not isinstance(decision, dict):
                    decision = {}
                decision = {
                    **decision,
                    "selected_candidate_id": chosen["candidate_id"],
                    "fallback_override": True,
                    "reason": decision.get("reason", "Qwen did not return a valid existing candidate; fallback to top shortlist candidate."),
                }

            bbox = clamp_bbox(chosen["bbox"], ctx.W, ctx.H)
            bbox_norm = normalize_bbox(bbox, ctx.W, ctx.H)

            predictions[qid] = {
                "query": query,
                "scene_id": scene_id,
                "parsed": parsed,
                "routing": routing,
                "suppressed": suppressed,
                "target_candidate_count": len(targets),
                "reference_candidate_counts": [len(x.get("candidates", [])) for x in references],
                "shortlist": shortlist,
                "decision": decision,
                "bbox": bbox,
                "bbox_normalized": bbox_norm,
                "fallback_used": fallback_used,
                "qwen_raw": raw,
                "runtime_seconds": round(time.time() - q_t0, 3),
            }
            failures.pop(qid, None)
            print("SELECTED:", chosen["candidate_id"], "BBOX_NORM:", bbox_norm, "FALLBACK:", fallback_used)

        except Exception as e:
            print(f"QUERY ERROR {qid}: {e}")
            traceback.print_exc()
            failures[qid] = {
                "stage": "query",
                "error": repr(e),
                "traceback": traceback.format_exc()[-6000:],
                "query": query,
                "parsed": parsed,
            }
            # Keep it unfinished so --resume will retry it.

        finally:
            processed_this_run += 1
            cleanup_query_runtime(scene_id, qid)
            if processed_this_run % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # Save this scene atomically. Full checkpoint/submission is rewritten less frequently.
    scene_done_this_run += 1
    save_scene_result(scene_id)
    if scene_done_this_run % max(1, ARGS.checkpoint_every_scenes) == 0:
        save_state()
    ctx.cleanup()
    if not ARGS.keep_runtime_images:
        try:
            qdir = RUNTIME_DIR / scene_id
            if qdir.exists() and not any(qdir.iterdir()):
                qdir.rmdir()
        except Exception:
            pass

    elapsed = time.time() - start_time
    done_now = max(1, processed_this_run)
    avg_q = elapsed / done_now
    remaining_queries = sum(
        1 for sid in scene_ids[scene_pos:] for q in SCENES[sid]
        if q not in predictions or predictions[q].get("bbox_normalized") is None
    )
    eta = avg_q * remaining_queries
    print(
        f"[{now_str()}] SCENE {scene_id} done in {human_eta(time.time() - scene_t0)} | "
        f"run processed={processed_this_run} | total predictions={len(predictions)} | "
        f"failures={len(failures)} | avg/query={avg_q:.2f}s | ETA≈{human_eta(eta)}"
    )


# ============================================================
# Final summary
# ============================================================

save_state()
submission = build_submission(predictions)
selected_run_qids = [q for sid in scene_ids for q in SCENES[sid]]
selected_complete = sum(1 for q in selected_run_qids if q in predictions and predictions[q].get("bbox_normalized") is not None)
summary = {
    **run_meta,
    "finished_at": now_str(),
    "elapsed_seconds": round(time.time() - start_time, 2),
    "predictions_total": len(predictions),
    "submission_entries": len(submission),
    "selected_run_queries": len(selected_run_qids),
    "selected_run_complete": selected_complete,
    "failure_count": len(failures),
    "fallback_count": sum(1 for x in predictions.values() if x.get("fallback_used")),
    "peak_cuda_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
    "checkpoint": str(CHECKPOINT_PATH),
    "submission": str(SUBMISSION_PATH),
    "failures": str(FAILURES_PATH),
}
atomic_json_dump(summary, SUMMARY_PATH)

print("\n" + "=" * 100)
print("V1 FULL RUN FINISHED")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("IMPORTANT: Before platform upload, make sure submission_entries equals the expected query count for the run.")
