#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIC V2-P0 Pilot Runner
======================

Purpose
-------
Run only the fixed Pilot-50 scenes on top of the proven V1 Stage2.2 checkpoint.

V2-P0 changes:
1) Reuse V1 parsed queries instead of calling Qwen parser again.
2) Load Qwen3-VL and Grounding DINO only once for the whole Pilot run.
3) Load RGB / IR / Depth once per scene.
4) Scene-level Grounding-DINO cache: repeated prompts in the same scene are inferred once.
5) Scene-level ordinal-family joint solver for repeated objects such as
   first/second/third/fourth from left/right.
6) Depth V2: valid-ratio + strong/weak/ambiguous margin instead of plain < / >.
7) Recovery for V1 judge-invalid/NONE fallback queries.
8) IR routing bug fix: whole-word/semantic matching (so "photo" no longer triggers "hot").
9) Conservative override: if V2 is not sufficiently confident, keep the V1 bbox.

Outputs
-------
<output-dir>/
    scene_results/
    checkpoint_predictions_v2_pilot.json
    submission_v2_pilot.json
    change_report.json
    failures.json
    run_summary.json

The pilot submission contains ALL queries belonging to the selected Pilot scenes.
Ordinary/non-overridden queries simply reuse the V1 bbox, so it can be merged safely
into the full V1 submission with merge_v2_pilot_into_v1.py.
"""

import argparse
import gc
import itertools
import json
import math
import os
import re
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
    p = argparse.ArgumentParser(description="Run AIC V2-P0 on fixed Pilot scenes")
    p.add_argument("--project-root", default="/root/autodl-tmp/aic_v1_project")
    p.add_argument("--data-root", default=None)
    p.add_argument("--queries", default=None)
    p.add_argument("--scene-list", default="configs/V2_PILOT_50.txt")
    p.add_argument("--v1-checkpoint", default="outputs/v1_full/checkpoint_predictions.json")
    p.add_argument("--v1-submission", default="outputs/v1_full/submission_v1_06011_backup.json")
    p.add_argument("--output-dir", default="outputs/v2_pilot50_p0")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--routes",
        default="ordinal,depth,fallback,irfix",
        help="Comma-separated: ordinal,depth,fallback,irfix",
    )
    p.add_argument("--dino-threshold", type=float, default=0.12)
    p.add_argument("--dino-text-threshold", type=float, default=0.10)
    p.add_argument("--dino-batch", type=int, default=2)
    p.add_argument("--max-targets", type=int, default=16)
    p.add_argument("--max-refs", type=int, default=5)
    p.add_argument("--max-shortlist", type=int, default=8)
    p.add_argument("--max-group-base", type=int, default=7)
    p.add_argument("--tile-overlap", type=float, default=0.15)
    p.add_argument("--keep-runtime-images", action="store_true")
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

def abs_from_project(p: str) -> Path:
    x = Path(p)
    return x if x.is_absolute() else PROJECT_ROOT / x

SCENE_LIST_PATH = abs_from_project(ARGS.scene_list)
V1_CHECKPOINT_PATH = abs_from_project(ARGS.v1_checkpoint)
V1_SUBMISSION_PATH = abs_from_project(ARGS.v1_submission)
OUTPUT_DIR = abs_from_project(ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENE_RESULTS_DIR = OUTPUT_DIR / "scene_results"
SCENE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = OUTPUT_DIR / "runtime_cache"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_predictions_v2_pilot.json"
PILOT_SUBMISSION_PATH = OUTPUT_DIR / "submission_v2_pilot.json"
CHANGE_REPORT_PATH = OUTPUT_DIR / "change_report.json"
FAILURES_PATH = OUTPUT_DIR / "failures.json"
SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"

HF_HUB = PROJECT_ROOT / "models/hf_cache/hub"
ENABLED_ROUTES = {x.strip().lower() for x in ARGS.routes.split(",") if x.strip()}

MAX_TARGETS = ARGS.max_targets
MAX_REFS = ARGS.max_refs
MAX_SHORTLIST = ARGS.max_shortlist
MAX_GROUP_BASE = ARGS.max_group_base


# ============================================================
# Utilities
# ============================================================

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json_dump(obj: Any, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def find_snapshot(repo_dir_name: str) -> str:
    root = HF_HUB / repo_dir_name / "snapshots"
    if not root.exists():
        raise RuntimeError(f"Model cache not found: {root}")
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
    raise RuntimeError(f"No valid snapshot found under: {root}")


def extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
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
            pass
    return None


def clamp_bbox(b, w, h):
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


def normalize_bbox(b, w, h):
    b = clamp_bbox(b, w, h)
    return [
        round(max(0.0, min(1.0, b[0] / w)), 6),
        round(max(0.0, min(1.0, b[1] / h)), 6),
        round(max(0.0, min(1.0, b[2] / w)), 6),
        round(max(0.0, min(1.0, b[3] / h)), 6),
    ]


def denormalize_bbox(b, w, h):
    return clamp_bbox([b[0] * w, b[1] * h, b[2] * w, b[3] * h], w, h)


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
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def human_eta(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def same_bbox(a, b, eps=1e-6):
    return len(a) == len(b) == 4 and all(abs(float(x) - float(y)) <= eps for x, y in zip(a, b))


# ============================================================
# Data / baseline
# ============================================================

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    ORIGINAL_QUERIES: Dict[str, dict] = json.load(f)

with open(V1_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
    v1_ckpt = json.load(f)

if "predictions" not in v1_ckpt:
    raise RuntimeError("V1 checkpoint has no 'predictions' object")
V1_PREDICTIONS: Dict[str, dict] = v1_ckpt["predictions"]

with open(V1_SUBMISSION_PATH, "r", encoding="utf-8") as f:
    V1_SUBMISSION: Dict[str, dict] = json.load(f)

PILOT_SCENES = []
for line in SCENE_LIST_PATH.read_text(encoding="utf-8").splitlines():
    x = line.strip()
    if x and not x.startswith("#"):
        PILOT_SCENES.append(x)

if len(PILOT_SCENES) != len(set(PILOT_SCENES)):
    raise RuntimeError("Duplicate scene IDs in scene list")

ALL_SCENE_QIDS: Dict[str, List[str]] = defaultdict(list)
for qid in ORIGINAL_QUERIES:
    ALL_SCENE_QIDS[qid.split("_", 1)[0]].append(qid)
for sid in ALL_SCENE_QIDS:
    ALL_SCENE_QIDS[sid] = sorted(ALL_SCENE_QIDS[sid])

for sid in PILOT_SCENES:
    if sid not in ALL_SCENE_QIDS:
        raise RuntimeError(f"Pilot scene not found in queries.json: {sid}")

PILOT_QIDS = [q for sid in PILOT_SCENES for q in ALL_SCENE_QIDS[sid]]

missing_v1 = [q for q in PILOT_QIDS if q not in V1_PREDICTIONS or q not in V1_SUBMISSION]
if missing_v1:
    raise RuntimeError(f"Pilot queries missing V1 prediction/submission: {missing_v1[:20]}")

print(f"[{now_str()}] Pilot scenes : {len(PILOT_SCENES)}")
print(f"[{now_str()}] Pilot queries: {len(PILOT_QIDS)}")
print(f"[{now_str()}] Routes       : {sorted(ENABLED_ROUTES)}")


# ============================================================
# Model loading - ONCE for entire Pilot run
# ============================================================

QWEN_PATH = find_snapshot("models--Qwen--Qwen3-VL-8B-Instruct")
DINO_PATH = find_snapshot("models--IDEA-Research--grounding-dino-base")

print(f"[{now_str()}] Loading Qwen3-VL ONCE: {QWEN_PATH}")
qwen_processor = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True)
qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
qwen_model.eval()

print(f"[{now_str()}] Loading Grounding DINO ONCE: {DINO_PATH}")
dino_processor = AutoProcessor.from_pretrained(DINO_PATH, local_files_only=True)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_PATH, local_files_only=True
).to("cuda")
dino_model.eval()


# ============================================================
# Query routing helpers
# ============================================================

IR_PATTERNS = [
    r"\bwarm\b", r"\bwarmer\b", r"\bwarmest\b",
    r"\bhot\b", r"\bhottest\b",
    r"\bheat\b", r"\bthermal\b", r"\binfrared\b",
    r"\bcold\b", r"\bcolder\b", r"\bcoldest\b",
    r"\bcool\b", r"\bcooler\b",
]

DEPTH_ABSOLUTE_PATTERNS = [
    r"\bclosest\b", r"\bnearest\b", r"\bfarthest\b",
    r"\bfrontmost\b", r"\brearmost\b",
]
DEPTH_PAIRWISE_PATTERNS = [
    r"\bin front of\b", r"\bfarther than\b", r"\bcloser than\b",
]
DEPTH_VISUAL_BEHIND_PATTERNS = [
    r"\bbehind\b",
]

ORDINAL_RE = re.compile(
    r"\b(leftmost|rightmost|topmost|bottommost|uppermost|lowermost|"
    r"first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b",
    re.I,
)


def regex_any(patterns, text):
    return any(re.search(p, text, flags=re.I) for p in patterns)


def semantic_ir(query: str) -> bool:
    return regex_any(IR_PATTERNS, query)


def depth_type(query: str, parsed: dict) -> Optional[str]:
    q = query.lower()
    rels = [str(c.get("relation", "")).lower() for c in parsed.get("constraints", [])]
    if regex_any(DEPTH_ABSOLUTE_PATTERNS, q):
        return "absolute_ordinal"
    if "in front of" in rels or regex_any(DEPTH_PAIRWISE_PATTERNS, q):
        return "pairwise_metric"
    if "behind" in rels or regex_any(DEPTH_VISUAL_BEHIND_PATTERNS, q):
        return "visual_behind"
    return None


def ordinal_info(parsed: dict, query: str):
    o = str(parsed.get("ordinal") or "none").strip().lower()
    q = query.lower()
    combined = f"{o} {q}"

    if o == "none" and not ORDINAL_RE.search(q):
        return None

    if "from right" in combined or "rightmost" in combined:
        axis, reverse = "x", True
    elif "from left" in combined or "leftmost" in combined:
        axis, reverse = "x", False
    elif "from bottom" in combined or "bottommost" in combined or "lowermost" in combined:
        axis, reverse = "y", True
    elif "from top" in combined or "topmost" in combined or "uppermost" in combined:
        axis, reverse = "y", False
    else:
        # Do not force a joint spatial ordinal if direction is not clear.
        return None

    index = 0
    if re.search(r"\b(second|2nd)\b", combined):
        index = 1
    elif re.search(r"\b(third|3rd)\b", combined):
        index = 2
    elif re.search(r"\b(fourth|4th)\b", combined):
        index = 3
    elif re.search(r"\b(fifth|5th)\b", combined):
        index = 4

    return {"axis": axis, "reverse": reverse, "index": index, "raw": o}


def canonical_target(parsed: dict):
    x = str(parsed.get("target_class") or "").lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x


def canonical_attributes(parsed: dict):
    attrs = parsed.get("target_attributes") or []
    if not isinstance(attrs, list):
        attrs = [attrs]
    clean = []
    for x in attrs:
        x = re.sub(r"\s+", " ", str(x).lower().strip())
        if x:
            clean.append(x)
    return tuple(sorted(set(clean)))


def canonical_constraints(parsed: dict):
    out = []
    for c in parsed.get("constraints", []) or []:
        if not isinstance(c, dict):
            continue
        rel = re.sub(r"\s+", " ", str(c.get("relation") or "").lower().strip())
        ref = re.sub(r"\s+", " ", str(c.get("reference_object") or "").lower().strip())
        if rel or ref:
            out.append((rel, ref))
    return tuple(sorted(out))


def needs_v2_route(qid: str):
    v1 = V1_PREDICTIONS[qid]
    query = v1["query"]
    parsed = v1.get("parsed", {})
    reasons = []

    if "ordinal" in ENABLED_ROUTES and ordinal_info(parsed, query) is not None:
        reasons.append("ordinal")
    if "depth" in ENABLED_ROUTES and depth_type(query, parsed) is not None:
        reasons.append("depth")
    if "fallback" in ENABLED_ROUTES and v1.get("fallback_used"):
        reasons.append("fallback")
    if "irfix" in ENABLED_ROUTES:
        old_ir = bool(v1.get("routing", {}).get("infrared"))
        new_ir = semantic_ir(query)
        if old_ir != new_ir:
            reasons.append("irfix")

    return reasons


# ============================================================
# Scene context + scene-level DINO cache
# ============================================================

def dedupe_prompts(prompts):
    out, seen = [], set()
    for p in prompts:
        p = re.sub(r"\s+", " ", str(p).strip().strip("."))
        if not p:
            continue
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


class SceneContext:
    def __init__(self, scene_id, visible_path, ir_path, depth_path):
        self.scene_id = scene_id
        self.visible_path = Path(visible_path)
        self.ir_path = Path(ir_path)
        self.depth_path = Path(depth_path)

        # Read each modality ONCE per scene.
        self.rgb_pil = Image.open(self.visible_path).convert("RGB")
        self.W, self.H = self.rgb_pil.size

        self.depth = cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED)
        if self.depth is None:
            self.depth = np.zeros((self.H, self.W), dtype=np.uint16)
        if self.depth.ndim == 3:
            self.depth = self.depth[..., 0]

        self.dino_cache: Dict[Tuple, List[dict]] = {}
        self.tiled_cache: Dict[Tuple, List[dict]] = {}

        self.depth_vis_path = RUNTIME_DIR / f"{scene_id}_depth_vis.png"
        self._build_depth_vis()

    def _build_depth_vis(self):
        valid = self.depth > 0
        vis = np.zeros((self.H, self.W), dtype=np.uint8)
        if valid.any():
            vals = self.depth[valid].astype(np.float32)
            lo, hi = np.percentile(vals, [2, 98])
            norm = (self.depth.astype(np.float32) - lo) / max(float(hi - lo), 1.0)
            norm = np.clip(norm, 0, 1)
            vis = ((1.0 - norm) * 255).astype(np.uint8)
            vis[~valid] = 0
        Image.fromarray(vis).convert("RGB").save(self.depth_vis_path)

    def depth_stats(self, box, ratio=0.65):
        x1, y1, x2, y2 = map(float, box)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ww, hh = (x2 - x1) * ratio, (y2 - y1) * ratio
        ix1 = int(max(0, cx - ww / 2))
        ix2 = int(min(self.W, cx + ww / 2))
        iy1 = int(max(0, cy - hh / 2))
        iy2 = int(min(self.H, cy + hh / 2))

        roi = self.depth[iy1:iy2, ix1:ix2]
        if roi.size == 0:
            return {"valid_ratio": 0.0, "median_mm": None, "p25_mm": None, "p75_mm": None}

        # V1-compatible validity, but keep robust quartiles.
        mask = (roi > 300) & (roi <= 19999)
        vals = roi[mask]
        vr = float(vals.size / roi.size)
        if vals.size == 0:
            return {"valid_ratio": round(vr, 4), "median_mm": None, "p25_mm": None, "p75_mm": None}

        return {
            "valid_ratio": round(vr, 4),
            "median_mm": int(np.median(vals)),
            "p25_mm": int(np.percentile(vals, 25)),
            "p75_mm": int(np.percentile(vals, 75)),
        }

    def dino_detect(self, prompts, threshold=None, text_threshold=None):
        prompts = dedupe_prompts(prompts)
        threshold = ARGS.dino_threshold if threshold is None else float(threshold)
        text_threshold = ARGS.dino_text_threshold if text_threshold is None else float(text_threshold)
        key = (tuple(p.lower() for p in prompts), round(threshold, 4), round(text_threshold, 4))
        if key in self.dino_cache:
            return [dict(x) for x in self.dino_cache[key]]

        candidates = self._dino_detect_image(
            self.rgb_pil, prompts, threshold, text_threshold, offset=(0, 0)
        )
        self.dino_cache[key] = [dict(x) for x in candidates]
        return candidates

    def tiled_dino_detect(self, prompts, threshold=0.07, text_threshold=0.07):
        prompts = dedupe_prompts(prompts)
        key = (
            tuple(p.lower() for p in prompts),
            round(float(threshold), 4),
            round(float(text_threshold), 4),
            round(float(ARGS.tile_overlap), 3),
        )
        if key in self.tiled_cache:
            return [dict(x) for x in self.tiled_cache[key]]

        ov = max(0.0, min(0.40, float(ARGS.tile_overlap)))
        W, H = self.W, self.H

        # 2x2 overlapping tiles.
        half_w, half_h = W / 2.0, H / 2.0
        pad_w, pad_h = half_w * ov, half_h * ov
        tiles = [
            (0, 0, int(min(W, half_w + pad_w)), int(min(H, half_h + pad_h))),
            (int(max(0, half_w - pad_w)), 0, W, int(min(H, half_h + pad_h))),
            (0, int(max(0, half_h - pad_h)), int(min(W, half_w + pad_w)), H),
            (int(max(0, half_w - pad_w)), int(max(0, half_h - pad_h)), W, H),
        ]

        out = []
        for x1, y1, x2, y2 in tiles:
            crop = self.rgb_pil.crop((x1, y1, x2, y2))
            out.extend(
                self._dino_detect_image(
                    crop,
                    prompts,
                    threshold,
                    text_threshold,
                    offset=(x1, y1),
                )
            )

        self.tiled_cache[key] = [dict(x) for x in out]
        return out

    def _dino_detect_image(self, image, prompts, threshold, text_threshold, offset=(0, 0)):
        prompts = dedupe_prompts(prompts)
        if not prompts:
            return []

        ox, oy = offset
        w, h = image.size
        candidates = []

        for start in range(0, len(prompts), max(1, ARGS.dino_batch)):
            batch = prompts[start:start + max(1, ARGS.dino_batch)]
            texts = [p + "." for p in batch]
            images = [image] * len(texts)

            inputs = dino_processor(images=images, text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = dino_model(**inputs)

            results = dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=threshold,
                text_threshold=text_threshold,
                target_sizes=[(h, w)] * len(texts),
            )

            for prompt, result in zip(batch, results):
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                if hasattr(boxes, "detach"):
                    boxes = boxes.detach().cpu().tolist()
                if hasattr(scores, "detach"):
                    scores = scores.detach().cpu().tolist()

                for b, s in zip(boxes, scores):
                    gb = [
                        float(b[0]) + ox, float(b[1]) + oy,
                        float(b[2]) + ox, float(b[3]) + oy,
                    ]
                    gb = clamp_bbox(gb, self.W, self.H)
                    if area(gb) < 4:
                        continue
                    candidates.append({
                        "bbox": gb,
                        "score": round(float(s), 6),
                        "prompt": prompt + ".",
                        "matched_prompts": [prompt + "."],
                    })
        return candidates

    def cleanup(self):
        if not ARGS.keep_runtime_images:
            try:
                self.depth_vis_path.unlink(missing_ok=True)
            except Exception:
                pass


# ============================================================
# Candidate helpers
# ============================================================

def merge_candidates(candidates, max_keep):
    ordered = sorted(candidates, key=lambda x: float(x["score"]), reverse=True)
    kept = []

    for c in ordered:
        merged = False
        for k in kept:
            ov = iou(c["bbox"], k["bbox"])
            cont = containment(c["bbox"], k["bbox"])
            if ov >= 0.72 or cont >= 0.93:
                for p in c.get("matched_prompts", [c.get("prompt")]):
                    if p and p not in k.setdefault("matched_prompts", []):
                        k["matched_prompts"].append(p)
                merged = True
                break
        if not merged:
            kept.append(dict(c))

    kept = sorted(kept, key=lambda x: float(x["score"]), reverse=True)[:max_keep]
    for i, c in enumerate(kept):
        c["candidate_id"] = f"T{i}"
    return kept


def suppress_group_containers(candidates):
    """
    Remove obvious boxes that cover multiple smaller instances.
    Only used before scene-level ordinal sorting.
    """
    if len(candidates) < 3:
        return candidates, []

    candidates = [dict(x) for x in candidates]
    areas = [area(c["bbox"]) for c in candidates if area(c["bbox"]) > 0]
    if not areas:
        return candidates, []

    med = float(np.median(areas))
    suppressed = set()
    reasons = []

    for i, c in enumerate(candidates):
        ca = area(c["bbox"])
        if ca < 2.5 * med:
            continue

        overlapping_small = []
        for j, other in enumerate(candidates):
            if i == j:
                continue
            oa = area(other["bbox"])
            if oa <= 0 or oa >= ca * 0.75:
                continue
            if intersection(c["bbox"], other["bbox"]) / oa >= 0.65:
                overlapping_small.append(j)

        if len(overlapping_small) >= 2:
            best_small = max(float(candidates[j]["score"]) for j in overlapping_small)
            if float(c["score"]) <= best_small + 0.10:
                suppressed.add(i)
                reasons.append({
                    "suppressed": c["candidate_id"],
                    "reason": "group_container",
                    "area_ratio_to_median": round(ca / max(med, 1.0), 3),
                    "covered_small_count": len(overlapping_small),
                })

    out = [c for i, c in enumerate(candidates) if i not in suppressed]
    for i, c in enumerate(out):
        c["candidate_id"] = f"T{i}"
    return out, reasons


def make_target_candidates(ctx, parsed, query, recovery=False):
    prompts = parsed.get("target_dino_prompts", []) or []
    core = parsed.get("target_class") or query
    if core.lower() not in {str(x).lower() for x in prompts}:
        prompts = list(prompts) + [core]

    threshold = 0.08 if recovery else ARGS.dino_threshold
    text_threshold = 0.08 if recovery else ARGS.dino_text_threshold
    cand = ctx.dino_detect(prompts, threshold=threshold, text_threshold=text_threshold)

    if recovery:
        cand += ctx.dino_detect(
            [query, core], threshold=0.06, text_threshold=0.06
        )
        cand += ctx.tiled_dino_detect(
            prompts + [core], threshold=0.06, text_threshold=0.06
        )

    merged = merge_candidates(cand, MAX_TARGETS)

    if not merged:
        merged = merge_candidates(
            ctx.dino_detect([core], threshold=0.05, text_threshold=0.05),
            MAX_TARGETS,
        )
    return merged


def make_reference_candidates(ctx, parsed):
    refs = []
    for ridx, c in enumerate(parsed.get("constraints", [])):
        prompts = c.get("reference_dino_prompts", []) or [c.get("reference_object", "")]
        ref_obj = c.get("reference_object", "")
        if ref_obj and ref_obj.lower() not in {str(x).lower() for x in prompts}:
            prompts = list(prompts) + [ref_obj]

        cand = merge_candidates(ctx.dino_detect(prompts), MAX_REFS)
        for j, x in enumerate(cand):
            x["candidate_id"] = f"R{ridx}_{j}"

        refs.append({
            "relation": str(c.get("relation", "")).strip().lower(),
            "reference_object": ref_obj,
            "candidates": cand,
        })
    return refs


# ============================================================
# Depth V2
# ============================================================

def depth_margin_state(target_stats, ref_stats, relation):
    td = target_stats.get("median_mm")
    rd = ref_stats.get("median_mm")
    tvr = float(target_stats.get("valid_ratio", 0))
    rvr = float(ref_stats.get("valid_ratio", 0))

    if td is None or rd is None or tvr < 0.30 or rvr < 0.30:
        return {
            "state": "invalid",
            "relation_ok": None,
            "delta_mm": None,
            "abs_delta_mm": None,
        }

    signed_delta = rd - td
    abs_delta = abs(signed_delta)
    base = min(td, rd)

    strong_thr = max(150.0, 0.015 * base)
    weak_thr = max(50.0, 0.005 * base)

    if relation == "in front of":
        ok = td < rd
    elif relation == "behind":
        ok = td > rd
    else:
        ok = None

    if abs_delta >= strong_thr:
        state = "strong"
    elif abs_delta >= weak_thr:
        state = "weak"
    else:
        state = "ambiguous"

    return {
        "state": state,
        "relation_ok": bool(ok) if ok is not None else None,
        "delta_mm": int(signed_delta),
        "abs_delta_mm": int(abs_delta),
        "strong_threshold_mm": round(strong_thr, 1),
        "weak_threshold_mm": round(weak_thr, 1),
    }


def relation_score(ctx, target, ref, relation, dtype):
    tb, rb = target["bbox"], ref["bbox"]
    tcx, tcy = center(tb)
    rcx, rcy = center(rb)
    hov = horizontal_overlap(tb, rb)
    vov = vertical_overlap(tb, rb)
    dx = abs(tcx - rcx) / ctx.W
    dy = abs(tcy - rcy) / ctx.H
    near_y = max(0.0, 1.0 - dy / 0.35)
    near_center = max(0.0, 1.0 - math.hypot(dx, dy) / 0.45)
    inside_ref = rb[0] <= tcx <= rb[2] and rb[1] <= tcy <= rb[3]

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
    elif relation == "inside":
        score = 0.75 * float(inside_ref) + 0.25 * min(1.0, max(hov, vov))
    elif relation == "left of":
        score = 0.65 * float(tcx < rcx) + 0.35 * near_center
    elif relation == "right of":
        score = 0.65 * float(tcx > rcx) + 0.35 * near_center
    elif relation in {"in front of", "behind"} and dtype in {"pairwise_metric", "visual_behind"}:
        ts = ctx.depth_stats(tb)
        rs = ctx.depth_stats(rb)
        dm = depth_margin_state(ts, rs, relation)
        details.update({
            "target_depth": ts,
            "reference_depth": rs,
            "depth_margin": dm,
        })

        # visual_behind: depth never dominates.
        if dtype == "visual_behind":
            depth_hint = 0.5
            if dm["state"] == "strong" and dm["relation_ok"] is True:
                depth_hint = 0.75
            elif dm["state"] == "strong" and dm["relation_ok"] is False:
                depth_hint = 0.25
            score = 0.35 * depth_hint + 0.35 * max(hov, vov) + 0.30 * near_center
        else:
            # pairwise metric depth:
            if dm["state"] == "strong":
                depth_component = 1.0 if dm["relation_ok"] else 0.0
            elif dm["state"] == "weak":
                depth_component = 0.70 if dm["relation_ok"] else 0.30
            else:
                depth_component = 0.50
            score = 0.65 * depth_component + 0.20 * max(hov, vov) + 0.15 * near_center
    else:
        score = near_center

    return float(score), details


def score_constraints(ctx, target, references, dtype):
    if not references:
        return 1.0, []

    infos = []
    for ref_info in references:
        best = None
        for ref in ref_info["candidates"][:MAX_REFS]:
            rs, details = relation_score(
                ctx, target, ref, ref_info["relation"], dtype
            )
            pair_score = 0.85 * rs + 0.15 * float(ref["score"])
            cur = {
                "reference_id": ref["candidate_id"],
                "reference_detector_score": ref["score"],
                "relation_score": round(rs, 4),
                "pair_score": round(pair_score, 4),
                "details": details,
            }
            if best is None or cur["pair_score"] > best["pair_score"]:
                best = cur

        infos.append({
            "relation": ref_info["relation"],
            "reference_object": ref_info["reference_object"],
            "best_pair": best,
        })

    vals = [
        x["best_pair"]["pair_score"]
        for x in infos
        if x["best_pair"] is not None
    ]
    return (min(vals) if vals else 0.0), infos


def tightness_filter(candidates, count):
    if count != 1:
        return [dict(x) for x in candidates], []

    candidates = [dict(x) for x in candidates]
    suppressed = set()
    reasons = []

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if i in suppressed or j in suppressed:
                continue
            a, b = candidates[i], candidates[j]
            ov = iou(a["bbox"], b["bbox"])
            if ov < 0.40:
                continue

            aa, ab = area(a["bbox"]), area(b["bbox"])
            si, li = (i, j) if aa <= ab else (j, i)
            small, large = candidates[si], candidates[li]
            sa, la = area(small["bbox"]), area(large["bbox"])
            if sa <= 0:
                continue

            area_ratio = la / sa
            ss = float(small["score"])
            ls = float(large["score"])
            if area_ratio >= 1.20 and ss / max(ls, 1e-6) >= 1.50 and ss - ls >= 0.15:
                suppressed.add(li)
                reasons.append({
                    "suppressed": large["candidate_id"],
                    "kept": small["candidate_id"],
                    "iou": round(ov, 3),
                    "area_ratio": round(area_ratio, 3),
                })

    return [x for i, x in enumerate(candidates) if i not in suppressed], reasons


def prepare_targets(ctx, parsed, targets, references, dtype):
    count = int(parsed.get("count", 1) or 1)
    targets, suppressed = tightness_filter(targets, count)
    out = []

    for c in targets:
        c = dict(c)
        rel, rel_info = score_constraints(ctx, c, references, dtype)
        c["relation_score"] = round(rel, 4)
        c["relation_info"] = rel_info
        c["depth"] = ctx.depth_stats(c["bbox"]) if dtype else None

        if not references:
            priority = float(c["score"])
        elif dtype == "pairwise_metric":
            priority = 0.40 * float(c["score"]) + 0.60 * rel
        else:
            priority = 0.75 * float(c["score"]) + 0.25 * rel

        c["hybrid_priority"] = round(priority, 4)
        out.append(c)
    return out, suppressed


def build_groups(ctx, targets, count, references, dtype):
    if count <= 1 or count > 5:
        return []

    base = sorted(targets, key=lambda x: float(x["score"]), reverse=True)[:MAX_GROUP_BASE]
    groups = []

    for combo in itertools.combinations(base, count):
        if any(iou(a["bbox"], b["bbox"]) > 0.35 for a, b in itertools.combinations(combo, 2)):
            continue

        box = union_boxes([x["bbox"] for x in combo])
        det = sum(float(x["score"]) for x in combo) / count
        fake = {"bbox": box, "score": det}
        rel, info = score_constraints(ctx, fake, references, dtype)

        if not references:
            pri = det
        elif dtype == "pairwise_metric":
            pri = 0.40 * det + 0.60 * rel
        else:
            pri = 0.75 * det + 0.25 * rel

        groups.append({
            "candidate_id": "",
            "members": [x["candidate_id"] for x in combo],
            "bbox": [round(v, 2) for v in box],
            "score": round(det, 4),
            "relation_score": round(rel, 4),
            "hybrid_priority": round(pri, 4),
            "relation_info": info,
            "depth": ctx.depth_stats(box) if dtype else None,
        })

    groups.sort(
        key=lambda x: x["hybrid_priority"] if dtype == "pairwise_metric" else x["score"],
        reverse=True,
    )
    groups = groups[:MAX_SHORTLIST]
    for i, g in enumerate(groups):
        g["candidate_id"] = f"G{i}"
    return groups


# ============================================================
# Ordinal joint solver
# ============================================================

def build_ordinal_families(scene_qids):
    families = defaultdict(list)
    for qid in scene_qids:
        v1 = V1_PREDICTIONS[qid]
        parsed = v1.get("parsed", {})
        oi = ordinal_info(parsed, v1["query"])
        if oi is None:
            continue
        # Joint ordinal solving is only safe when target class, target attributes,
        # and reference constraints are the same. Example: "first white car" and
        # "second black car" must NOT share one geometry-only ordering pool.
        key = (
            canonical_target(parsed),
            canonical_attributes(parsed),
            canonical_constraints(parsed),
            oi["axis"],
            oi["reverse"],
        )
        if key[0]:
            families[key].append(qid)
    return families


def solve_ordinal_family(ctx, family_qids):
    """
    Shared candidate pool + one deterministic ordering for the entire family.
    High-confidence only when enough independent candidates exist.
    """
    prompts = []
    max_index = 0
    family_meta = {}

    for qid in family_qids:
        v1 = V1_PREDICTIONS[qid]
        parsed = v1["parsed"]
        oi = ordinal_info(parsed, v1["query"])
        family_meta[qid] = oi
        max_index = max(max_index, oi["index"])
        prompts.extend(parsed.get("target_dino_prompts", []))
        prompts.append(parsed.get("target_class", ""))

    prompts = dedupe_prompts(prompts)
    candidates = merge_candidates(ctx.dino_detect(prompts), max(24, MAX_TARGETS))
    candidates, container_suppressed = suppress_group_containers(candidates)

    # If candidate count is insufficient for the highest ordinal, run tiled recovery.
    if len(candidates) <= max_index:
        extra = ctx.tiled_dino_detect(prompts, threshold=0.06, text_threshold=0.06)
        candidates = merge_candidates(candidates + extra, max(32, MAX_TARGETS * 2))
        candidates, extra_suppressed = suppress_group_containers(candidates)
        container_suppressed.extend(extra_suppressed)

    # Family direction is shared by construction.
    first_oi = family_meta[family_qids[0]]
    axis = first_oi["axis"]
    reverse = first_oi["reverse"]

    if axis == "x":
        ordered = sorted(candidates, key=lambda c: center(c["bbox"])[0], reverse=reverse)
    else:
        ordered = sorted(candidates, key=lambda c: center(c["bbox"])[1], reverse=reverse)

    results = {}

    for qid in family_qids:
        oi = family_meta[qid]
        idx = oi["index"]

        if idx >= len(ordered):
            results[qid] = {
                "confident": False,
                "reason": "insufficient_independent_candidates",
                "candidate_count": len(ordered),
                "needed_index": idx,
                "candidates": ordered[:MAX_SHORTLIST],
                "container_suppressed": container_suppressed,
            }
            continue

        chosen = ordered[idx]

        # Conservative confidence gate.
        enough = len(ordered) > idx
        score_ok = float(chosen.get("score", 0.0)) >= 0.07

        # Check that neighboring ordered centers are not essentially identical.
        sep_ok = True
        if len(ordered) > 1:
            coords = [center(c["bbox"])[0 if axis == "x" else 1] for c in ordered]
            span = ctx.W if axis == "x" else ctx.H
            nearest_sep = min(
                [abs(coords[idx] - coords[j]) for j in range(len(coords)) if j != idx] or [span]
            )
            sep_ok = nearest_sep >= 0.01 * span

        results[qid] = {
            "confident": bool(enough and score_ok and sep_ok),
            "reason": "ordinal_joint_high_confidence" if (enough and score_ok and sep_ok) else "ordinal_joint_low_confidence",
            "selected_candidate": chosen,
            "selected_rank": idx,
            "axis": axis,
            "reverse": reverse,
            "candidate_count": len(ordered),
            "candidates": ordered[:max(MAX_SHORTLIST, idx + 2)],
            "container_suppressed": container_suppressed,
        }

    return results


# ============================================================
# Qwen visual judge
# ============================================================

def draw_overlay(ctx, candidates, path, color="red"):
    img = ctx.rgb_pil.copy()
    draw = ImageDraw.Draw(img)
    for c in candidates:
        x1, y1, x2, y2 = c["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.rectangle([x1, max(0, y1 - 24), x1 + 78, y1], fill="white")
        draw.text((x1 + 2, max(0, y1 - 21)), c["candidate_id"], fill="black")
    img.save(path, quality=90)


def draw_reference_overlay(ctx, references, path):
    img = ctx.rgb_pil.copy()
    draw = ImageDraw.Draw(img)
    for ref_info in references:
        for ref in ref_info.get("candidates", [])[:MAX_REFS]:
            x1, y1, x2, y2 = ref["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
            draw.rectangle([x1, max(0, y1 - 22), x1 + 92, y1], fill="white")
            draw.text((x1 + 2, max(0, y1 - 19)), ref["candidate_id"], fill="green")
    img.save(path, quality=90)


def make_crop_sheet(ctx, candidates, path):
    if not candidates:
        return
    tw, th = 220, 170
    sheet = Image.new("RGB", (tw * len(candidates), th), "white")
    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = map(int, c["bbox"])
        crop = ctx.rgb_pil.crop(
            (max(0, x1), max(0, y1), min(ctx.W, x2), min(ctx.H, y2))
        )
        crop.thumbnail((tw - 16, th - 35))
        tile = Image.new("RGB", (tw, th), "white")
        tile.paste(crop, ((tw - crop.width) // 2, 30))
        ImageDraw.Draw(tile).text((8, 8), c["candidate_id"], fill="black")
        sheet.paste(tile, (i * tw, 0))
    sheet.save(path, quality=90)


def qwen_judge(ctx, qid, query, parsed, routing, candidates, references):
    qdir = RUNTIME_DIR / ctx.scene_id
    qdir.mkdir(parents=True, exist_ok=True)

    overlay = qdir / f"{qid}_targets.jpg"
    crops = qdir / f"{qid}_crops.jpg"
    refs_overlay = qdir / f"{qid}_refs.jpg"

    visual_candidates = candidates[:5]
    draw_overlay(ctx, visual_candidates, overlay)
    make_crop_sheet(ctx, visual_candidates, crops)

    content = [
        {"type": "image", "image": str(ctx.visible_path)},
        {"type": "image", "image": str(overlay)},
        {"type": "image", "image": str(crops)},
    ]

    if references:
        draw_reference_overlay(ctx, references, refs_overlay)
        content.append({"type": "image", "image": str(refs_overlay)})

    if routing.get("depth"):
        content.append({"type": "image", "image": str(ctx.depth_vis_path)})

    if routing.get("infrared") and ctx.ir_path.exists():
        content.append({"type": "image", "image": str(ctx.ir_path)})

    evidence = []
    for c in candidates:
        evidence.append({
            "candidate_id": c["candidate_id"],
            "bbox": c["bbox"],
            "detector_score": c.get("score"),
            "relation_score_hint": c.get("relation_score"),
            "hybrid_priority": c.get("hybrid_priority"),
            "depth": c.get("depth") if routing.get("depth") else None,
            "relation_info": c.get("relation_info"),
            "members": c.get("members"),
        })

    prompt = f"""You are the final visual grounding candidate judge for V2-P0.

QUERY: {query}
Parsed query: {json.dumps(parsed, ensure_ascii=False)}
Routing: {json.dumps(routing, ensure_ascii=False)}
Candidates: {json.dumps(evidence, ensure_ascii=False)}

Rules:
- Select only an existing T* or G* candidate ID, or NONE.
- RGB is primary for object identity and appearance.
- Geometry is only a soft hint.
- Depth is usable only when routing.depth=true.
- Depth margin states:
  strong = meaningful numerical evidence;
  weak = supporting evidence only;
  ambiguous = ignore as a hard relation.
- For visual 'behind' relations, depth is only auxiliary.
- Infrared is usable only when routing.infrared=true.
- Do not invent evidence.
- Prefer a tight box around exactly the requested target.
- Return NONE if none of the supplied candidates reasonably matches the query.

Return ONLY JSON:
{{"selected_candidate_id":"T0 or G0 or NONE","reason":"brief grounded reason"}}"""

    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(qwen_model.device)

    with torch.inference_mode():
        out = qwen_model.generate(**inputs, max_new_tokens=140, do_sample=False)

    n = inputs.input_ids.shape[1]
    raw = qwen_processor.batch_decode(out[:, n:], skip_special_tokens=True)[0]
    return extract_json_object(raw), raw


def cleanup_query_runtime(scene_id, qid):
    if ARGS.keep_runtime_images:
        return
    qdir = RUNTIME_DIR / scene_id
    for suffix in ("_targets.jpg", "_crops.jpg", "_refs.jpg"):
        try:
            (qdir / f"{qid}{suffix}").unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================
# V2 generic rerun
# ============================================================

def corrected_routing(v1, query, parsed):
    dtype = depth_type(query, parsed)
    return {
        "RGB": True,
        "geometry": bool(parsed.get("constraints")) or ordinal_info(parsed, query) is not None,
        "depth": dtype is not None,
        "depth_type": dtype,
        "infrared": semantic_ir(query),
    }


def run_generic_v2(ctx, qid, route_reasons):
    v1 = V1_PREDICTIONS[qid]
    query = v1["query"]
    parsed = v1["parsed"]
    routing = corrected_routing(v1, query, parsed)
    dtype = routing["depth_type"]

    recovery = "fallback" in route_reasons
    targets = make_target_candidates(ctx, parsed, query, recovery=recovery)
    references = make_reference_candidates(ctx, parsed)

    if not targets:
        return {
            "override": False,
            "reason": "no_v2_candidates",
            "routing": routing,
            "route_reasons": route_reasons,
        }

    scored, suppressed = prepare_targets(
        ctx, parsed, targets, references, dtype
    )

    # Absolute nearest/farthest can use target depth ordering directly.
    dtype_abs = dtype == "absolute_ordinal"
    q = query.lower()

    if dtype_abs:
        valid = []
        for c in scored:
            ds = c.get("depth") or ctx.depth_stats(c["bbox"])
            c["depth"] = ds
            if ds.get("median_mm") is not None and ds.get("valid_ratio", 0) >= 0.30:
                valid.append(c)

        if valid:
            if "farthest" in q or "rearmost" in q:
                ordered = sorted(valid, key=lambda c: c["depth"]["median_mm"], reverse=True)
            else:
                ordered = sorted(valid, key=lambda c: c["depth"]["median_mm"])

            chosen = ordered[0]
            # Require meaningful separation from second candidate.
            strong = False
            if len(ordered) == 1:
                strong = True
            else:
                d0 = chosen["depth"]["median_mm"]
                d1 = ordered[1]["depth"]["median_mm"]
                delta = abs(d1 - d0)
                strong = delta >= max(150.0, 0.015 * min(d0, d1))

            if strong:
                return {
                    "override": True,
                    "reason": "absolute_depth_strong",
                    "routing": routing,
                    "route_reasons": route_reasons,
                    "selected_candidate": chosen,
                    "shortlist": ordered[:MAX_SHORTLIST],
                    "suppressed": suppressed,
                    "qwen_raw": None,
                    "decision": {
                        "selected_candidate_id": chosen["candidate_id"],
                        "reason": "strong absolute depth ordinal",
                    },
                }

    count = int(parsed.get("count", 1) or 1)

    if count == 1:
        if dtype == "pairwise_metric":
            shortlist = sorted(scored, key=lambda c: c["hybrid_priority"], reverse=True)[:MAX_SHORTLIST]
        else:
            shortlist = sorted(scored, key=lambda c: float(c["score"]), reverse=True)[:MAX_SHORTLIST]
    else:
        shortlist = build_groups(ctx, scored, count, references, dtype)
        if not shortlist:
            shortlist = sorted(scored, key=lambda c: float(c["score"]), reverse=True)[:MAX_SHORTLIST]

    if not shortlist:
        return {
            "override": False,
            "reason": "empty_shortlist",
            "routing": routing,
            "route_reasons": route_reasons,
        }

    decision, raw = qwen_judge(
        ctx, qid, query, parsed, routing, shortlist, references
    )
    cmap = {c["candidate_id"]: c for c in shortlist}
    selected_id = decision.get("selected_candidate_id") if isinstance(decision, dict) else None
    chosen = cmap.get(selected_id)

    if chosen is None:
        return {
            "override": False,
            "reason": "qwen_none_or_invalid_keep_v1",
            "routing": routing,
            "route_reasons": route_reasons,
            "shortlist": shortlist,
            "suppressed": suppressed,
            "decision": decision,
            "qwen_raw": raw,
        }

    # Conservative confidence gate.
    allow = False
    reason = "qwen_valid_but_not_high_confidence"

    if "fallback" in route_reasons:
        allow = float(chosen.get("score", 0.0)) >= 0.07
        reason = "fallback_recovered" if allow else reason

    if "irfix" in route_reasons:
        # If V1 routing was polluted by substring matching and Qwen can make a valid
        # decision without the false IR input, allow the correction.
        allow = True
        reason = "ir_routing_corrected"

    if dtype == "pairwise_metric":
        strong_depth = False
        for ri in chosen.get("relation_info", []):
            bp = ri.get("best_pair")
            if not bp:
                continue
            dm = bp.get("details", {}).get("depth_margin", {})
            if dm.get("state") == "strong":
                strong_depth = True
                break
        if strong_depth:
            allow = True
            reason = "pairwise_depth_strong"

    if dtype == "visual_behind":
        # Visual behind should remain conservative; require Qwen + adequate detector score.
        if float(chosen.get("score", 0.0)) >= 0.12:
            allow = True
            reason = "visual_behind_qwen_confirmed"

    # Singleton ordinal (not solved jointly) may still benefit from cached rerun,
    # but only if Qwen gives a valid candidate with reasonable detector support.
    if "ordinal" in route_reasons and float(chosen.get("score", 0.0)) >= 0.12:
        allow = True
        reason = "ordinal_qwen_confirmed"

    return {
        "override": bool(allow),
        "reason": reason,
        "routing": routing,
        "route_reasons": route_reasons,
        "selected_candidate": chosen,
        "shortlist": shortlist,
        "suppressed": suppressed,
        "decision": decision,
        "qwen_raw": raw,
    }


# ============================================================
# State
# ============================================================

predictions: Dict[str, dict] = {}
failures: Dict[str, dict] = {}
changes: Dict[str, dict] = {}

if ARGS.resume:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            ck = json.load(f)
        predictions.update(ck.get("predictions", {}))
        failures.update(ck.get("failures", {}))
        changes.update(ck.get("changes", {}))

    for sf in sorted(SCENE_RESULTS_DIR.glob("*.json")):
        try:
            with open(sf, "r", encoding="utf-8") as f:
                sd = json.load(f)
            predictions.update(sd.get("predictions", {}))
            failures.update(sd.get("failures", {}))
            changes.update(sd.get("changes", {}))
        except Exception:
            pass


def save_scene_result(scene_id):
    qids = ALL_SCENE_QIDS[scene_id]
    atomic_json_dump({
        "scene_id": scene_id,
        "saved_at": now_str(),
        "predictions": {q: predictions[q] for q in qids if q in predictions},
        "failures": {q: failures[q] for q in qids if q in failures},
        "changes": {q: changes[q] for q in qids if q in changes},
    }, SCENE_RESULTS_DIR / f"{scene_id}.json")


def build_pilot_submission():
    out = {}
    for qid in PILOT_QIDS:
        if qid not in predictions:
            continue
        item = dict(ORIGINAL_QUERIES[qid])
        item["bbox"] = predictions[qid]["bbox_normalized"]
        out[qid] = item
    return out


def save_state():
    atomic_json_dump({
        "version": "V2-P0-Pilot",
        "saved_at": now_str(),
        "routes": sorted(ENABLED_ROUTES),
        "predictions": predictions,
        "failures": failures,
        "changes": changes,
    }, CHECKPOINT_PATH)
    atomic_json_dump(build_pilot_submission(), PILOT_SUBMISSION_PATH)
    atomic_json_dump(failures, FAILURES_PATH)
    atomic_json_dump({
        "changed_count": len(changes),
        "changes": changes,
    }, CHANGE_REPORT_PATH)


def copy_v1_prediction(qid, source="v1_copy", v2_extra=None):
    v1 = V1_PREDICTIONS[qid]
    pred = {
        "query": v1["query"],
        "scene_id": v1["scene_id"],
        "bbox": v1["bbox"],
        "bbox_normalized": v1["bbox_normalized"],
        "source": source,
        "v1_bbox_normalized": v1["bbox_normalized"],
        "v2_override": False,
        "v2": v2_extra or {},
    }
    return pred


# ============================================================
# Main
# ============================================================

start_time = time.time()
processed_scenes = 0
v2_route_query_count = 0
v2_override_count = 0
v1_copy_count = 0
dino_cache_hits_before = 0

for scene_pos, scene_id in enumerate(PILOT_SCENES, 1):
    qids = ALL_SCENE_QIDS[scene_id]

    if all(q in predictions for q in qids):
        print(f"[{now_str()}] [{scene_pos}/{len(PILOT_SCENES)}] {scene_id}: complete, skip")
        continue

    first = ORIGINAL_QUERIES[qids[0]]
    visible = DATA_ROOT / first["visible"]
    infrared = DATA_ROOT / first["infrared"]
    depth = DATA_ROOT / first["depth"]

    print("\n" + "=" * 100)
    print(f"[{now_str()}] [{scene_pos}/{len(PILOT_SCENES)}] SCENE {scene_id} | {len(qids)} queries")

    try:
        ctx = SceneContext(scene_id, visible, infrared, depth)
    except Exception as e:
        print("SCENE LOAD ERROR:", e)
        for qid in qids:
            failures[qid] = {
                "stage": "scene_load",
                "error": repr(e),
                "traceback": traceback.format_exc()[-4000:],
            }
        save_scene_result(scene_id)
        continue

    # Precompute V2 route reasons.
    reasons_map = {qid: needs_v2_route(qid) for qid in qids}

    # Build ordinal families and solve multi-query families once.
    ordinal_results = {}
    if "ordinal" in ENABLED_ROUTES:
        families = build_ordinal_families(qids)
        for family_key, family_qids in families.items():
            if len(family_qids) < 2:
                continue
            try:
                result_map = solve_ordinal_family(ctx, family_qids)
                ordinal_results.update(result_map)
                print(
                    "ORDINAL FAMILY:",
                    family_key,
                    "queries=", family_qids,
                    "candidate_count=",
                    next(iter(result_map.values())).get("candidate_count") if result_map else 0,
                )
            except Exception as e:
                print("ORDINAL FAMILY ERROR", family_key, e)

    for qi, qid in enumerate(qids, 1):
        if qid in predictions:
            continue

        v1 = V1_PREDICTIONS[qid]
        route_reasons = reasons_map[qid]
        query = v1["query"]
        q_t0 = time.time()

        print("-" * 100)
        print(f"{scene_id} [{qi}/{len(qids)}] {qid}: {query}")
        print("V2 ROUTES:", route_reasons)

        try:
            if not route_reasons:
                predictions[qid] = copy_v1_prediction(qid)
                v1_copy_count += 1
                print("ACTION: COPY V1")
                continue

            v2_route_query_count += 1

            # Priority 1: scene-level ordinal family high-confidence override.
            ord_res = ordinal_results.get(qid)
            if ord_res and ord_res.get("confident"):
                chosen = ord_res["selected_candidate"]
                bbox = clamp_bbox(chosen["bbox"], ctx.W, ctx.H)
                bbox_norm = normalize_bbox(bbox, ctx.W, ctx.H)

                predictions[qid] = {
                    "query": query,
                    "scene_id": scene_id,
                    "bbox": bbox,
                    "bbox_normalized": bbox_norm,
                    "source": "v2_ordinal_joint",
                    "v1_bbox_normalized": v1["bbox_normalized"],
                    "v2_override": True,
                    "v2": {
                        "route_reasons": route_reasons,
                        "ordinal_joint": ord_res,
                    },
                    "runtime_seconds": round(time.time() - q_t0, 3),
                }

                if not same_bbox(bbox_norm, v1["bbox_normalized"]):
                    changes[qid] = {
                        "query_id": qid,
                        "scene_id": scene_id,
                        "query": query,
                        "route": "ordinal_joint",
                        "reason": ord_res["reason"],
                        "v1_bbox": v1["bbox_normalized"],
                        "v2_bbox": bbox_norm,
                    }
                v2_override_count += 1
                print("ACTION: V2 ORDINAL OVERRIDE", bbox_norm)
                continue

            # Priority 2: generic P0 rerun.
            result = run_generic_v2(ctx, qid, route_reasons)

            if result.get("override") and result.get("selected_candidate") is not None:
                chosen = result["selected_candidate"]
                bbox = clamp_bbox(chosen["bbox"], ctx.W, ctx.H)
                bbox_norm = normalize_bbox(bbox, ctx.W, ctx.H)

                predictions[qid] = {
                    "query": query,
                    "scene_id": scene_id,
                    "bbox": bbox,
                    "bbox_normalized": bbox_norm,
                    "source": "v2_p0_override",
                    "v1_bbox_normalized": v1["bbox_normalized"],
                    "v2_override": True,
                    "v2": result,
                    "runtime_seconds": round(time.time() - q_t0, 3),
                }

                if not same_bbox(bbox_norm, v1["bbox_normalized"]):
                    changes[qid] = {
                        "query_id": qid,
                        "scene_id": scene_id,
                        "query": query,
                        "route": "+".join(route_reasons),
                        "reason": result.get("reason"),
                        "v1_bbox": v1["bbox_normalized"],
                        "v2_bbox": bbox_norm,
                    }

                v2_override_count += 1
                print("ACTION: V2 OVERRIDE", result.get("reason"), bbox_norm)
            else:
                predictions[qid] = copy_v1_prediction(
                    qid,
                    source="v1_safe_fallback",
                    v2_extra=result,
                )
                v1_copy_count += 1
                print("ACTION: KEEP V1", result.get("reason"))

        except Exception as e:
            print("QUERY ERROR:", e)
            traceback.print_exc()
            failures[qid] = {
                "stage": "query",
                "error": repr(e),
                "query": query,
                "traceback": traceback.format_exc()[-6000:],
            }
            # Safety: still produce V1 result.
            predictions[qid] = copy_v1_prediction(
                qid,
                source="v1_exception_fallback",
                v2_extra={"error": repr(e)},
            )
            v1_copy_count += 1

        finally:
            cleanup_query_runtime(scene_id, qid)

    save_scene_result(scene_id)
    save_state()
    processed_scenes += 1

    print(
        f"[{now_str()}] SCENE {scene_id} COMPLETE | "
        f"DINO cache entries={len(ctx.dino_cache)} | tiled={len(ctx.tiled_cache)} | "
        f"total predictions={len(predictions)}/{len(PILOT_QIDS)} | "
        f"changes={len(changes)} | failures={len(failures)}"
    )

    ctx.cleanup()
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# Final validation / summary
# ============================================================

save_state()
pilot_submission = build_pilot_submission()

invalid = []
for qid, item in pilot_submission.items():
    b = item.get("bbox")
    if not (
        isinstance(b, list)
        and len(b) == 4
        and all(isinstance(v, (int, float)) for v in b)
        and 0 <= b[0] < b[2] <= 1
        and 0 <= b[1] < b[3] <= 1
    ):
        invalid.append((qid, b))

source_counts = defaultdict(int)
for p in predictions.values():
    source_counts[p.get("source", "unknown")] += 1

summary = {
    "version": "V2-P0-Pilot",
    "finished_at": now_str(),
    "elapsed_seconds": round(time.time() - start_time, 2),
    "pilot_scene_count": len(PILOT_SCENES),
    "pilot_query_count": len(PILOT_QIDS),
    "prediction_count": len(predictions),
    "submission_count": len(pilot_submission),
    "enabled_routes": sorted(ENABLED_ROUTES),
    "v2_route_query_count_this_run": v2_route_query_count,
    "v2_override_count_this_run": v2_override_count,
    "v1_copy_count_this_run": v1_copy_count,
    "bbox_changed_count": len(changes),
    "failure_count": len(failures),
    "invalid_bbox_count": len(invalid),
    "source_counts": dict(source_counts),
    "peak_cuda_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
    "pilot_submission": str(PILOT_SUBMISSION_PATH),
    "change_report": str(CHANGE_REPORT_PATH),
}
atomic_json_dump(summary, SUMMARY_PATH)

print("\n" + "=" * 100)
print("V2-P0 PILOT FINISHED")
print(json.dumps(summary, ensure_ascii=False, indent=2))

if len(pilot_submission) != len(PILOT_QIDS):
    print("ERROR: submission count does not match Pilot query count")
elif invalid:
    print("ERROR: invalid bbox exists:", invalid[:10])
else:
    print("PILOT OUTPUT VALID")
    print("Next: merge submission_v2_pilot.json into the full V1 submission.")
