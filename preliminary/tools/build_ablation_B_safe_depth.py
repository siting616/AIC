#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build / inspect Ablation-B: Ordinal-Joint baseline + SAFE camera-relative depth changes.

Design:
1) Base submission should be the scored Ablation-A (0.612).
2) Candidate depth changes come from the previous V2 change_report.json.
3) This tool FIRST identifies a conservative set of "pure camera-depth" candidates.
4) It can draw V1-vs-V2 overlays for manual review.
5) Final submission is built only from query IDs explicitly listed in a whitelist.

This prevents accidental inclusion of:
- "closest to the top-right corner" (2D geometry, not camera depth)
- part scope such as "rear wheel of the nearest bicycle"
- visual behind / in-front-of relations
- generic ordinal changes
- fallback changes
"""

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image, ImageDraw


PART_WORDS = {
    "wheel", "wheels", "mirror", "mirrors", "rearview mirror",
    "leg", "legs", "ear", "ears", "tail", "tails", "horn", "horns",
    "handle", "handles", "screw", "screws", "plate", "license plate",
    "head", "heads", "arm", "arms", "hand", "hands", "foot", "feet",
    "beak", "beaks", "wing", "wings", "door handle",
}

SPATIAL_ANCHOR_PATTERNS = [
    r"\btop[\s-]?right\b",
    r"\btop[\s-]?left\b",
    r"\bbottom[\s-]?right\b",
    r"\bbottom[\s-]?left\b",
    r"\bleftmost\b",
    r"\brightmost\b",
    r"\bfar left\b",
    r"\bfar right\b",
    r"\bcorner\b",
    r"\bedge\b",
    r"\bfrom left\b",
    r"\bfrom right\b",
]

REFERENCE_RELATION_PATTERNS = [
    r"\bbehind\b",
    r"\bin front of\b",
    r"\bbeside\b",
    r"\bnext to\b",
    r"\bleft of\b",
    r"\bright of\b",
    r"\babove\b",
    r"\bbelow\b",
    r"\bunder\b",
    r"\bover\b",
]

CAMERA_DEPTH_PATTERNS = [
    r"\bclosest to (?:the )?camera\b",
    r"\bnearest to (?:the )?camera\b",
    r"\bfarthest from (?:the )?camera\b",
    r"\bfurthest from (?:the )?camera\b",
    r"\bfarthest to (?:the )?camera\b",
    r"\bnearest to (?:the )?viewer\b",
    r"\bclosest to (?:the )?viewer\b",
    r"\bfarthest from (?:the )?viewer\b",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def valid_bbox(b):
    return (
        isinstance(b, list)
        and len(b) == 4
        and all(isinstance(x, (int, float)) for x in b)
        and 0 <= b[0] < b[2] <= 1
        and 0 <= b[1] < b[3] <= 1
    )


def has_pattern(patterns, text):
    return any(re.search(p, text, flags=re.I) for p in patterns)


def contains_part_scope(text):
    t = text.lower()
    return any(re.search(rf"\b{re.escape(w)}\b", t) for w in PART_WORDS)


def is_strict_safe_depth_candidate(change):
    q = str(change.get("query", "")).lower()
    route = str(change.get("route", "")).lower().strip()
    reason = str(change.get("reason", "")).lower().strip()

    checks = {
        "route_exact_depth": route == "depth",
        "reason_absolute_depth": reason == "absolute_depth_strong",
        "explicit_camera_relation": has_pattern(CAMERA_DEPTH_PATTERNS, q),
        "no_part_scope": not contains_part_scope(q),
        "no_spatial_anchor": not has_pattern(SPATIAL_ANCHOR_PATTERNS, q),
        "no_reference_relation": not has_pattern(REFERENCE_RELATION_PATTERNS, q),
    }
    return all(checks.values()), checks


def read_whitelist(path):
    ids = []
    if not path:
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def norm_to_px(b, W, H):
    return [b[0] * W, b[1] * H, b[2] * W, b[3] * H]


def draw_overlay(img_path, v1_bbox, v2_bbox, out_path):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)

    a = norm_to_px(v1_bbox, W, H)
    b = norm_to_px(v2_bbox, W, H)

    # Default colors only for diagnostic clarity.
    d.rectangle(a, outline="red", width=5)
    d.rectangle(b, outline="lime", width=5)

    d.rectangle([8, 8, 320, 72], fill="white")
    d.text((16, 16), "RED = Ablation-A / V1 bbox", fill="red")
    d.text((16, 42), "GREEN = V2 depth bbox", fill="green")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="Scored Ablation-A submission (0.612)")
    ap.add_argument("--change-report", required=True)
    ap.add_argument("--candidate-report", required=True)
    ap.add_argument("--whitelist-template", required=True)
    ap.add_argument("--whitelist", default=None,
                    help="Approved query IDs, one per line")
    ap.add_argument("--output", default=None,
                    help="Final Ablation-B submission")
    ap.add_argument("--data-root", default=None,
                    help="Dataset root for diagnostic overlays")
    ap.add_argument("--overlay-dir", default=None)
    args = ap.parse_args()

    base = load_json(args.base)
    report = load_json(args.change_report)
    changes = report.get("changes", {})

    candidates = []
    rejected = []

    for qid, ch in changes.items():
        ok, checks = is_strict_safe_depth_candidate(ch)
        rec = {
            "query_id": qid,
            "query": ch.get("query"),
            "route": ch.get("route"),
            "reason": ch.get("reason"),
            "v1_bbox": ch.get("v1_bbox"),
            "v2_bbox": ch.get("v2_bbox"),
            "checks": checks,
        }
        if ok:
            candidates.append(rec)
        elif "depth" in str(ch.get("route", "")).lower():
            rejected.append(rec)

    save_json(args.candidate_report, {
        "strict_candidate_count": len(candidates),
        "strict_candidates": candidates,
        "rejected_depth_change_count": len(rejected),
        "rejected_depth_changes": rejected,
    })

    # Whitelist template contains candidates uncommented so user can delete
    # any visually suspicious line before final build.
    wt = Path(args.whitelist_template)
    wt.parent.mkdir(parents=True, exist_ok=True)
    with open(wt, "w", encoding="utf-8") as f:
        f.write("# Ablation-B SAFE DEPTH whitelist\n")
        f.write("# Review overlays first. Delete/comment any suspicious query.\n")
        for x in candidates:
            f.write(f'{x["query_id"]}\n')

    print("=" * 80)
    print("SAFE DEPTH PREVIEW")
    print("Strict candidates:", len(candidates))
    print()
    for x in candidates:
        print(x["query_id"], "|", x["query"])
        print("  V1:", x["v1_bbox"])
        print("  V2:", x["v2_bbox"])
    print()
    print("Candidate report :", args.candidate_report)
    print("Whitelist        :", args.whitelist_template)

    # Optional visual diagnostic.
    if args.data_root and args.overlay_dir:
        data_root = Path(args.data_root)
        overlay_dir = Path(args.overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        for x in candidates:
            qid = x["query_id"]
            if qid not in base:
                continue
            visible = base[qid].get("visible")
            if not visible:
                continue
            img_path = data_root / visible
            if img_path.exists():
                draw_overlay(
                    img_path,
                    x["v1_bbox"],
                    x["v2_bbox"],
                    overlay_dir / f"{qid}_v1_red_v2_green.jpg",
                )
        print("Overlays          :", overlay_dir)

    # Preview-only if no explicit whitelist/output.
    if not args.whitelist or not args.output:
        print("\nPREVIEW COMPLETE")
        print("Review the candidate list/overlays, edit the whitelist, then rerun with")
        print("--whitelist <file> --output <submission.json>")
        return

    approved = read_whitelist(args.whitelist)
    candidate_map = {x["query_id"]: x for x in candidates}

    unknown = [qid for qid in approved if qid not in candidate_map]
    if unknown:
        raise SystemExit(
            "Whitelist contains IDs that are not strict safe-depth candidates: "
            + ", ".join(unknown)
        )

    out = copy.deepcopy(base)
    applied = []

    for qid in approved:
        if qid not in out:
            raise SystemExit(f"Query not found in base submission: {qid}")
        ch = candidate_map[qid]
        b = ch["v2_bbox"]
        if not valid_bbox(b):
            raise SystemExit(f"Invalid V2 bbox for {qid}: {b}")
        old = list(out[qid]["bbox"])
        out[qid]["bbox"] = [float(v) for v in b]
        applied.append({
            "query_id": qid,
            "query": ch["query"],
            "base_bbox": old,
            "depth_bbox": b,
        })

    bad = [(qid, x.get("bbox")) for qid, x in out.items()
           if not valid_bbox(x.get("bbox"))]
    if bad:
        raise SystemExit(f"Invalid final bbox: {bad[:10]}")

    save_json(args.output, out)
    summary_path = str(Path(args.output).with_name(Path(args.output).stem + "_summary.json"))
    save_json(summary_path, {
        "base": args.base,
        "change_report": args.change_report,
        "final_query_count": len(out),
        "strict_candidate_count": len(candidates),
        "approved_depth_change_count": len(applied),
        "approved_changes": applied,
        "invalid_bbox_count": len(bad),
    })

    print("\n" + "=" * 80)
    print("ABLATION-B READY")
    print("Base queries             :", len(base))
    print("Strict depth candidates  :", len(candidates))
    print("Approved depth changes   :", len(applied))
    print("Final queries            :", len(out))
    print("Invalid bbox             :", len(bad))
    print("Output                   :", args.output)
    print("Summary                  :", summary_path)


if __name__ == "__main__":
    main()
