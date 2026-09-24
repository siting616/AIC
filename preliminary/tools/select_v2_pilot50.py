#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select a fixed, high-information 50-scene V2 pilot set from V1 checkpoint.

Usage:
    python tools/select_v2_pilot50.py \
      --checkpoint outputs/v1_full/checkpoint_predictions.json \
      --output configs/V2_PILOT_50.txt \
      --report outputs/v2_pilot50/pilot50_selection_report.json

The selection is deterministic. It prioritizes:
- ordinal families
- depth queries
- V1 fallback scenes
- part/small-target queries
- OCR/text queries
- thermal queries
- ordinary RGB controls
- mixed high-risk scenes (suppression / low-rank Qwen choices)
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PART_PATTERNS = [
    r"\bear\b", r"\bears\b", r"\bwheel\b", r"\bwheels\b",
    r"\btail\b", r"\btails\b", r"\bhorn\b", r"\bhorns\b",
    r"\bbeak\b", r"\bwing\b", r"\bwings\b",
    r"\bhandle\b", r"\bhandles\b", r"\bscrew\b", r"\bscrews\b",
    r"\bbutton\b", r"\bbuttons\b", r"\bdoor handle\b",
    r"\blicense plate\b", r"\bhead\b", r"\bhand\b", r"\bhands\b",
    r"\bfoot\b", r"\bfeet\b", r"\bleg\b", r"\blegs\b",
    r"\barm\b", r"\barms\b", r"\beye\b", r"\beyes\b",
]

OCR_PATTERNS = [
    r"\btext\b", r"\bword\b", r"\bwords\b", r"\bletter\b", r"\bletters\b",
    r"\bnumber\b", r"\bnumbers\b", r"\blabel\b", r"\blabeled\b",
    r"\bmarked\b", r"\bwritten\b", r"\bcharacters\b", r"\bcharacter\b",
    r"\blogo\b", r"\bsign reading\b", r"\bmodel number\b",
    r"\blicense plate\b",
]

THERMAL_PATTERNS = [
    r"\bwarm\b", r"\bwarmer\b", r"\bwarmest\b", r"\bhot\b",
    r"\bthermal\b", r"\binfrared\b", r"\bheat\b",
    r"\bcold\b", r"\bcooler\b",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_any(patterns, text):
    return any(re.search(p, text) for p in patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    data = load_json(args.checkpoint)
    preds = data["predictions"]

    stats = defaultdict(Counter)

    for qid, x in preds.items():
        sid = x["scene_id"]
        q = x["query"].lower()
        st = stats[sid]
        st["queries"] += 1

        ordinal = str(x.get("parsed", {}).get("ordinal", "none")).lower()
        routing = x.get("routing", {})

        if ordinal != "none":
            st["ordinal"] += 1
        if routing.get("depth"):
            st["depth"] += 1
        if x.get("fallback_used"):
            st["fallback"] += 1
        if match_any(PART_PATTERNS, q):
            st["part"] += 1
        if match_any(OCR_PATTERNS, q) or re.search(r"""['"][^'"]+['"]""", x["query"]):
            st["ocr"] += 1
        if match_any(THERMAL_PATTERNS, q):
            st["thermal"] += 1
        if x.get("suppressed"):
            st["suppressed"] += 1

        selected_id = x.get("decision", {}).get("selected_candidate_id")
        rank = None
        for i, cand in enumerate(x.get("shortlist", []), 1):
            if cand.get("candidate_id") == selected_id:
                rank = i
                break
        if rank and rank > 3:
            st["lowrank"] += 1

        special = (
            ordinal != "none"
            or routing.get("depth")
            or x.get("fallback_used")
            or match_any(PART_PATTERNS, q)
            or match_any(OCR_PATTERNS, q)
            or match_any(THERMAL_PATTERNS, q)
        )
        if not special:
            st["ordinary"] += 1

    selected = []
    reasons = defaultdict(list)

    def add(sid, reason):
        if sid not in selected:
            selected.append(sid)
        reasons[sid].append(reason)

    # Known V1 ordinal diagnostic scene; keep it fixed for regression tests.
    if "000023" in stats:
        add("000023", "known_ordinal_failure")

    # Distinct-scene quotas. Categories can still overlap naturally.
    quotas = [
        ("ordinal", 12),
        ("depth", 8),
        ("fallback", 6),
        ("part", 6),
        ("ocr", 5),
        ("thermal", 3),
        ("ordinary", 5),
    ]

    for cat, quota in quotas:
        already = sum(1 for sid in selected if stats[sid][cat] > 0)
        need = max(0, quota - already)

        candidates = sorted(
            [
                sid for sid in stats
                if sid not in selected and stats[sid][cat] > 0
            ],
            key=lambda sid: (
                stats[sid][cat],
                stats[sid]["suppressed"] + stats[sid]["lowrank"],
                stats[sid]["queries"],
                sid,
            ),
            reverse=True,
        )

        for sid in candidates[:need]:
            add(sid, cat)

    def risk_score(sid):
        st = stats[sid]
        return (
            2.5 * st["ordinal"]
            + 2.2 * st["depth"]
            + 4.0 * st["fallback"]
            + 2.0 * st["part"]
            + 2.0 * st["ocr"]
            + 2.5 * st["thermal"]
            + 1.5 * st["suppressed"]
            + 1.2 * st["lowrank"]
            + 0.15 * st["queries"]
        )

    remaining = 50 - len(selected)
    if remaining > 0:
        mixed = sorted(
            [sid for sid in stats if sid not in selected],
            key=lambda sid: (risk_score(sid), sid),
            reverse=True,
        )
        for sid in mixed[:remaining]:
            add(sid, "mixed_high_risk")

    selected = selected[:50]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(selected) + "\n", encoding="utf-8")

    coverage = Counter()
    total_queries = 0
    scene_records = []

    for sid in selected:
        total_queries += stats[sid]["queries"]
        for cat in (
            "ordinal", "depth", "fallback", "part", "ocr",
            "thermal", "ordinary", "suppressed", "lowrank"
        ):
            coverage[cat] += stats[sid][cat]

        scene_records.append({
            "scene_id": sid,
            "reasons": reasons[sid],
            "stats": dict(stats[sid]),
            "risk_score": round(risk_score(sid), 3),
        })

    report = {
        "scene_count": len(selected),
        "query_count_in_pilot": total_queries,
        "coverage_query_counts": dict(coverage),
        "scenes": scene_records,
    }

    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("V2 PILOT-50 SELECTED")
    print("Scenes:", len(selected))
    print("Queries in pilot scenes:", total_queries)
    print("Coverage:", dict(coverage))
    print("Scene list:", out)
    print("Report:", rp)


if __name__ == "__main__":
    main()
