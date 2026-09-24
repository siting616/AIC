#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reconstruct an ablation submission directly from:
1) the scored V2 hybrid submission, and
2) its change_report.json.

Default Ablation-A:
- KEEP only changes where:
    route == "ordinal_joint"
    reason == "ordinal_joint_qwen_validated"
- REVERT every other V2 change to its recorded v1_bbox.

This avoids needing to locate the original V1 baseline file.
"""

import argparse
import copy
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def valid_bbox(b):
    return (
        isinstance(b, list)
        and len(b) == 4
        and all(isinstance(x, (int, float)) for x in b)
        and 0 <= b[0] < b[2] <= 1
        and 0 <= b[1] < b[3] <= 1
    )


def same_bbox(a, b, eps=1e-6):
    return len(a) == len(b) == 4 and all(abs(float(x)-float(y)) <= eps for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", required=True, help="The exact V2 hybrid JSON that was scored")
    ap.add_argument("--change-report", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--keep-route", default="ordinal_joint")
    ap.add_argument("--keep-reason", default="ordinal_joint_qwen_validated")
    args = ap.parse_args()

    hybrid = load_json(args.hybrid)
    report = load_json(args.change_report)

    if not isinstance(hybrid, dict):
        raise SystemExit("Hybrid submission must be a dict keyed by query_id.")

    changes = report.get("changes", {})
    if not isinstance(changes, dict):
        raise SystemExit("change_report.json has no valid 'changes' dict.")

    out = copy.deepcopy(hybrid)

    kept = []
    reverted = []
    mismatched = []

    for qid, ch in changes.items():
        if qid not in out:
            raise SystemExit(f"Changed query missing in hybrid submission: {qid}")

        v1_bbox = ch.get("v1_bbox")
        v2_bbox = ch.get("v2_bbox")
        if not valid_bbox(v1_bbox):
            raise SystemExit(f"Invalid v1_bbox for {qid}: {v1_bbox}")
        if not valid_bbox(v2_bbox):
            raise SystemExit(f"Invalid v2_bbox for {qid}: {v2_bbox}")

        # The scored hybrid should contain the V2 bbox for every recorded change.
        if not same_bbox(out[qid].get("bbox"), v2_bbox):
            mismatched.append({
                "query_id": qid,
                "hybrid_bbox": out[qid].get("bbox"),
                "report_v2_bbox": v2_bbox,
            })

        keep = (
            ch.get("route") == args.keep_route
            and ch.get("reason") == args.keep_reason
        )

        if keep:
            out[qid]["bbox"] = [float(x) for x in v2_bbox]
            kept.append(qid)
        else:
            out[qid]["bbox"] = [float(x) for x in v1_bbox]
            reverted.append(qid)

    if mismatched:
        print("WARNING: Some report V2 bboxes do not exactly match the hybrid file.")
        print("First mismatches:", mismatched[:10])
        print("Aborting to avoid building an invalid ablation.")
        raise SystemExit(2)

    bad = [(qid, item.get("bbox")) for qid, item in out.items() if not valid_bbox(item.get("bbox"))]
    if bad:
        raise SystemExit(f"Invalid bbox in output: {bad[:10]}")

    save_json(args.output, out)

    summary_path = args.summary or str(
        Path(args.output).with_name(Path(args.output).stem + "_summary.json")
    )
    save_json(summary_path, {
        "source_hybrid": args.hybrid,
        "change_report": args.change_report,
        "output": args.output,
        "keep_route": args.keep_route,
        "keep_reason": args.keep_reason,
        "query_count": len(out),
        "all_v2_change_count": len(changes),
        "kept_change_count": len(kept),
        "reverted_change_count": len(reverted),
        "mismatch_count": len(mismatched),
        "invalid_bbox_count": len(bad),
        "kept_query_ids": kept,
        "reverted_query_ids": reverted,
    })

    print("=" * 80)
    print("ABLATION-A READY")
    print("Total queries       :", len(out))
    print("All V2 changes      :", len(changes))
    print("Kept ordinal_joint  :", len(kept))
    print("Reverted changes    :", len(reverted))
    print("Mismatch count      :", len(mismatched))
    print("Invalid bbox        :", len(bad))
    print("Output              :", args.output)
    print("Summary             :", summary_path)
    print()
    print("Kept query IDs:")
    for qid in kept:
        print(" ", qid)


if __name__ == "__main__":
    main()
