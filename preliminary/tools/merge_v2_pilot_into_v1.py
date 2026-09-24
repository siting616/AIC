#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import sys
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def scene_id(qid):
    return qid.split("_", 1)[0]


def extract_predictions(obj):
    # checkpoint-style
    if isinstance(obj, dict) and isinstance(obj.get("predictions"), dict):
        return obj["predictions"]

    # submission-style
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if (
                isinstance(k, str)
                and "_" in k
                and isinstance(v, dict)
                and ("bbox" in v or "bbox_normalized" in v)
            ):
                out[k] = v
        if out:
            return out

    # list-style
    if isinstance(obj, list):
        out = {}
        for x in obj:
            if not isinstance(x, dict):
                continue
            qid = x.get("query_id") or x.get("qid")
            if qid and ("bbox" in x or "bbox_normalized" in x):
                out[str(qid)] = x
        if out:
            return out

    raise ValueError("Cannot find predictions in pilot JSON.")


def get_norm_bbox(entry):
    b = entry.get("bbox_normalized", entry.get("bbox"))
    if not isinstance(b, list) or len(b) != 4:
        raise ValueError(f"invalid bbox: {b}")

    b = [float(x) for x in b]

    if not all(0 <= x <= 1 for x in b):
        raise ValueError(
            f"bbox is not normalized: {b}. "
            "Pilot file must contain bbox_normalized or normalized bbox."
        )

    x1, y1, x2, y2 = b
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"invalid bbox ordering: {b}")

    return [round(x, 6) for x in b]


def read_scene_list(path):
    scenes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                scenes.add(s)
    return scenes


def validate_submission(sub):
    bad = []
    for qid, item in sub.items():
        b = item.get("bbox")
        if not isinstance(b, list) or len(b) != 4:
            bad.append((qid, b))
            continue
        try:
            b = [float(x) for x in b]
        except Exception:
            bad.append((qid, b))
            continue
        if not (all(0 <= x <= 1 for x in b) and b[0] < b[2] and b[1] < b[3]):
            bad.append((qid, b))
    return bad


def main():
    ap = argparse.ArgumentParser(
        description="Replace V1 bboxes with V2 pilot bboxes for selected scenes."
    )
    ap.add_argument("--baseline", required=True, help="Full V1 submission.json")
    ap.add_argument("--pilot", required=True, help="V2 pilot JSON")
    ap.add_argument("--output", required=True, help="Hybrid output JSON")
    ap.add_argument("--report", default=None, help="Merge report JSON")
    ap.add_argument("--scene-list", default=None,
                    help="Optional txt, one pilot scene id per line")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Allow missing queries inside selected pilot scenes")
    args = ap.parse_args()

    baseline = load_json(args.baseline)
    pilot_raw = load_json(args.pilot)
    pilot = extract_predictions(pilot_raw)

    if not isinstance(baseline, dict):
        raise SystemExit("Baseline must be a dict keyed by query_id.")

    if args.scene_list:
        selected_scenes = read_scene_list(args.scene_list)
    else:
        selected_scenes = {scene_id(qid) for qid in pilot}

    baseline_by_scene = {}
    for qid in baseline:
        baseline_by_scene.setdefault(scene_id(qid), []).append(qid)

    unknown_scenes = sorted(selected_scenes - set(baseline_by_scene))
    if unknown_scenes:
        raise SystemExit(f"Pilot scenes not found in baseline: {unknown_scenes[:20]}")

    expected_qids = set()
    for sid in selected_scenes:
        expected_qids.update(baseline_by_scene[sid])

    pilot_qids = set(pilot)
    missing = sorted(expected_qids - pilot_qids)

    if missing and not args.allow_partial:
        print("ERROR: pilot is missing queries from selected scenes.")
        print("Selected scenes:", len(selected_scenes))
        print("Expected queries:", len(expected_qids))
        print("Available:", len(expected_qids & pilot_qids))
        print("First missing:", missing[:20])
        print("Use --allow-partial only if this is intentional.")
        sys.exit(2)

    hybrid = copy.deepcopy(baseline)
    changed = []
    same = []
    skipped = []

    for qid in sorted(expected_qids):
        if qid not in pilot:
            skipped.append(qid)
            continue

        new_bbox = get_norm_bbox(pilot[qid])
        old_bbox = [float(x) for x in baseline[qid]["bbox"]]
        hybrid[qid]["bbox"] = new_bbox

        if all(abs(a - b) < 1e-9 for a, b in zip(old_bbox, new_bbox)):
            same.append(qid)
        else:
            changed.append({
                "query_id": qid,
                "scene_id": scene_id(qid),
                "query": baseline[qid].get("query"),
                "v1_bbox": old_bbox,
                "v2_bbox": new_bbox,
            })

    # Critical validation
    if len(hybrid) != len(baseline):
        raise SystemExit("Final query count changed.")
    if set(hybrid) != set(baseline):
        raise SystemExit("Final query IDs changed.")

    protected = ("visible", "infrared", "depth", "query")
    for qid in baseline:
        for key in protected:
            if baseline[qid].get(key) != hybrid[qid].get(key):
                raise SystemExit(f"Protected field changed: {qid} / {key}")

    bad = validate_submission(hybrid)
    if bad:
        raise SystemExit(f"Invalid bbox found, first examples: {bad[:10]}")

    save_json(args.output, hybrid)

    report = args.report
    if report is None:
        op = Path(args.output)
        report = str(op.with_name(op.stem + "_merge_report.json"))

    save_json(report, {
        "baseline": args.baseline,
        "pilot": args.pilot,
        "output": args.output,
        "baseline_query_count": len(baseline),
        "final_query_count": len(hybrid),
        "pilot_scene_count": len(selected_scenes),
        "pilot_scenes": sorted(selected_scenes),
        "queries_in_selected_scenes": len(expected_qids),
        "pilot_predictions_available": len(expected_qids & pilot_qids),
        "bbox_changed_count": len(changed),
        "bbox_same_count": len(same),
        "missing_pilot_query_count": len(skipped),
        "changed": changed,
        "same_query_ids": same,
        "missing_query_ids": skipped,
    })

    print("=" * 72)
    print("MERGE COMPLETE")
    print("Baseline queries          :", len(baseline))
    print("Pilot scenes              :", len(selected_scenes))
    print("Queries in pilot scenes   :", len(expected_qids))
    print("Pilot predictions         :", len(expected_qids & pilot_qids))
    print("Actually changed bbox     :", len(changed))
    print("Same bbox as V1           :", len(same))
    print("Missing pilot predictions :", len(skipped))
    print("Final submission queries  :", len(hybrid))
    print("Invalid bbox              :", len(bad))
    print("Output                    :", args.output)
    print("Report                    :", report)
    print("READY TO SUBMIT")


if __name__ == "__main__":
    main()
