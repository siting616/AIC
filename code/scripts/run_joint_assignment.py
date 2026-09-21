"""Offline diagnostic for joint ordinal/position assignment.

This script never creates a submission archive.  It groups mutually-exclusive
queries on the same image, pools their existing candidates, applies a small
one-to-one assignment, and reports both competition impact and RefCOCO ACC.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_iou


POSITION_PATTERNS = [
    (r"\b(?:first|1st|leftmost|far left)\b", "x", 0, "leftmost"),
    (r"\b(?:second|2nd)(?:\s+\w+){0,3}\s+from the left\b", "x", 1, "second-left"),
    (r"\b(?:third|3rd)(?:\s+\w+){0,3}\s+from the left\b", "x", 2, "third-left"),
    (r"\b(?:last|rightmost|far right)\b", "x", -1, "rightmost"),
    (r"\b(?:second|2nd)(?:\s+\w+){0,3}\s+from the right\b", "x", -2, "second-right"),
    (r"\b(?:third|3rd)(?:\s+\w+){0,3}\s+from the right\b", "x", -3, "third-right"),
    (r"\b(?:topmost|at the top|on top)\b", "y", 0, "top"),
    (r"\b(?:bottommost|at the bottom)\b", "y", -1, "bottom"),
]

# Standalone left/right are intentionally narrower than a raw word match, which
# would otherwise turn relational phrases such as "left of the car" into groups.
EDGE_PATTERNS = [
    (r"\b(?:on|at|to) the left(?: side)?\b|\bleft side\b", "x", 0, "left"),
    (r"\b(?:on|at|to) the right(?: side)?\b|\bright side\b", "x", -1, "right"),
    (r"\b(?:in|at) the (?:middle|center)\b|\b(?:middle|center) one\b", "x", "middle", "middle"),
    (r"\b(?:upper|top) (?:one|most)\b", "y", 0, "top"),
    (r"\b(?:lower|bottom) (?:one|most)\b", "y", -1, "bottom"),
]

CLASS_WORDS = {
    "person", "man", "woman", "boy", "girl", "child", "kid", "player", "skier",
    "surfer", "rider", "worker", "lady", "guy", "dog", "cat", "horse", "cow",
    "sheep", "bird", "elephant", "bear", "zebra", "giraffe", "car", "truck", "bus",
    "train", "bike", "bicycle", "motorcycle", "boat", "airplane", "chair", "bench",
    "table", "bottle", "cup", "bowl", "plate", "dish", "umbrella", "sign", "bag",
    "backpack", "suitcase", "ball", "kite", "board", "skateboard", "surfboard",
    "phone", "laptop", "book", "clock", "vase", "plant", "tree", "building", "door",
    "window", "light", "lamp", "screen", "monitor", "box", "cone", "pole", "helmet",
}

CLASS_ALIASES = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "kid": "person", "player": "person", "skier": "person",
    "surfer": "person", "rider": "person", "worker": "person", "lady": "person",
    "guy": "person", "bike": "bicycle", "dish": "bowl",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def bbox_from(value):
    if isinstance(value, dict):
        value = value.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in box) or box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box


def query_class(query: str, candidates: list[dict]) -> str | None:
    words = re.findall(r"[a-z]+", query.lower())
    for word in words:
        if word in CLASS_WORDS:
            return CLASS_ALIASES.get(word, word)
    for item in candidates[:3]:
        for word in re.findall(r"[a-z]+", str(item.get("label", "")).lower()):
            if word in CLASS_WORDS:
                return CLASS_ALIASES.get(word, word)
    return None


def position_intent(query: str):
    text = query.lower()
    for pattern, axis, rank, label in POSITION_PATTERNS + EDGE_PATTERNS:
        if re.search(pattern, text):
            return {"axis": axis, "rank": rank, "label": label}
    return None


def candidate_list(detail: dict, baseline_box: list[float]) -> list[dict]:
    raw = detail.get("reranked_candidates", []) if isinstance(detail, dict) else []
    result = []
    for rank, item in enumerate(raw[:10]):
        box = bbox_from(item)
        if box is None:
            continue
        result.append({
            "bbox": box,
            "score": float(item.get("rerank_score", item.get("score", 0.0))),
            "model_score": float(item.get("score", 0.0)),
            "label": item.get("label", ""),
            "rank": rank + 1,
        })
    if baseline_box and not any(compute_iou(baseline_box, x["bbox"]) >= 0.999 for x in result):
        result.insert(0, {"bbox": baseline_box, "score": 0.5, "model_score": 0.5,
                          "label": "baseline", "rank": 0})
    return result


def dedupe_pool(records: list[dict]) -> list[dict]:
    pool = []
    for record in records:
        for candidate in record["candidates"]:
            match = next((x for x in pool if compute_iou(x["bbox"], candidate["bbox"]) >= 0.85), None)
            if match is None:
                pool.append({"bbox": candidate["bbox"], "sources": {record["query_id"]},
                             "scores": {record["query_id"]: candidate["score"]}})
            else:
                match["sources"].add(record["query_id"])
                match["scores"][record["query_id"]] = max(
                    match["scores"].get(record["query_id"], 0.0), candidate["score"]
                )
    return pool


def desired_index(intent: dict, count: int) -> float:
    rank = intent["rank"]
    if rank == "middle":
        return (count - 1) / 2
    if rank < 0:
        return max(0, count + rank)
    return min(count - 1, rank)


def assignment_score(record: dict, candidate: dict, ordered_index: int, pool_size: int) -> float:
    own = candidate["scores"].get(record["query_id"])
    if own is None:
        # Cross-query candidates are useful, but must not overwhelm an existing
        # query-specific candidate on weak spatial evidence alone.
        base = max(candidate["scores"].values(), default=0.0) * 0.80
    else:
        base = own
    wanted = desired_index(record["intent"], pool_size)
    distance = abs(ordered_index - wanted)
    positional = max(0.0, 1.0 - distance / max(1, pool_size - 1))
    consensus = min(1.0, len(candidate["sources"]) / max(1, len(record["group_records"])))
    return 0.60 * base + 0.35 * positional + 0.05 * consensus


def best_assignment(records: list[dict], pool: list[dict]):
    if len(pool) < len(records) or len(records) > 7:
        return None
    axis = records[0]["intent"]["axis"]
    ordered = sorted(range(len(pool)), key=lambda i: (pool[i]["bbox"][0] + pool[i]["bbox"][2]) / 2
                     if axis == "x" else (pool[i]["bbox"][1] + pool[i]["bbox"][3]) / 2)
    order_index = {candidate_index: rank for rank, candidate_index in enumerate(ordered)}
    # Exact rectangular assignment using DP over the small query dimension.
    # Complexity is O(candidates * queries * 2**queries), rather than enumerating
    # every candidate permutation.
    query_count = len(records)
    states = {0: [(0.0, ())]}
    for candidate_index, candidate in enumerate(pool):
        updated = {mask: list(items) for mask, items in states.items()}
        for mask, items in states.items():
            for query_index, record in enumerate(records):
                bit = 1 << query_index
                if mask & bit:
                    continue
                new_mask = mask | bit
                value = assignment_score(
                    record, candidate, order_index[candidate_index], len(pool)
                )
                bucket = updated.setdefault(new_mask, [])
                for score, chosen in items:
                    selection = list(chosen) if chosen else [-1] * query_count
                    selection[query_index] = candidate_index
                    bucket.append((score + value, tuple(selection)))
                # Retain two distinct best paths so the safety margin remains
                # available without materializing all assignments.
                unique = {}
                for score, chosen in sorted(bucket, reverse=True, key=lambda x: x[0]):
                    unique.setdefault(chosen, score)
                    if len(unique) == 2:
                        break
                updated[new_mask] = [(score, chosen) for chosen, score in unique.items()]
        states = updated
    scored = states.get((1 << query_count) - 1, [])
    scored.sort(reverse=True, key=lambda x: x[0])
    if not scored or any(index < 0 for index in scored[0][1]):
        return None
    best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else float("-inf")
    return {"score": best[0], "second_score": second_score, "indices": best[1],
            "margin": best[0] - second_score}


def baseline_assignment_score(records: list[dict], pool: list[dict]) -> float:
    axis = records[0]["intent"]["axis"]
    ordered = sorted(range(len(pool)), key=lambda i: (pool[i]["bbox"][0] + pool[i]["bbox"][2]) / 2
                     if axis == "x" else (pool[i]["bbox"][1] + pool[i]["bbox"][3]) / 2)
    order_index = {candidate_index: rank for rank, candidate_index in enumerate(ordered)}
    total = 0.0
    for record in records:
        index = max(range(len(pool)), key=lambda i: compute_iou(record["baseline_bbox"], pool[i]["bbox"]))
        total += assignment_score(record, pool[index], order_index[index], len(pool))
    return total


def process_dataset(name: str, annotations: dict, baseline: dict, details: dict, output_dir: Path,
                    labeled: bool):
    image_groups = defaultdict(list)
    records_by_id = {}
    for query_id, sample in annotations.items():
        base = bbox_from(baseline.get(query_id))
        if base is None:
            continue
        detail = details.get(query_id, {})
        candidates = candidate_list(detail, base)
        intent = position_intent(sample.get("query", detail.get("query", "")))
        object_class = query_class(sample.get("query", detail.get("query", "")), candidates)
        image_id = str(sample.get("source_image_id", sample.get("visible", "")))
        record = {"query_id": query_id, "query": sample.get("query", ""), "image_id": image_id,
                  "baseline_bbox": base, "candidates": candidates, "intent": intent,
                  "object_class": object_class, "source_ref_id": sample.get("source_ref_id")}
        records_by_id[query_id] = record
        if intent and object_class:
            image_groups[(image_id, object_class, intent["axis"])].append(record)

    predictions = {qid: list(record["baseline_bbox"]) for qid, record in records_by_id.items()}
    group_reports = []
    counters = Counter()
    for (image_id, object_class, axis), records in image_groups.items():
        if len(records) < 2:
            continue
        # RefCOCO has multiple sentences per referent. Collapse them for conflict
        # detection; groups containing only one referent are not mutually exclusive.
        distinct_refs = {x["source_ref_id"] for x in records if x["source_ref_id"] is not None}
        if labeled and len(distinct_refs) < 2:
            continue
        labels = {x["intent"]["label"] for x in records}
        if len(labels) < 2:
            continue
        counters["ordinal_groups"] += 1
        counters["group_queries"] += len(records)
        conflict_pairs = []
        for a, b in itertools.combinations(records, 2):
            if labeled and a["source_ref_id"] == b["source_ref_id"]:
                continue
            overlap = compute_iou(a["baseline_bbox"], b["baseline_bbox"])
            if overlap >= 0.85:
                conflict_pairs.append([a["query_id"], b["query_id"], overlap])
        counters["conflict_pairs"] += len(conflict_pairs)
        if conflict_pairs:
            counters["conflict_groups"] += 1
        pool = dedupe_pool(records)
        for record in records:
            record["group_records"] = records
        assignment = best_assignment(records, pool)
        report = {"image_id": image_id, "object_class": object_class, "axis": axis,
                  "query_ids": [x["query_id"] for x in records], "labels": sorted(labels),
                  "candidate_count": len(pool), "conflicts": conflict_pairs,
                  "assignable": assignment is not None, "changes": []}
        if assignment is not None:
            counters["assignable_groups"] += 1
            counters["assignable_queries"] += len(records)
            old_score = baseline_assignment_score(records, pool)
            report.update({"old_score": old_score, "new_score": assignment["score"],
                           "margin": assignment["margin"]})
            # Conservative gate: solve a real conflict, improve total score, and
            # keep a non-trivial separation from the runner-up assignment.
            safe = bool(conflict_pairs) and assignment["score"] > old_score and assignment["margin"] >= 0.02
            for record, index in zip(records, assignment["indices"]):
                new_box = pool[index]["bbox"]
                if compute_iou(new_box, record["baseline_bbox"]) < 0.85:
                    counters["proposed_changes"] += 1
                    change = {"query_id": record["query_id"], "before": record["baseline_bbox"],
                              "after": new_box, "safe": safe}
                    report["changes"].append(change)
                    if safe:
                        predictions[record["query_id"]] = new_box
                        counters["actual_changes"] += 1
        group_reports.append(report)

    result = {"dataset": name, "samples": len(records_by_id), **dict(counters)}
    if labeled:
        before_correct = after_correct = improved = degraded = new_correct = new_wrong = 0
        before_sum = after_sum = 0.0
        for query_id, record in records_by_id.items():
            gt = bbox_from(annotations[query_id].get("bbox"))
            if gt is None:
                continue
            before = compute_iou(record["baseline_bbox"], gt)
            after = compute_iou(predictions[query_id], gt)
            before_sum += before
            after_sum += after
            before_correct += before >= 0.5
            after_correct += after >= 0.5
            improved += after > before + 1e-9
            degraded += after + 1e-9 < before
            new_correct += before < 0.5 <= after
            new_wrong += after < 0.5 <= before
        total = len(records_by_id)
        result.update({"acc_before": before_correct / total, "acc_after": after_correct / total,
                       "delta_pp": 100 * (after_correct - before_correct) / total,
                       "mean_iou_before": before_sum / total, "mean_iou_after": after_sum / total,
                       "improved": improved, "degraded": degraded,
                       "new_correct": new_correct, "new_wrong": new_wrong})
    dump_json(output_dir / f"{name}_groups.json", group_reports)
    dump_json(output_dir / f"{name}_predictions_after.json", predictions)
    return result


def competition_inputs(prediction_path: Path, details_path: Path):
    prediction = load_json(prediction_path)
    annotations = {}
    baseline = {}
    for query_id, sample in prediction.items():
        annotations[query_id] = sample
        baseline[query_id] = sample.get("bbox")
    return annotations, baseline, load_json(details_path)


def markdown_report(comp: dict, ref: dict) -> str:
    def n(data, key):
        return data.get(key, 0)
    lines = [
        "# Joint Assignment V31 Diagnostic",
        "",
        (f"比赛序数组 {n(comp, 'ordinal_groups')} 组；重复冲突 {n(comp, 'conflict_pairs')} 对；"
         f"可一对一分配 {n(comp, 'assignable_queries')} 条；相对 V28 预计修改 {n(comp, 'actual_changes')} 条；"
         f"RefCOCO ACC：{100*ref['acc_before']:.2f}% → {100*ref['acc_after']:.2f}% "
         f"({ref['delta_pp']:+.2f} pp)。"),
        "", "## Competition", "",
        f"- Samples: {comp['samples']}",
        f"- Ordinal/position groups: {n(comp, 'ordinal_groups')}",
        f"- Conflict groups / pairs: {n(comp, 'conflict_groups')} / {n(comp, 'conflict_pairs')}",
        f"- Assignable groups / queries: {n(comp, 'assignable_groups')} / {n(comp, 'assignable_queries')}",
        f"- Proposed / actual changes: {n(comp, 'proposed_changes')} / {n(comp, 'actual_changes')}",
        "", "## RefCOCO", "",
        f"- Samples: {ref['samples']}",
        f"- ACC@0.5 before / after: {100*ref['acc_before']:.2f}% / {100*ref['acc_after']:.2f}%",
        f"- Change: {ref['delta_pp']:+.2f} pp",
        f"- New correct / new wrong: {ref['new_correct']} / {ref['new_wrong']}",
        f"- IoU improved / degraded: {ref['improved']} / {ref['degraded']}",
        f"- Proposed / actual changes: {n(ref, 'proposed_changes')} / {n(ref, 'actual_changes')}",
        "", "## Gate", "",
        "PASS" if ref["delta_pp"] >= 1.0 else "STOP: the +1.00 pp gate was not reached.",
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/experiments/joint_assignment_v31")
    parser.add_argument("--competition-prediction", default="outputs/experiments/joint_assignment_v31/v28/prediction.json")
    parser.add_argument("--competition-details", default="outputs/leaderboard/best_v20_02695/reranked_details_v20.json")
    parser.add_argument("--ref-annotations", default="data/validation/refcoco_val.json")
    parser.add_argument("--ref-baseline", default="outputs/experiments/refcoco_qwen3b_complex_all/reranked_predictions.json")
    parser.add_argument("--ref-details", default="outputs/experiments/refcoco_v18_all/details.json")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comp_ann, comp_base, comp_details = competition_inputs(
        resolve(args.competition_prediction), resolve(args.competition_details))
    comp = process_dataset("competition", comp_ann, comp_base, comp_details, output_dir, False)
    ref_ann = load_json(resolve(args.ref_annotations))
    ref_base = load_json(resolve(args.ref_baseline))
    ref_details = load_json(resolve(args.ref_details))
    ref = process_dataset("refcoco", ref_ann, ref_base, ref_details, output_dir, True)
    metrics = {"competition": comp, "refcoco": ref}
    dump_json(output_dir / "metrics.json", metrics)
    report = markdown_report(comp, ref)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Details: {output_dir}")


if __name__ == "__main__":
    main()
