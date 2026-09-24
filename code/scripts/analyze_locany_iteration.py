"""Measure box agreement between LocateAnything and archived competition runs.

Agreement is not an accuracy metric because the competition set has no labels.
"""

import argparse
import json
from pathlib import Path
import re
import statistics


ORDINAL = re.compile(r"\b(first|second|third|fourth|leftmost|rightmost|topmost|bottommost)\b", re.I)
RELATION = re.compile(r"\b(near|next to|beside|behind|in front of|between|closest|farthest)\b", re.I)


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a, b):
    cross = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    overlap = area(cross)
    return overlap / max(area(a) + area(b) - overlap, 1e-12)


def summarize(a, b, ids):
    scores = [iou(a[qid]["bbox"], b[qid]["bbox"]) for qid in ids]
    return {
        "count": len(scores),
        "median_iou": round(statistics.median(scores), 4),
        "iou_below_0_5": sum(score < 0.5 for score in scores),
        "iou_below_0_5_percent": round(100 * sum(score < 0.5 for score in scores) / len(scores), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locany", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    args = parser.parse_args()
    runs = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in (
        ("locany", args.locany), ("baseline", args.baseline), ("v3", args.v3)
    )}
    ids = list(runs["locany"])
    if not all(list(run) == ids for run in runs.values()):
        raise ValueError("Query IDs or order differ between runs")
    groups = {
        "all": ids,
        "ordinal": [qid for qid in ids if ORDINAL.search(runs["locany"][qid]["query"])],
        "relation": [qid for qid in ids if RELATION.search(runs["locany"][qid]["query"])],
    }
    output = {
        name: {
            group: summarize(runs["locany"], runs[name], qids)
            for group, qids in groups.items()
        }
        for name in ("baseline", "v3")
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
