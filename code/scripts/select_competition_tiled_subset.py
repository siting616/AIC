"""Build the frozen competition subset from assignable joint-position groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", default="outputs/experiments/joint_assignment_v31/v28/prediction.json")
    parser.add_argument("--groups", default="outputs/experiments/joint_assignment_v31/competition_groups.json")
    parser.add_argument("--output", default="outputs/experiments/tiled_competition/competition_subset.json")
    args = parser.parse_args()
    resolve = lambda x: Path(x) if Path(x).is_absolute() else PROJECT_ROOT / x
    prediction = json.loads(resolve(args.prediction).read_text(encoding="utf-8"))
    groups = json.loads(resolve(args.groups).read_text(encoding="utf-8"))
    ids = []
    seen = set()
    for group in groups:
        if not group.get("assignable"):
            continue
        for query_id in group.get("query_ids", []):
            if query_id not in seen:
                ids.append(query_id)
                seen.add(query_id)
    subset = []
    for query_id in ids:
        sample = prediction[query_id]
        subset.append({
            "query_id": query_id, "query": sample["query"], "visible": sample["visible"],
            "baseline_bbox": sample["bbox"],
        })
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Competition tiled subset: {len(subset)} queries")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
