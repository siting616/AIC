"""Evaluate pairwise verifier decisions against frozen pilot labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = "ABCDEFGHIJKLMNOPQRST"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--pairwise", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stage1 = json.loads(Path(args.stage1).read_text(encoding="utf-8"))
    pairwise = json.loads(Path(args.pairwise).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    result = {"samples": len(labels), "accepted": 0, "controls": 0, "preserved": 0,
              "rescuable": 0, "rescued": 0, "damaged": 0, "correct": 0}
    for sample_id, truth in labels.items():
        accept = pairwise.get(sample_id, {}).get("accept_challenger", False)
        label = stage1.get(sample_id, {}).get("majority_label") if accept else None
        selected = LABELS.index(label) if isinstance(label, str) and label in LABELS else None
        correct = selected in truth["correct_candidate_indices"] if selected is not None else truth["baseline_correct"]
        result["accepted"] += accept; result["correct"] += correct
        if truth["baseline_correct"]:
            result["controls"] += 1; result["preserved"] += correct; result["damaged"] += not correct
        else:
            result["rescuable"] += 1; result["rescued"] += correct
    result.update({"accuracy": result["correct"] / result["samples"],
                   "preservation_rate": result["preserved"] / result["controls"],
                   "rescue_rate": result["rescued"] / result["rescuable"],
                   "net_rescues": result["rescued"] - result["damaged"]})
    result["passed"] = (result["accuracy"] >= .60 and result["preservation_rate"] >= .85
                        and result["rescue_rate"] >= .35 and result["net_rescues"] > 0)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
