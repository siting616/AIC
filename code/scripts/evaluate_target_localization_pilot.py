"""Evaluate blind target-localization choices against separate pilot labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = "ABCDEFGHIJKLMNOPQRST"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choices", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    choices = json.loads(Path(args.choices).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    totals = {"samples": len(labels), "parsed": 0, "consensus": 0, "correct": 0,
              "controls": 0, "controls_preserved": 0, "rescuable": 0, "rescued": 0,
              "damaged": 0}
    for sample_id, truth in labels.items():
        choice = choices.get(sample_id, {})
        label = choice.get("majority_label")
        totals["parsed"] += bool(choice)
        totals["consensus"] += bool(label)
        selected = LABELS.index(label) if isinstance(label, str) and label in LABELS else None
        correct = selected in truth["correct_candidate_indices"] if selected is not None else truth["baseline_correct"]
        totals["correct"] += correct
        if truth["baseline_correct"]:
            totals["controls"] += 1; totals["controls_preserved"] += correct; totals["damaged"] += not correct
        else:
            totals["rescuable"] += 1; totals["rescued"] += correct
    totals.update({
        "accuracy": totals["correct"] / totals["samples"],
        "consensus_rate": totals["consensus"] / totals["samples"],
        "control_preservation_rate": totals["controls_preserved"] / totals["controls"],
        "rescue_rate": totals["rescued"] / totals["rescuable"],
        "net_rescues": totals["rescued"] - totals["damaged"],
        "passed": False,
    })
    totals["passed"] = (totals["accuracy"] >= .60 and totals["control_preservation_rate"] >= .85
                        and totals["rescue_rate"] >= .35 and totals["net_rescues"] > 0)
    Path(args.output).write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
