"""Apply accepted visual-audit groups to a diagnostic V28 copy."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def main():
    root = "outputs/experiments/v31_change_plan"
    decisions = load(f"{root}/audit/manual_decisions.json")
    groups = load(f"{root}/accepted_groups.json")
    changes = load(f"{root}/changes.json")
    v28 = load("outputs/experiments/joint_assignment_v31/v28/prediction.json")
    if decisions.get("pending"):
        raise SystemExit(f"Manual review still has pending groups: {decisions['pending']}")
    accepted_indices = set(decisions["accepted"])
    accepted_groups = [group for index, group in enumerate(groups, 1) if index in accepted_indices]
    accepted_ids = {qid for group in accepted_groups for qid in group["changed_query_ids"]}
    final_changes = {qid: changes[qid] for qid in sorted(accepted_ids)}
    prediction = json.loads(json.dumps(v28))
    for qid, change in final_changes.items():
        prediction[qid]["bbox"] = change["after"]
    output = PROJECT_ROOT / root / "manual_final"
    output.mkdir(parents=True, exist_ok=True)
    values = {
        "manual_groups_accepted": len(accepted_groups),
        "manual_groups_rejected": len(groups) - len(accepted_groups),
        "changed_queries": len(final_changes),
        "reasons": dict(Counter(x["reason"] for x in final_changes.values())),
        "submission_created": False,
    }
    for name, value in (("accepted_groups.json", accepted_groups), ("changes.json", final_changes),
                        ("prediction_diagnostic_only.json", prediction), ("metrics.json", values)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(values, ensure_ascii=False, indent=2))
    print("Manual review applied; no submission archive was created.")


if __name__ == "__main__":
    main()
