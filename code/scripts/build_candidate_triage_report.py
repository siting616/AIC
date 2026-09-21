"""Build a triage report for tiled-rescue candidate review."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "color": re.compile(
        r"\b(red|blue|green|white|black|pink|orange|yellow|brown|gray|grey|purple|silver|gold)\b",
        re.I,
    ),
    "ordinal": re.compile(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|"
        r"rightmost|leftmost|topmost|bottommost|from left to right|from right to left)\b",
        re.I,
    ),
    "directional": re.compile(
        r"\b(far right|far left|rightmost|leftmost|topmost|bottommost|from left to right|"
        r"from right to left|on the right|on the left|right side|left side)\b",
        re.I,
    ),
    "spatial": re.compile(
        r"\b(left of|right of|to the left of|to the right of|above|below|beside|between|"
        r"near|closest|farthest|behind|in front of|across|lower|upper|top|bottom|center|corner)\b",
        re.I,
    ),
    "relational": re.compile(
        r"\b(left of|right of|to the left of|to the right of|above|below|beside|between|"
        r"closest|farthest|furthest|behind|in front of|adjacent to)\b",
        re.I,
    ),
    "small_target": re.compile(
        r"\b(small|tiny|little|distant|distance|background|far|remote|partly hidden|"
        r"partially hidden|partially cut|cropped|extreme)\b",
        re.I,
    ),
}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path: str):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def pairwise_support(item: dict) -> bool:
    checks = item.get("pairwise_checks", [])
    return bool(checks) and all(check.get("supports_challenger") for check in checks)


def classify(item: dict) -> tuple[list[str], list[str]]:
    query = item["query"]
    tags = [name for name, pattern in PATTERNS.items() if pattern.search(query)]
    candidate_iou = float(item.get("candidate_iou", 0.0))
    if candidate_iou < 0.1:
        tags.append("identity_switch")
    elif candidate_iou >= 0.5:
        tags.append("boundary_only")
    else:
        tags.append("partial_overlap")

    risks = []
    area_ratio = float(item.get("area_ratio", 1.0))
    rank = int(item.get("candidate_rank", 999))
    score = float(item.get("candidate_score", 0.0))
    agreement = int(item.get("stage1_agreement", 0))
    before_area = area(item["before"])
    after_area = area(item["after"])
    if area_ratio < 0.18 or area_ratio > 5.0:
        risks.append("extreme_area_ratio")
    if area_ratio < 0.35 or area_ratio > 2.5:
        risks.append("large_area_shift")
    if after_area > 0.18:
        risks.append("large_candidate_box")
    if before_area > 0 and after_area / before_area > 4:
        risks.append("much_larger_than_baseline")
    if rank > 10:
        risks.append("low_rank_candidate")
    if score < 0.15:
        risks.append("low_detector_score")
    elif score < 0.2:
        risks.append("medium_low_detector_score")
    if agreement < 3:
        risks.append("stage1_not_unanimous")
    if not pairwise_support(item):
        risks.append("pairwise_not_unanimous")
    if "boundary_only" in tags:
        risks.append("boundary_only")
    if "identity_switch" in tags and "ordinal" in tags:
        risks.append("ordinal_identity_switch")
    if "identity_switch" in tags and "directional" in tags:
        risks.append("directional_position_switch")
    if "identity_switch" in tags and "relational" in tags:
        risks.append("relational_position_switch")
    if "partial_overlap" in tags and (rank > 5 or area_ratio < 0.7 or area_ratio > 1.6):
        risks.append("partial_overlap_refinement")
    if "identity_switch" in tags and score < 0.2:
        risks.append("low_score_identity_switch")
    if "large_area_shift" in risks and ("much_larger_than_baseline" in risks or rank > 3):
        risks.append("geometry_unstable")
    return tags, risks


def score_item(item: dict, tags: list[str], risks: list[str]) -> tuple[int, str, str]:
    score = 0
    reasons = []
    if int(item.get("stage1_agreement", 0)) == 3:
        score += 3
        reasons.append("stage1 unanimous")
    if pairwise_support(item):
        score += 3
        reasons.append("pairwise unanimous")
    if float(item.get("candidate_iou", 0.0)) < 0.1:
        score += 2
        reasons.append("identity switch")
    if any(tag in tags for tag in ("color", "small_target")):
        score += 2
        reasons.append("query has strong visual cue")
    elif "spatial" in tags:
        score += 1
        reasons.append("query has spatial cue")
    if float(item.get("candidate_score", 0.0)) >= 0.25:
        score += 1
        reasons.append("detector score >= 0.25")
    if int(item.get("candidate_rank", 999)) <= 3:
        score += 1
        reasons.append("candidate rank <= 3")

    penalties = {
        "extreme_area_ratio": 3,
        "large_candidate_box": 2,
        "much_larger_than_baseline": 2,
        "low_rank_candidate": 1,
        "low_detector_score": 1,
        "medium_low_detector_score": 2,
        "stage1_not_unanimous": 4,
        "pairwise_not_unanimous": 4,
        "boundary_only": 3,
        "large_area_shift": 2,
        "ordinal_identity_switch": 4,
        "directional_position_switch": 4,
        "relational_position_switch": 3,
        "partial_overlap_refinement": 3,
        "low_score_identity_switch": 3,
        "geometry_unstable": 3,
    }
    for risk in risks:
        penalty = penalties.get(risk, 0)
        if penalty:
            score -= penalty
            reasons.append(f"-{penalty} {risk}")

    hard_risks = {
        "stage1_not_unanimous",
        "pairwise_not_unanimous",
        "boundary_only",
        "ordinal_identity_switch",
        "directional_position_switch",
        "relational_position_switch",
        "partial_overlap_refinement",
        "low_score_identity_switch",
        "geometry_unstable",
    }
    if score >= 8 and not any(r in risks for r in hard_risks):
        action = "ACCEPT_REVIEW"
        risk_level = "low"
    elif score >= 5 or any(
        r in risks for r in ("ordinal_identity_switch", "directional_position_switch", "relational_position_switch")
    ):
        action = "DEFER_REVIEW"
        risk_level = "medium"
    elif "boundary_only" in risks:
        action = "BOUNDARY_HOLD"
        risk_level = "medium"
    else:
        action = "REJECT_OR_LATE"
        risk_level = "high"
    return score, risk_level, action + " | " + "; ".join(reasons)


def row_for_item(item: dict) -> dict:
    tags, risks = classify(item)
    priority_score, risk_level, recommendation = score_item(item, tags, risks)
    return {
        "sample_id": item["sample_id"],
        "query": item["query"],
        "type_tags": ",".join(tags),
        "risk_tags": ",".join(risks),
        "candidate_score": f"{float(item.get('candidate_score', 0.0)):.6f}",
        "candidate_rank": item.get("candidate_rank"),
        "candidate_iou": f"{float(item.get('candidate_iou', 0.0)):.6f}",
        "area_ratio": f"{float(item.get('area_ratio', 1.0)):.6f}",
        "stage1_agreement": item.get("stage1_agreement"),
        "pairwise_support": pairwise_support(item),
        "priority_score": priority_score,
        "risk_level": risk_level,
        "recommended_action": recommendation.split(" | ", 1)[0],
        "reason": recommendation.split(" | ", 1)[1],
    }


def write_markdown(rows: list[dict], path: Path, limit: int) -> None:
    action_counts = {
        action: sum(r["recommended_action"] == action for r in rows)
        for action in ("ACCEPT_REVIEW", "DEFER_REVIEW", "BOUNDARY_HOLD", "REJECT_OR_LATE")
    }
    lines = [
        "# Candidate Triage Report",
        "",
        f"- Total candidates: {len(rows)}",
        f"- ACCEPT_REVIEW: {action_counts['ACCEPT_REVIEW']}",
        f"- DEFER_REVIEW: {action_counts['DEFER_REVIEW']}",
        f"- BOUNDARY_HOLD: {action_counts['BOUNDARY_HOLD']}",
        f"- REJECT_OR_LATE: {action_counts['REJECT_OR_LATE']}",
        "",
        "## Calibration",
        "",
        "- Ordinal/directional identity switches are deferred because the failed clean5 package showed that unanimous votes can still pick the wrong instance.",
        "- Partial-overlap refinements with weak geometry are held back; they often move a box without changing the target correctly.",
        "- Color and small-target cues keep positive weight because they are more inspectable and less dependent on relative counting.",
        "",
        f"## Top {limit}",
        "",
        "| rank | sample_id | score | action | tags | risks | query |",
        "|---:|---|---:|---|---|---|---|",
    ]
    for index, row in enumerate(rows[:limit], start=1):
        query = row["query"].replace("|", "\\|")
        lines.append(
            f"| {index} | {row['sample_id']} | {row['priority_score']} | "
            f"{row['recommended_action']} | {row['type_tags']} | {row['risk_tags']} | {query} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--used-plan", nargs="*", default=[])
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    eligible = load(args.eligible)
    used_ids = set()
    for used_plan in args.used_plan:
        used = load(used_plan)
        if isinstance(used, list):
            used_ids.update(item["sample_id"] for item in used if "sample_id" in item)
    eligible = [item for item in eligible if item["sample_id"] not in used_ids]
    rows = [row_for_item(item) for item in eligible]
    rows.sort(
        key=lambda row: (
            int(row["priority_score"]),
            float(row["candidate_score"]),
            -int(row["candidate_rank"]),
        ),
        reverse=True,
    )
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "candidate_triage_report.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    accept_ids = [row["sample_id"] for row in rows if row["recommended_action"] == "ACCEPT_REVIEW"]
    (output / "accept_review_ids.json").write_text(
        json.dumps(accept_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "top_review_ids.json").write_text(
        json.dumps([row["sample_id"] for row in rows[: args.top_n]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(rows, output / "candidate_triage_report.md", args.top_n)
    metrics = {
        "total": len(rows),
        "excluded_used": len(used_ids),
        "accept_review": len(accept_ids),
        "top_n": args.top_n,
        "csv": str(csv_path),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
