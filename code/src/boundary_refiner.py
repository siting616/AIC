"""Conservative application of cached SAM boundary refinements."""

from __future__ import annotations

from .metrics import is_valid_bbox


def apply_cached_boundary_refinements(base_records, refinement_records, minimum_score=0.95):
    output = {}
    stats = {"samples": len(base_records), "matched_input": 0, "score_pass": 0, "changed": 0}
    for sample_id, base in base_records.items():
        ranked = [dict(item) for item in base.get("candidates", [])]
        current = ranked[0]["bbox"] if ranked else base.get("bbox")
        refinement = refinement_records.get(sample_id)
        if refinement and current == refinement.get("input_bbox"):
            stats["matched_input"] += 1
            refined = refinement.get("refined_bbox")
            if float(refinement.get("sam_score", 0.0)) >= minimum_score:
                stats["score_pass"] += 1
                if is_valid_bbox(refined) and refined != current:
                    if ranked:
                        ranked[0]["bbox_before_boundary_refine"] = current
                        ranked[0]["bbox"] = [float(value) for value in refined]
                        ranked[0]["boundary_refiner"] = "sam2_cached"
                        ranked[0]["sam_score"] = float(refinement["sam_score"])
                    current = [float(value) for value in refined]
                    stats["changed"] += 1
        output[sample_id] = {"bbox": current, "candidates": ranked}
    return output, stats
