"""Merge GPU tiled family candidates and rerun proposal-only ordinal solving."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v2_delta import dump_json, load_frozen_baseline, load_json, resolve
from src.v2.ordinal_solver import build_ordinal_proposals


def inject_tiled_candidates(details: dict, tiled: dict) -> tuple[dict, int]:
    merged = {key: dict(value) for key, value in details.items()}
    injected = 0
    for family in tiled.values():
        candidates = list(family.get("candidates", []))
        for query_id in family.get("query_ids", []):
            record = dict(merged.get(query_id, {}))
            existing = list(record.get("reranked_candidates") or record.get("candidates") or [])
            record["reranked_candidates"] = existing + candidates
            record["v2_tiled_candidate_count"] = len(candidates)
            merged[query_id] = record
            injected += len(candidates)
    return merged, injected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="outputs/leaderboard/best_baseline.json")
    parser.add_argument("--details", default="outputs/leaderboard/best_v20_02695/reranked_details_v20.json")
    parser.add_argument("--tiled", default="outputs/v2/ordinal/tiled_gpu/tiled_family_candidates.json")
    parser.add_argument("--output-dir", default="outputs/v2/ordinal/merged")
    args = parser.parse_args()
    registry, records, _ = load_frozen_baseline(resolve(args.registry))
    details = load_json(resolve(args.details))
    tiled = load_json(resolve(args.tiled))
    merged_details, injected = inject_tiled_candidates(details, tiled)
    result = build_ordinal_proposals(records, merged_details)
    result.update({
        "baseline_version": registry["version"],
        "baseline_score": registry.get("leaderboard_score"),
        "tiled_families_loaded": len(tiled),
        "candidate_references_injected": injected,
        "proposal_only": True,
    })
    output = resolve(args.output_dir)
    dump_json(output / "ordinal_merged_report.json", result)
    dump_json(output / "change_proposals.json", result["change_proposals"])
    dump_json(
        output / "remaining_recovery_manifest.json",
        [family for family in result["families"] if family["needs_tiled_recovery"]],
    )
    print(json.dumps({
        **result["summary"],
        "tiled_families_loaded": len(tiled),
        "candidate_references_injected": injected,
        "proposal_only": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

