"""Generate CPU-only, proposal-only V2 ordinal diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v2_delta import dump_json, load_frozen_baseline, load_json, resolve
from src.v2.ordinal_solver import build_ordinal_proposals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="outputs/leaderboard/best_baseline.json")
    parser.add_argument("--details", default="outputs/leaderboard/best_v20_02695/reranked_details_v20.json")
    parser.add_argument("--output-dir", default="outputs/v2/ordinal")
    parser.add_argument("--scene")
    args = parser.parse_args()
    registry, records, _ = load_frozen_baseline(resolve(args.registry))
    details = load_json(resolve(args.details))
    if args.scene:
        prefix = f"{args.scene}_"
        records = {key: value for key, value in records.items() if key.startswith(prefix)}
        details = {key: value for key, value in details.items() if key.startswith(prefix)}
    result = build_ordinal_proposals(records, details)
    result["baseline_version"] = registry["version"]
    result["baseline_score"] = registry.get("leaderboard_score")
    result["proposal_only"] = True
    output_dir = resolve(args.output_dir)
    dump_json(output_dir / "ordinal_report.json", result)
    dump_json(output_dir / "change_proposals.json", result["change_proposals"])
    recovery = [family for family in result["families"] if family["needs_tiled_recovery"]]
    dump_json(output_dir / "tiled_recovery_manifest.json", recovery)
    print(json.dumps({**result["summary"], "proposal_only": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
