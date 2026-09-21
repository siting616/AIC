"""Stage-1 V2 planner: inspect routes and copy no boxes.

The command resolves the authoritative baseline registry, reads prediction.json
from its ZIP, and writes planning diagnostics.  It never modifies the registry,
the baseline archive, or any bbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.v2.scene_planner import build_scene_plan


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_frozen_baseline(registry_path: Path) -> tuple[dict, dict, Path]:
    registry = load_json(registry_path)
    archive = resolve(registry["submission_zip"])
    expected = registry.get("zip_sha256")
    actual = file_sha256(archive)
    if expected and actual != expected:
        raise RuntimeError(f"Frozen baseline hash mismatch: expected {expected}, got {actual}")
    member = registry.get("zip_member", "prediction.json")
    with zipfile.ZipFile(archive) as bundle:
        records = json.loads(bundle.read(member).decode("utf-8"))
    if len(records) != int(registry["sample_count"]):
        raise RuntimeError("Frozen baseline sample count does not match registry")
    return registry, records, archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="outputs/leaderboard/best_baseline.json")
    parser.add_argument("--output-dir", default="outputs/v2")
    parser.add_argument("--scene", help="Optionally keep one scene in the diagnostic plan")
    args = parser.parse_args()

    registry, records, archive = load_frozen_baseline(resolve(args.registry))
    if args.scene:
        prefix = f"{args.scene}_"
        records = {key: value for key, value in records.items() if key.startswith(prefix)}
    plan = build_scene_plan(records)
    output_dir = resolve(args.output_dir)
    dump_json(output_dir / "route_plan.json", plan)
    dump_json(output_dir / "change_report.json", [])
    summary = {
        "stage": "V2-stage1-plan-only",
        "baseline_version": registry["version"],
        "baseline_score": registry.get("leaderboard_score"),
        "baseline_archive": str(archive),
        "samples_planned": plan["sample_count"],
        "scenes_planned": plan["scene_count"],
        "route_counts": plan["route_counts"],
        "changed_bbox_count": 0,
        "baseline_immutable": True,
    }
    dump_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

