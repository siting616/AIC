"""Evaluate saved candidate pools on any labeled competition-format dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import evaluate_candidate_records
from src.experiment_registry import append_registry, build_registry_row, run_fingerprint
from src.validation_split import dataset_fingerprint, load_manifest, subset_annotations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Labeled competition-format JSON")
    parser.add_argument("--candidates", required=True, help="Candidate records or bbox prediction JSON")
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-manifest", help="Frozen split manifest JSON")
    parser.add_argument("--split", choices=("dev", "holdout"), help="Partition from split manifest")
    parser.add_argument("--experiment-name", default="unnamed")
    parser.add_argument("--registry", default="outputs/validation/experiments.csv")
    parser.add_argument("--no-register", action="store_true")
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pct(value):
    return f"{100.0 * value:.2f}%"


def markdown_report(result):
    overall = result["overall"]
    lines = [
        "# Candidate Validation Report",
        "",
        f"- Labeled samples: {result['labeled_samples']}",
        f"- Skipped unlabeled samples: {result['skipped_unlabeled']}",
        f"- IoU threshold: {result['threshold']}",
        f"- Candidate coverage: {pct(overall['candidate_coverage'])}",
        f"- Invalid candidate rate: {pct(overall['invalid_candidate_rate'])}",
        f"- Top-1 ACC@0.5: {pct(overall['acc_at_05'])}",
        f"- Mean Top-1 IoU: {overall['mean_top1_iou']:.4f}",
        f"- Mean best-candidate IoU: {overall['mean_best_iou']:.4f}",
        "",
        "## Oracle ACC@0.5",
        "",
        "| Top-K | Oracle ACC@0.5 | Gap vs Top-1 |",
        "|---:|---:|---:|",
    ]
    for k, value in overall["oracle_acc"].items():
        lines.append(f"| {k} | {pct(value)} | {pct(value - overall['acc_at_05'])} |")
    lines += [
        "",
        "## Query categories",
        "",
        "| Category | Samples | Top-1 ACC | Oracle@5 | Oracle@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in result["by_category"].items():
        oracle = metrics["oracle_acc"]
        at5 = oracle.get("5", oracle.get(max(oracle, key=int), 0.0))
        at20 = oracle.get("20", oracle.get(max(oracle, key=int), 0.0))
        lines.append(f"| {name} | {metrics['samples']} | {pct(metrics['acc_at_05'])} | {pct(at5)} | {pct(at20)} |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    annotations = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    full_dataset_fp = dataset_fingerprint(annotations)
    if bool(args.split_manifest) != bool(args.split):
        raise SystemExit("--split-manifest and --split must be supplied together")
    if args.split_manifest:
        annotations = subset_annotations(
            annotations, load_manifest(resolve(args.split_manifest)), args.split
        )
    candidates_path = resolve(args.candidates)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if isinstance(candidates, dict) and candidates.get("schema_version") == 1 and isinstance(candidates.get("records"), dict):
        candidates = candidates["records"]
    result = evaluate_candidate_records(annotations, candidates, args.top_k, args.threshold)
    if result["labeled_samples"] == 0:
        raise SystemExit("No valid labeled bbox found in annotations; offline evaluation cannot run.")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(
        full_dataset_fp, candidates_path, args.split or "all", args.threshold, args.top_k
    )
    result["experiment"] = {
        "name": args.experiment_name,
        "split": args.split or "all",
        "run_fingerprint": fingerprint,
        "dataset_fingerprint": full_dataset_fp,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(markdown_report(result), encoding="utf-8")
    if not args.no_register:
        row = build_registry_row(
            args.experiment_name, args.split or "all", fingerprint, full_dataset_fp,
            result, candidates_path, metrics_path,
        )
        append_registry(resolve(args.registry), row)
    print(markdown_report(result))
    print(f"Detailed metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
