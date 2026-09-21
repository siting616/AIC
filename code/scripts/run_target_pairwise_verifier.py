"""Pairwise verify proposed target changes; never creates a submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qwen_candidate_selector import LABELS, load_model
from scripts.run_qwen_pairwise_rescue import annotate_pair, select_pair
from src.dataset import AICDataset


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images-dir")
    parser.add_argument("--ids-json", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--stage1-choices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dataset = AICDataset(str(PROJECT_ROOT), str(resolve(args.annotations)), images_dir=args.images_dir)
    samples = {dataset[i]["sample_id"]: dataset[i] for i in range(len(dataset))}
    ids = json.loads(resolve(args.ids_json).read_text(encoding="utf-8"))
    cache = json.loads(resolve(args.candidates).read_text(encoding="utf-8"))
    records = cache.get("records", cache)
    baseline = json.loads(resolve(args.baseline).read_text(encoding="utf-8"))
    stage1 = json.loads(resolve(args.stage1_choices).read_text(encoding="utf-8"))
    eligible = []
    for sample_id in ids:
        choice = stage1.get(sample_id, {})
        selected = choice.get("majority_label")
        if not isinstance(selected, str) or selected not in LABELS or selected == choice.get("baseline_label"):
            continue
        index = LABELS.index(selected)
        if index < len(records[sample_id]["candidates"]):
            eligible.append((sample_id, records[sample_id]["candidates"][index]["bbox"]))
    output = resolve(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "eligible.json").write_text(json.dumps(
        [{"sample_id": sid, "candidate_bbox": box} for sid, box in eligible], indent=2), encoding="utf-8")
    checkpoint = output / "pairwise_checkpoint.json"
    decisions = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_OK", "eligible": len(eligible),
                          "completed": len(decisions)}, indent=2)); return
    model, processor = load_model(args.model_id, not args.no_4bit)
    for sample_id, challenger in tqdm(eligible, desc="Pairwise target verification", unit="query"):
        if sample_id in decisions and len(decisions[sample_id].get("checks", [])) == 2:
            continue
        sample = samples[sample_id]
        old = baseline[sample_id].get("bbox") if isinstance(baseline[sample_id], dict) else baseline[sample_id]
        checks = decisions.get(sample_id, {}).get("checks", [])
        for check_index in range(len(checks), 2):
            swapped = int(hashlib.sha256(f"{sample_id}:{check_index}".encode()).hexdigest(), 16) % 2 == 1
            first, second = (challenger, old) if swapped else (old, challenger)
            image = annotate_pair(sample["visible_path"], first, second)
            label, raw = select_pair(model, processor, image, sample["query"], verification=bool(check_index))
            challenger_label = "A" if swapped else "B"
            checks.append({"label": label, "raw": raw, "swapped": swapped,
                           "challenger_label": challenger_label,
                           "supports_challenger": label == challenger_label})
            decisions[sample_id] = {"checks": checks, "candidate_bbox": challenger}
            checkpoint.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
        decisions[sample_id]["accept_challenger"] = all(x["supports_challenger"] for x in checks)
        checkpoint.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"eligible={len(eligible)} completed={len(decisions)} output={checkpoint}")


if __name__ == "__main__":
    main()
