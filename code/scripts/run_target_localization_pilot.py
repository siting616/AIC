"""Run a three-view VLM target-selection pilot without creating a submission."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qwen_candidate_selector import LABELS, load_model
from src.dataset import AICDataset
from src.metrics import compute_iou


PROMPTS = {
    "semantic": (
        "Resolve the referring expression in stages: identify the target object class, "
        "its attributes, then its spatial/ordinal relations. Choose the single candidate "
        "that satisfies the complete expression."
    ),
    "elimination": (
        "Eliminate candidates with the wrong object identity or attributes first. Then "
        "use the full-scene image to resolve position and relations. Choose the one "
        "remaining candidate that best matches the complete expression."
    ),
    "counterfactual": (
        "Independently verify the current choice against every alternative. Reject a box "
        "that merely covers a larger related object. Select the exact object referred to."
    ),
}


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def annotated_scene(image, candidates):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    width, height = result.size
    for index, candidate in enumerate(candidates):
        box = candidate["bbox"]
        pixels = [int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)]
        color = (255, 32, 32)
        draw.rectangle(pixels, outline=color, width=max(3, round(min(width, height) / 250)))
        draw.rectangle([pixels[0], max(0, pixels[1] - 24), pixels[0] + 24, pixels[1]], fill=color)
        draw.text((pixels[0] + 7, max(0, pixels[1] - 20)), LABELS[index], fill="white", font=font)
    return result


def crop_montage(image, candidates, cell=224, columns=4):
    rows = (len(candidates) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell, rows * cell), "white")
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
    width, height = image.size
    for index, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate["bbox"]
        crop = image.crop((max(0, int(x1 * width)), max(0, int(y1 * height)),
                           min(width, int(x2 * width)), min(height, int(y2 * height))))
        crop.thumbnail((cell - 8, cell - 28))
        left = (index % columns) * cell + (cell - crop.width) // 2
        top = (index // columns) * cell + 24
        canvas.paste(crop, (left, top))
        draw.rectangle([(index % columns) * cell, (index // columns) * cell,
                        (index % columns + 1) * cell - 1, (index // columns + 1) * cell - 1], outline=(80, 80, 80))
        draw.text(((index % columns) * cell + 8, (index // columns) * cell + 6),
                  LABELS[index], fill=(220, 0, 0), font=font)
    return canvas


def select(model, processor, scene, crops, query, instruction, valid):
    from qwen_vl_utils import process_vision_info

    prompt = (f"{instruction} Referring expression: {query!r}. Valid candidate labels: {valid}. "
              "Reply with exactly one label and no explanation.")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": scene}, {"type": "image", "image": crops},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().upper()
    match = re.search(r"\b([A-T])\b", raw)
    label = match.group(1) if match and match.group(1) in valid else None
    return label, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images-dir")
    parser.add_argument("--ids-json", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dataset = AICDataset(str(PROJECT_ROOT), str(resolve(args.annotations)), images_dir=args.images_dir)
    samples = {dataset[i]["sample_id"]: dataset[i] for i in range(len(dataset))}
    ids = json.loads(resolve(args.ids_json).read_text(encoding="utf-8"))
    cache = json.loads(resolve(args.candidates).read_text(encoding="utf-8"))
    records = cache.get("records", cache)
    baseline = json.loads(resolve(args.baseline).read_text(encoding="utf-8"))
    missing = [sample_id for sample_id in ids if sample_id not in samples or sample_id not in records]
    if missing:
        raise ValueError(f"missing {len(missing)} pilot records; first: {missing[0]}")
    invalid = [sample_id for sample_id in ids if not records[sample_id].get("candidates")]
    if invalid:
        raise ValueError(f"empty candidates for {len(invalid)} pilot records; first: {invalid[0]}")
    output = resolve(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "choices_checkpoint.json"
    choices = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    if args.dry_run:
        counts = [min(len(records[sample_id]["candidates"]), len(LABELS)) for sample_id in ids]
        print(json.dumps({"status": "DRY_RUN_OK", "samples": len(ids),
                          "minimum_candidates": min(counts), "maximum_candidates": max(counts),
                          "already_completed": len(choices)}, indent=2))
        return
    model, processor = load_model(args.model_id, not args.no_4bit)
    for sample_id in tqdm(ids, desc="Target localization", unit="query"):
        if sample_id in choices and len(choices[sample_id].get("votes", {})) == len(PROMPTS):
            continue
        sample = samples[sample_id]
        candidates = records[sample_id]["candidates"][:len(LABELS)]
        image = Image.open(sample["visible_path"]).convert("RGB")
        scene, crops = annotated_scene(image, candidates), crop_montage(image, candidates)
        valid = LABELS[:len(candidates)]
        votes = choices.get(sample_id, {}).get("votes", {})
        for name, instruction in PROMPTS.items():
            if name not in votes:
                label, raw = select(model, processor, scene, crops, sample["query"], instruction, valid)
                votes[name] = {"label": label, "raw": raw}
                choices[sample_id] = {"votes": votes}
                checkpoint.write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8")
        labels = [value["label"] for value in votes.values() if value.get("label")]
        counts = Counter(labels)
        majority = counts.most_common(1)[0] if counts else (None, 0)
        baseline_box = baseline[sample_id].get("bbox") if isinstance(baseline[sample_id], dict) else baseline[sample_id]
        baseline_index = max(range(len(candidates)), key=lambda i: compute_iou(candidates[i]["bbox"], baseline_box))
        choices[sample_id].update({
            "majority_label": majority[0] if majority[1] >= 2 else None,
            "agreement": majority[1], "baseline_label": LABELS[baseline_index],
        })
        checkpoint.write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"completed={len(choices)} output={checkpoint}")


if __name__ == "__main__":
    main()
