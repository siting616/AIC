"""Conservatively compare the frozen baseline box with one rescue candidate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qwen_candidate_selector import load_model
from src.dataset import AICDataset
from src.postprocess import sanitize_bbox
from src.submit import generate_submission, save_predictions


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union else 0.0


def annotate_pair(image_path, first, second):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    for label, bbox, color in (("A", first, (255, 32, 32)), ("B", second, (32, 128, 255))):
        x1, y1, x2, y2 = bbox
        box = [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]
        line_width = max(3, round(min(width, height) / 250))
        draw.rectangle(box, outline=color, width=line_width)
        tx, ty = box[0], max(0, box[1] - 26)
        draw.rectangle([tx, ty, tx + 24, ty + 24], fill=color)
        draw.text((tx + 7, ty + 5), label, fill="white", font=font)
    return image


def select_pair(model, processor, image, query, verification=False):
    from qwen_vl_utils import process_vision_info

    if verification:
        instruction = (
            "Independently verify the two boxes. Check object identity, attributes, "
            "ordinal position, and spatial relations before choosing."
        )
    else:
        instruction = (
            "Compare the two boxes carefully. Check the complete expression, not just "
            "the object category."
        )
    prompt = (
        f"{instruction} Referring expression: {query!r}. "
        "Which box is the intended target, A or B? Reply with exactly A or B."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().upper()
    match = re.search(r"\b([AB])\b", raw)
    return (match.group(1) if match else None), raw


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--original-candidates", required=True)
    parser.add_argument("--rescue-candidates", required=True)
    parser.add_argument("--rescue-choices", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    dataset = AICDataset(str(PROJECT_ROOT / cfg["data_dir"]), str(PROJECT_ROOT / cfg["json_file"]))
    samples = {dataset[i]["sample_id"]: dataset[i] for i in range(len(dataset))}
    original = json.loads(resolve(args.original_candidates).read_text(encoding="utf-8"))
    rescue = json.loads(resolve(args.rescue_candidates).read_text(encoding="utf-8"))
    prior_choices = json.loads(resolve(args.rescue_choices).read_text(encoding="utf-8"))
    baseline = json.loads(resolve(args.baseline).read_text(encoding="utf-8"))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "pairwise_checkpoint.json"
    decisions = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}

    eligible = []
    labels = "ABCDE"
    for query_id, choice in prior_choices.items():
        label = choice.get("label")
        if not isinstance(label, str) or label not in labels:
            continue
        candidates = rescue.get(query_id, {}).get("candidates", [])
        index = labels.index(label)
        if index >= len(candidates):
            continue
        selected = candidates[index]
        old_record = original[query_id]
        old_candidates = old_record.get("candidates", [])[:5]
        old_boxes = [item["bbox"] for item in old_candidates]
        old_scores = [float(item.get("score", 0.0)) for item in old_candidates]
        is_new = not any(bbox_iou(selected["bbox"], box) >= 0.85 for box in old_boxes)
        old_weak = (
            bool(old_record.get("used_fallback"))
            or old_record.get("selection_mode") == "rescue_threshold"
            or len(old_boxes) < 3
            or max(old_scores, default=0.0) < 0.25
        )
        box = selected["bbox"]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if (
            is_new and old_weak
            and selected.get("prompt_type") in {"target_phrase", "head_noun"}
            and selected.get("agreement_count", 1) >= 2
            and float(selected.get("model_score", 0.0)) >= 0.15
            and 0.002 <= area <= 0.70
        ):
            eligible.append((query_id, selected))

    (output_dir / "eligible.json").write_text(
        json.dumps([{"query_id": q, "candidate": c} for q, c in eligible], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model, processor = load_model(args.model_id, not args.no_4bit)
    for query_id, candidate in tqdm(eligible, desc="Pairwise rescue"):
        if query_id in decisions:
            continue
        sample = samples[query_id]
        baseline_box = sanitize_bbox(baseline[query_id])
        rescue_box = sanitize_bbox(candidate["bbox"])
        image = annotate_pair(sample["visible_path"], baseline_box, rescue_box)
        first, raw_first = select_pair(model, processor, image, sample["query"], verification=False)
        second, raw_second = select_pair(model, processor, image, sample["query"], verification=True)
        accept = first == "B" and second == "B"
        decisions[query_id] = {
            "first": first, "second": second,
            "raw_first": raw_first, "raw_second": raw_second,
            "accept_rescue": accept,
            "candidate": candidate,
        }
        checkpoint_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")

    predictions = {query_id: sanitize_bbox(box) for query_id, box in baseline.items()}
    accepted = 0
    for query_id, decision in decisions.items():
        if decision.get("accept_rescue"):
            predictions[query_id] = sanitize_bbox(decision["candidate"]["bbox"])
            accepted += 1
    save_predictions(predictions, str(output_dir / "reranked_predictions.json"))
    generate_submission(
        str(PROJECT_ROOT / cfg["json_file"]), predictions,
        str(output_dir / "prediction.json"), str(output_dir / "submission_pairwise_rescue.zip"),
    )
    print(f"Eligible: {len(eligible)}; decisions: {len(decisions)}; accepted rescues: {accepted}")


if __name__ == "__main__":
    main()
