"""Use local Qwen2.5-VL to select among saved GroundingDINO candidates.

The experiment is deliberately hybrid: only a deterministic, stratified subset
is reviewed by the VLM and every other query retains the supplied baseline box.
Checkpoints make the expensive inference resumable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import AICDataset
from src.postprocess import sanitize_bbox
from src.submit import generate_submission, save_predictions


CATEGORIES = OrderedDict(
    [
        ("relation", re.compile(r"\b(left of|right of|next to|near|closest to|behind|in front of|between|under|above)\b", re.I)),
        ("color", re.compile(r"\b(red|blue|green|yellow|white|black|pink|purple|brown|gray|grey|orange)\b", re.I)),
        ("action", re.compile(r"\b(sitting|standing|walking|holding|wearing|eating|riding|flying|looking|squatting)\b", re.I)),
        ("ordinal", re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b", re.I)),
        ("body_part", re.compile(r"\b(eye|ear|leg|foot|hand|head|tail|arm|pants|shirt|cap|cover)\b", re.I)),
    ]
)
DEFAULT_QUOTAS = {"relation": 150, "color": 100, "action": 100, "ordinal": 75, "body_part": 75}
LABELS = "ABCDEFGHIJKLMNOPQRST"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preliminary.yaml")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/v25_qwen500")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument(
        "--selection-mode",
        choices=("stratified", "complex_all"),
        default="stratified",
        help="Use the 500-query quota sample or every query matching a complex category.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_subset(dataset, saved, limit):
    samples = {dataset[i]["sample_id"]: dataset[i] for i in range(len(dataset))}
    chosen, used = [], set()
    for category, pattern in CATEGORIES.items():
        quota = DEFAULT_QUOTAS[category]
        for query_id in dataset.sample_ids:
            if len([x for x in chosen if x[1] == category]) >= quota:
                break
            if query_id in used or not (saved.get(query_id, {}).get("candidates") or []):
                continue
            if pattern.search(samples[query_id]["query"]):
                chosen.append((query_id, category))
                used.add(query_id)
    if len(chosen) < limit:
        for query_id in dataset.sample_ids:
            if query_id not in used and (saved.get(query_id, {}).get("candidates") or []):
                chosen.append((query_id, "other"))
                used.add(query_id)
                if len(chosen) >= limit:
                    break
    return chosen[:limit], samples


def choose_complex_subset(dataset, saved, limit):
    samples = {dataset[i]["sample_id"]: dataset[i] for i in range(len(dataset))}
    chosen = []
    for query_id in dataset.sample_ids:
        if not (saved.get(query_id, {}).get("candidates") or []):
            continue
        query = samples[query_id]["query"]
        categories = [name for name, pattern in CATEGORIES.items() if pattern.search(query)]
        if categories:
            chosen.append((query_id, "+".join(categories)))
            if limit > 0 and len(chosen) >= limit:
                break
    return chosen, samples


def annotate(image_path, candidates):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    for index, candidate in enumerate(candidates[: len(LABELS)]):
        x1, y1, x2, y2 = candidate["bbox"]
        box = [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]
        color = (255, 32, 32)
        line_width = max(3, round(min(width, height) / 250))
        draw.rectangle(box, outline=color, width=line_width)
        label = LABELS[index]
        tx, ty = box[0], max(0, box[1] - 26)
        draw.rectangle([tx, ty, tx + 24, ty + 24], fill=color)
        draw.text((tx + 7, ty + 5), label, fill="white", font=font)
    return image


def load_model(model_id, use_4bit):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    return model, processor


def select_label(model, processor, image, query, candidate_count, max_new_tokens):
    from qwen_vl_utils import process_vision_info

    valid = ", ".join(LABELS[:candidate_count])
    prompt = (
        "The image contains red candidate boxes labeled with letters. "
        f"Referring expression: {query!r}. "
        f"Choose the single box that the expression refers to. Valid answers: {valid}. "
        "Reply with exactly one letter and no explanation."
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().upper()
    match = re.search(r"\b([A-T])\b", answer)
    return (match.group(1) if match and match.group(1) in LABELS[:candidate_count] else None), answer


def main():
    args = parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    dataset = AICDataset(str(PROJECT_ROOT / cfg["data_dir"]), str(PROJECT_ROOT / cfg["json_file"]))
    saved = json.loads(resolve(args.candidates).read_text(encoding="utf-8"))
    baseline = json.loads(resolve(args.baseline).read_text(encoding="utf-8"))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "qwen_choices_checkpoint.json"
    choices = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    if args.selection_mode == "complex_all":
        subset, samples = choose_complex_subset(dataset, saved, args.max_samples)
    else:
        subset, samples = choose_subset(dataset, saved, args.max_samples)
    (output_dir / "subset.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    model, processor = load_model(args.model_id, not args.no_4bit)

    for query_id, category in tqdm(subset, desc="Qwen candidate selection"):
        if query_id in choices:
            continue
        sample = samples[query_id]
        candidates = saved[query_id].get("candidates", [])[: len(LABELS)]
        image = annotate(sample["visible_path"], candidates)
        label, raw = select_label(model, processor, image, sample["query"], len(candidates), args.max_new_tokens)
        choices[query_id] = {"label": label, "raw": raw, "category": category}
        checkpoint_path.write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8")

    predictions = {query_id: sanitize_bbox(box) for query_id, box in baseline.items()}
    changed = 0
    for query_id, record in choices.items():
        label = record.get("label")
        if not label:
            continue
        index = LABELS.index(label)
        candidates = saved[query_id].get("candidates", [])
        if index < len(candidates):
            box = sanitize_bbox(candidates[index]["bbox"])
            if box != predictions[query_id]:
                changed += 1
            predictions[query_id] = box
    save_predictions(predictions, str(output_dir / "reranked_predictions.json"))
    original = str(PROJECT_ROOT / cfg["json_file"])
    generate_submission(original, predictions, str(output_dir / "prediction.json"), str(output_dir / "submission_v25_qwen500.zip"))
    print(f"Subset: {len(subset)}; parsed choices: {sum(bool(x.get('label')) for x in choices.values())}; changed: {changed}")
    print(f"Submission: {output_dir / 'submission_v25_qwen500.zip'}")


if __name__ == "__main__":
    main()
