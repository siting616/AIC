"""Double-confirm A/B Qwen evaluation of baseline versus tiled candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qwen_candidate_selector import load_model
from src.metrics import compute_iou


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def annotate(path, box_a, box_b):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for label, box, color in (("A", box_a, (255, 32, 32)), ("B", box_b, (32, 128, 255))):
        xy = [round(box[0]*width), round(box[1]*height), round(box[2]*width), round(box[3]*height)]
        line = max(3, round(min(width, height)/220))
        draw.rectangle(xy, outline=color, width=line)
        draw.rectangle([xy[0], max(0, xy[1]-28), xy[0]+28, xy[1]], fill=color)
        draw.text((xy[0]+8, max(1, xy[1]-24)), label, fill="white", font=ImageFont.load_default())
    return image


def select(model, processor, image, query, verification):
    from qwen_vl_utils import process_vision_info
    instruction = (
        "Independently verify object identity, attributes, ordinal position and spatial relations. "
        if verification else "Compare the complete referring expression carefully. "
    )
    prompt = instruction + f"Expression: {query!r}. Which target box is correct, A or B? Reply exactly A or B."
    messages = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().upper()
    match = re.search(r"\b([AB])\b", raw)
    return (match.group(1) if match else None), raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/validation/refcoco_val.json")
    parser.add_argument("--baseline", default="outputs/experiments/refcoco_qwen3b_complex_all/reranked_predictions.json")
    parser.add_argument("--tiled-selected", default="outputs/experiments/tiled_position_rerank/selected_candidates.json")
    parser.add_argument("--metadata", default="outputs/experiments/tiled_difficult/refcoco_difficult.json")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/qwen_tiled_pairwise500")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Start offset in the fixed eligible list, for an independent holdout batch.",
    )
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    annotations, baseline = load(resolve(args.annotations)), load(resolve(args.baseline))
    tiled, metadata = load(resolve(args.tiled_selected)), load(resolve(args.metadata))
    metadata = {x["query_id"]: x for x in metadata}
    images_dir = resolve(args.images_dir)
    eligible = []
    for qid in sorted(tiled):
        if qid not in baseline or qid not in annotations or qid not in metadata:
            continue
        box_a, box_b = baseline[qid], tiled[qid]["bbox"]
        if compute_iou(box_a, box_b) >= .85:
            continue
        eligible.append(qid)
    # Fixed deterministic sample; no GT field is consulted here.
    subset = eligible[args.start_index:args.start_index + args.max_samples]
    output = resolve(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    save(output/"subset.json", subset)
    checkpoint = output/"checkpoint.json"
    decisions = load(checkpoint) if checkpoint.exists() else {}
    model, processor = load_model(args.model_id, not args.no_4bit)
    for qid in tqdm(subset, desc="Qwen tiled A/B", unit="query"):
        if qid in decisions: continue
        raw_visible = str(metadata[qid]["visible"])
        # RefCOCO metadata was prepared on Windows.  On Linux, Path treats
        # backslashes as ordinary characters, so explicitly parse Windows paths.
        name = PureWindowsPath(raw_visible).name if "\\" in raw_visible else Path(raw_visible).name
        path = images_dir/name
        image = annotate(path, baseline[qid], tiled[qid]["bbox"])
        first, raw1 = select(model, processor, image, annotations[qid]["query"], False)
        second, raw2 = select(model, processor, image, annotations[qid]["query"], True)
        decisions[qid] = {"first":first,"second":second,"raw_first":raw1,"raw_second":raw2,
                          "accept_b":first=="B" and second=="B"}
        save(checkpoint, decisions)
    before = after = new_correct = new_wrong = accepted = 0
    rows = {}
    for qid in subset:
        gt = annotations[qid]["bbox"]; a=baseline[qid]; b=tiled[qid]["bbox"]
        use_b = decisions.get(qid,{}).get("accept_b",False)
        chosen = b if use_b else a
        ai, ci = compute_iou(a,gt), compute_iou(chosen,gt)
        before += ai>=.5; after += ci>=.5; accepted += use_b
        new_correct += ai<.5<=ci; new_wrong += ci<.5<=ai
        rows[qid]={"before_iou":ai,"after_iou":ci,"accept_b":use_b,"decision":decisions.get(qid)}
    metrics={"samples":len(subset),"accepted_b":accepted,"correct_before":before,"correct_after":after,
             "new_correct":new_correct,"new_wrong":new_wrong,"net_correct":new_correct-new_wrong,
             "delta_pp":100*(after-before)/len(subset) if subset else 0,
             "gate_pass":new_correct-new_wrong>=5}
    save(output/"records.json",rows); save(output/"metrics.json",metrics)
    report=("# Qwen Tiled Pairwise 500\n\n"+"\n".join(f"- {k}: {v}" for k,v in metrics.items())+"\n")
    (output/"report.md").write_text(report,encoding="utf-8")
    print(report)


if __name__ == "__main__": main()
