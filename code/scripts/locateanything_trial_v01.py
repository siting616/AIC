#!/usr/bin/env python3
"""Small, isolated LocateAnything-3B trial on the uploaded contest data."""

import argparse
import json
from pathlib import Path
import re
import time

from PIL import Image, ImageDraw
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer


DEFAULT_QIDS = [
    "000001_001",  # person
    "000001_002",  # long spatial description
    "000001_003",  # ordinal
    "000001_006",  # group of three
    "000015_001",  # closest to camera
    "000036_001",  # simple object
    "000463_004",  # thermal cue
    "001051_003",  # thermal and relational cue
]
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_boxes(answer):
    boxes = []
    for match in BOX_RE.finditer(answer):
        box = [min(1000, max(0, int(value))) / 1000 for value in match.groups()]
        if box[0] < box[2] and box[1] < box[3]:
            boxes.append(box)
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/aic_multimodal_dataset"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/aic_v1_project/models/locateanything_3b"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/aic_v1_project/outputs/locany_trial"))
    parser.add_argument("--qid", action="append", help="Query ID; repeat for more queries")
    parser.add_argument("--query-override", help="alternate wording for a single --qid")
    parser.add_argument("--sample-count", type=int, default=0, help="uniform sample across all query IDs")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--mode", choices=["hybrid", "fast", "slow"], default="hybrid")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-side", type=int, default=1024, help="resize longest image side; 0 keeps original size")
    args = parser.parse_args()

    data_root = args.data_root.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    queries_path = data_root / "queries" / "queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if args.qid and args.sample_count:
        parser.error("use either --qid or --sample-count")
    if args.query_override and (not args.qid or len(args.qid) != 1):
        parser.error("--query-override requires exactly one --qid")
    if args.sample_count:
        all_qids = list(queries)
        if not 1 <= args.sample_count <= len(all_qids):
            parser.error("sample count outside query range")
        qids = [all_qids[index * len(all_qids) // args.sample_count] for index in range(args.sample_count)]
    else:
        qids = args.qid or DEFAULT_QIDS
    for qid in qids:
        if qid not in queries:
            parser.error(f"Unknown query ID: {qid}")

    print(f"Loading model from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to("cuda").eval()
    print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)

    results = {}
    for qid in qids:
        record = queries[qid]
        image_path = data_root / record["visible"]
        query = args.query_override or record["query"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        original_size = image.size
        if args.max_side > 0 and max(image.size) > args.max_side:
            image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
        prompt = f"Locate a single instance that matches the following description: {query}."
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = processor.process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        try:
            with torch.inference_mode():
                response = model.generate(
                    pixel_values=pixel_values,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws"),
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    generation_mode=args.mode,
                    do_sample=False,
                    verbose=False,
                )
            torch.cuda.synchronize()
            answer = response[0] if isinstance(response, tuple) else response
            if not isinstance(answer, str):
                answer = str(answer)
            boxes = parse_boxes(answer)
            result = {
                "qid": qid,
                "query": query,
                "visible": record["visible"],
                "original_size": original_size,
                "input_size": image.size,
                "answer": answer,
                "boxes_normalized": boxes,
                "box_count": len(boxes),
                "seconds": round(time.monotonic() - started, 3),
                "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
            }
            if not args.no_overlays:
                draw = ImageDraw.Draw(image)
                width, height = image.size
                for index, (x1, y1, x2, y2) in enumerate(boxes[:10], 1):
                    rectangle = [x1 * width, y1 * height, x2 * width, y2 * height]
                    draw.rectangle(rectangle, outline="red", width=5)
                    draw.text((rectangle[0], rectangle[1]), str(index), fill="yellow")
                image.save(args.output / f"{qid}.jpg", quality=90)
            print(json.dumps({
                "qid": qid,
                "box_count": len(boxes),
                "first_boxes": boxes[:4],
                "answer_preview": answer[:200],
                "seconds": result["seconds"],
                "peak_vram_gb": result["peak_vram_gb"],
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            result = {"qid": qid, "query": query, "error": repr(exc)}
            print(json.dumps(result, ensure_ascii=False), flush=True)
        results[qid] = result
        (args.output / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not any("boxes_normalized" in result for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
