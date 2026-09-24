#!/usr/bin/env python3
"""Resumable LocateAnything-3B inference and contest submission builder."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import zipfile

from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer


BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
PROMPT = "Locate a single instance that matches the following description: {}."


def parse_boxes(answer):
    boxes = []
    for match in BOX_RE.finditer(answer):
        box = [min(1000, max(0, int(value))) / 1000 for value in match.groups()]
        if box[0] < box[2] and box[1] < box[3]:
            boxes.append(box)
    return boxes


def load_progress(path):
    if not path.exists():
        return {}
    # A killed process can leave one incomplete final line. Preserve all complete records.
    with path.open("rb+") as stream:
        content = stream.read()
        end = content.rfind(b"\n") + 1
        if end < len(content):
            stream.truncate(end)
        content = content[:end]
    records = {}
    for line in content.splitlines():
        item = json.loads(line)
        records[item["qid"]] = item
    return records


def valid_box(box):
    return (
        isinstance(box, list) and len(box) == 4
        and all(isinstance(value, (float, int)) and math.isfinite(value) for value in box)
        and 0 <= box[0] < box[2] <= 1
        and 0 <= box[1] < box[3] <= 1
    )


def make_submission(queries, predictions, output):
    missing = [qid for qid in queries if qid not in predictions or not valid_box(predictions[qid].get("bbox"))]
    if missing:
        print(f"Submission deferred: {len(missing)} missing/invalid boxes; first IDs: {missing[:10]}", flush=True)
        return False
    submission = {qid: {**entry, "bbox": predictions[qid]["bbox"]} for qid, entry in queries.items()}
    if set(submission) != set(queries):
        raise RuntimeError("Query IDs changed while building submission")
    json_bytes = json.dumps(submission, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_tmp = output / "submission.json.tmp"
    json_tmp.write_bytes(json_bytes)
    os.replace(json_tmp, output / "submission.json")
    zip_tmp = output / "submission_locany_1792.zip.tmp"
    with zipfile.ZipFile(zip_tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("submission.json", json_bytes)
    os.replace(zip_tmp, output / "submission_locany_1792.zip")
    print(f"Submission ready: {len(submission)} queries, {output / 'submission_locany_1792.zip'}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/aic_multimodal_dataset"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/aic_v1_project/models/locateanything_3b"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/aic_v1_project/outputs/locany_full_1792"))
    parser.add_argument("--max-side", type=int, default=1792)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="process only the first N queries for a smoke test")
    args = parser.parse_args()
    data_root = args.data_root.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)

    queries_path = data_root / "queries" / "queries.json"
    queries_bytes = queries_path.read_bytes()
    queries = json.loads(queries_bytes)
    qids = list(queries)[: args.limit] if args.limit else list(queries)
    config = {
        "queries_sha256": hashlib.sha256(queries_bytes).hexdigest(),
        "model": str(model_path),
        "max_side": args.max_side,
        "max_new_tokens": args.max_new_tokens,
        "prompt": PROMPT,
    }
    config_path = args.output / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError(f"Run configuration changed: {config_path}")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    progress_path = args.output / "predictions.jsonl"
    predictions = load_progress(progress_path)
    unexpected = set(predictions) - set(queries)
    if unexpected:
        raise RuntimeError(f"Unknown Query IDs in progress: {sorted(unexpected)[:10]}")
    pending = [qid for qid in qids if qid not in predictions or not valid_box(predictions[qid].get("bbox"))]
    print(f"Queries: {len(qids)}; already complete: {len(qids) - len(pending)}; pending: {len(pending)}", flush=True)
    if not pending:
        if not args.limit:
            make_submission(queries, predictions, args.output)
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to("cuda").eval()
    print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)

    last_image_path = None
    original_image = None
    started = time.monotonic()
    completed_this_run = 0
    errors_this_run = 0
    with progress_path.open("a", encoding="utf-8") as stream:
        for qid in pending:
            entry = queries[qid]
            image_path = data_root / entry["visible"]
            if image_path != last_image_path:
                with Image.open(image_path) as source:
                    original_image = source.convert("RGB")
                last_image_path = image_path
            exceptions = []
            outcome = None
            for max_side in dict.fromkeys((args.max_side, 1536, 1024)):
                image = original_image.copy()
                if max_side > 0 and max(image.size) > max_side:
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT.format(entry["query"])},
                ]}]
                query_started = time.monotonic()
                try:
                    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images, videos = processor.process_vision_info(messages)
                    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        response = model.generate(
                            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            image_grid_hws=inputs.get("image_grid_hws"),
                            tokenizer=tokenizer,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            generation_mode="hybrid",
                            do_sample=False,
                            verbose=False,
                        )
                    torch.cuda.synchronize()
                    answer = response[0] if isinstance(response, tuple) else response
                    if not isinstance(answer, str):
                        answer = str(answer)
                    boxes = parse_boxes(answer)
                    if not boxes:
                        raise ValueError("model produced no valid box")
                    outcome = {
                        "qid": qid,
                        "bbox": boxes[0],
                        "box_count": len(boxes),
                        "boxes_normalized": boxes[:20],
                        "raw_answer": answer,
                        "input_size": image.size,
                        "seconds": round(time.monotonic() - query_started, 3),
                        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
                        "retry_errors": exceptions,
                    }
                    break
                except Exception as exc:
                    exceptions.append({"max_side": max_side, "error": repr(exc)})
                    print(f"RETRY {qid} at {max_side}: {exc}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                finally:
                    if "inputs" in locals():
                        del inputs
            if outcome is None:
                errors_this_run += 1
                outcome = {"qid": qid, "error": exceptions, "bbox": None}
            else:
                completed_this_run += 1
                if outcome["box_count"] != 1:
                    print(f"MULTI {qid}: {outcome['box_count']} boxes; first selected", flush=True)
            stream.write(json.dumps(outcome, ensure_ascii=False) + "\n")
            stream.flush()
            predictions[qid] = outcome
            processed = completed_this_run + errors_this_run
            if processed % 25 == 0 or processed == len(pending):
                os.fsync(stream.fileno())
                elapsed = time.monotonic() - started
                rate = processed / max(elapsed, 0.001)
                remaining = (len(pending) - processed) / max(rate, 1e-6)
                print(
                    f"PROGRESS {len(qids) - len(pending) + processed}/{len(qids)} "
                    f"errors={errors_this_run} elapsed={elapsed / 60:.1f}m "
                    f"eta={remaining / 60:.1f}m rate={rate:.3f} q/s",
                    flush=True,
                )

    if not args.limit:
        make_submission(queries, predictions, args.output)
    if errors_this_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
