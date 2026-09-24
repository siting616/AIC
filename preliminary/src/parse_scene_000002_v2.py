import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

PROJECT_ROOT = Path("/root/autodl-tmp/aic_v1_project")

DATA_ROOT = (
    PROJECT_ROOT
    / "data/official/初赛数据集-基于大模型的多模态视觉理解与推理"
)

QUERY_JSON = DATA_ROOT / "queries/queries.json"
OUTPUT = PROJECT_ROOT / "outputs/stage1_000002/parsed_queries_v2.json"

HF_HUB = PROJECT_ROOT / "models/hf_cache/hub"


def find_snapshot(repo):
    root = HF_HUB / repo / "snapshots"
    for p in root.iterdir():
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
    raise RuntimeError(repo)


QWEN_PATH = find_snapshot(
    "models--Qwen--Qwen3-VL-8B-Instruct"
)

processor = AutoProcessor.from_pretrained(
    QWEN_PATH,
    local_files_only=True,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

model.eval()


def extract_json(text):
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(text)


def parse_query(query):

    prompt = f"""
You are a parser for a multimodal visual grounding system.

Parse the referring expression into TARGET information and
REFERENCE-OBJECT spatial constraints.

Grounding DINO will be used only to DETECT visible nouns.
Spatial reasoning will happen later.

Rules:

1. target_dino_prompts must ONLY describe the final target.
2. NEVER put a reference object into target_dino_prompts.
3. NEVER use phrases such as:
   "below black awning",
   "mounted on pillar",
   "beside lamp"
   as target detector prompts.
4. For hard semantic targets, provide useful visual synonyms.
5. Preserve explicit counts such as one/two/three.
6. A query can contain MULTIPLE spatial constraints.
7. Ordinals such as leftmost/rightmost are evaluated AFTER
   spatial constraints.
8. reference_dino_prompts should contain short visible noun phrases.

Return ONLY JSON:

{{
  "target_class": "",
  "target_attributes": [],
  "count": 1,
  "ordinal": "none",
  "target_dino_prompts": [],
  "constraints": [
    {{
      "relation": "",
      "reference_object": "",
      "reference_dino_prompts": []
    }}
  ]
}}

If there is no relation, constraints must be [].

Examples:

Query:
"The leftmost security camera mounted on the stone pillar,
located directly below the black awning."

Correct structure:
{{
  "target_class": "security camera",
  "target_attributes": [],
  "count": 1,
  "ordinal": "leftmost",
  "target_dino_prompts": [
    "security camera",
    "surveillance camera",
    "camera"
  ],
  "constraints": [
    {{
      "relation": "on",
      "reference_object": "stone pillar",
      "reference_dino_prompts": [
        "stone pillar",
        "pillar"
      ]
    }},
    {{
      "relation": "below",
      "reference_object": "black awning",
      "reference_dino_prompts": [
        "black awning",
        "awning"
      ]
    }}
  ]
}}

Query:
{query}
""".strip()

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": prompt
        }]
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        k: v.to(model.device)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    n = inputs["input_ids"].shape[1]

    answer = processor.batch_decode(
        out[:, n:],
        skip_special_tokens=True,
    )[0]

    return extract_json(answer), answer


with open(QUERY_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

result = {}

for qid in sorted(data):

    if not qid.startswith("000002_"):
        continue

    query = data[qid]["query"]

    print("\n" + "=" * 70)
    print(qid)
    print(query)

    try:
        parsed, raw = parse_query(query)
    except Exception as e:
        print("ERROR:", e)
        continue

    result[qid] = {
        "query": query,
        "parsed": parsed,
        "raw": raw,
    }

    print(
        json.dumps(
            parsed,
            ensure_ascii=False,
            indent=2,
        )
    )


with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\nSaved:", OUTPUT)
