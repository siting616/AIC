"""Select competition queries with at least two non-reference detector prompts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.query_prompts import generate_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompts", type=int, default=4)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    annotations = json.loads(resolve(args.annotations).read_text(encoding="utf-8"))
    selected = []
    counts = Counter()
    for sample_id, item in annotations.items():
        prompts = generate_prompts(item.get("query", ""), args.max_prompts)
        detector = [prompt for prompt in prompts if prompt["type"] != "reference"]
        counts[len(detector)] += 1
        if len(detector) >= 2:
            selected.append({"query_id": sample_id, "query": item["query"], "prompts": prompts})
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"samples={len(annotations)} selected={len(selected)}")
    print(f"detector_prompt_distribution={dict(sorted(counts.items()))}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
