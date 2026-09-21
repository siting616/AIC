"""Classify official AIC query text without modifying competition data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


CATEGORIES = {
    "horizontal_position": ("left", "right", "leftmost", "rightmost"),
    "vertical_position": ("top", "bottom", "upper", "lower", "above", "below"),
    "depth_relation": (
        "near", "nearest", "close", "closest", "far", "farthest",
        "front", "behind", "foreground", "background",
    ),
    "person": (
        "person", "people", "man", "men", "woman", "women", "boy",
        "boys", "girl", "girls", "child", "children", "pedestrian",
    ),
    "animal": (
        "dog", "cat", "bird", "horse", "cow", "sheep", "monkey",
        "elephant", "giraffe", "zebra", "animal",
    ),
    "color": (
        "red", "orange", "yellow", "green", "blue", "purple", "pink",
        "brown", "black", "white", "gray", "grey",
    ),
    "action_pose": (
        "standing", "sitting", "walking", "running", "riding", "holding",
        "wearing", "lying", "crouching", "bending", "looking",
    ),
    "count_quantity": (
        "one", "two", "three", "four", "five", "single", "pair",
        "group", "several", "multiple",
    ),
    "comparison": (
        "largest", "smallest", "tallest", "shortest", "biggest",
        "closest", "nearest", "farthest", "leftmost", "rightmost",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze AIC query categories.")
    parser.add_argument(
        "--input",
        default="初赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json",
    )
    parser.add_argument("--output-dir", default="outputs/preliminary/query_analysis")
    return parser.parse_args()


def has_term(text: str, term: str) -> bool:
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    category_counts = Counter()
    term_counts = Counter()
    category_examples = {name: [] for name in CATEGORIES}
    per_query = {}
    uncategorized = []

    for query_id, item in data.items():
        query = item["query"]
        normalized = query.lower()
        tags = []
        matched_terms = []
        for category, terms in CATEGORIES.items():
            hits = [term for term in terms if has_term(normalized, term)]
            if hits:
                tags.append(category)
                matched_terms.extend(hits)
                category_counts[category] += 1
                term_counts.update(hits)
                if len(category_examples[category]) < 5:
                    category_examples[category].append(
                        {"query_id": query_id, "query": query, "terms": hits}
                    )
        if not tags:
            uncategorized.append({"query_id": query_id, "query": query})
        per_query[query_id] = {
            "query": query,
            "tags": tags,
            "matched_terms": sorted(set(matched_terms)),
        }

    summary = {
        "total_queries": len(data),
        "category_counts": dict(category_counts.most_common()),
        "top_terms": dict(term_counts.most_common(50)),
        "uncategorized_count": len(uncategorized),
        "category_examples": category_examples,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "query_tags.json").write_text(
        json.dumps(per_query, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "uncategorized_examples.json").write_text(
        json.dumps(uncategorized[:200], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# AIC 初赛 Query 自动分类报告",
        "",
        f"- Query 总数：{len(data)}",
        f"- 未匹配预设类别：{len(uncategorized)}",
        "",
        "## 类别统计",
        "",
        "| 类别 | Query 数量 | 占比 |",
        "|---|---:|---:|",
    ]
    for category, count in category_counts.most_common():
        report_lines.append(f"| {category} | {count} | {count / len(data):.2%} |")
    report_lines.extend(["", "## 高频关键词", ""])
    for term, count in term_counts.most_common(30):
        report_lines.append(f"- `{term}`：{count}")
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(f"Queries analyzed: {len(data)}")
    for category, count in category_counts.most_common():
        print(f"{category}: {count} ({count / len(data):.2%})")
    print(f"Uncategorized: {len(uncategorized)}")
    print(f"Report: {output_dir / 'report.md'}")
    print("QUERY ANALYSIS PASSED")


if __name__ == "__main__":
    main()
