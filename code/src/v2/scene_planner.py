"""Build scene-level plans while preserving every frozen baseline answer."""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from .query_parser_v2 import parse_query
from .query_taxonomy import classify_query, operator_chain


def scene_id(sample: dict[str, Any], query_id: str) -> str:
    visible = str(sample.get("visible", ""))
    if visible:
        name = PureWindowsPath(visible).stem or PurePosixPath(visible).stem
        if name:
            return name
    return str(query_id).split("_", 1)[0]


def build_scene_plan(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_counts: dict[str, int] = defaultdict(int)
    for query_id, sample in records.items():
        parsed = parse_query(sample.get("query", ""), query_id=query_id)
        tags = classify_query(parsed)
        for tag in tags:
            route_counts[tag] += 1
        scenes[scene_id(sample, query_id)].append({
            "query_id": query_id,
            "query": sample.get("query", ""),
            "tags": tags,
            "operator_chain": operator_chain(parsed),
            "parsed": parsed,
            "override_allowed": False,
        })
    return {
        "schema_version": 1,
        "policy": "plan-only; all bboxes remain frozen baseline",
        "sample_count": len(records),
        "scene_count": len(scenes),
        "route_counts": dict(sorted(route_counts.items())),
        "scenes": dict(sorted(scenes.items())),
    }

