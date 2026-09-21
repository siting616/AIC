"""Multi-label taxonomy for V2 operator routing."""

from __future__ import annotations

from typing import Any


def classify_query(parsed: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if parsed["ordinal"]["enabled"]:
        tags.append("ordinal")
    if parsed["depth"]["enabled"]:
        tags.append("depth")
    if parsed["target"]["part"]:
        tags.append("part")
    if parsed["ocr"]["enabled"]:
        tags.append("ocr")
    if parsed["thermal"]["enabled"]:
        tags.append("thermal")
    return tags or ["ordinary"]


def operator_chain(parsed: dict[str, Any]) -> list[str]:
    """Order operators by semantic scope, leaving judge/refine for later phases."""
    tags = set(classify_query(parsed))
    chain = ["baseline_candidates"]
    if "depth" in tags:
        chain.append("depth_filter")
    if "ordinal" in tags:
        chain.append("ordinal_select")
    if "part" in tags:
        chain.extend(["parent_select", "part_localize"])
    if "ocr" in tags:
        chain.extend(["carrier_select", "text_read"])
    if "thermal" in tags:
        chain.append("thermal_scoped_compare")
    chain.append("conservative_override")
    return chain

