"""Deterministic, dependency-free parser for V2 routing and planning.

This module does not pretend to replace a VLM parser.  It extracts only
high-precision operators that are safe enough to decide which specialist may
inspect a query.  Uncertain semantics remain ordinary and keep the baseline.
"""

from __future__ import annotations

import re
from typing import Any


_NUMBER_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_PARTS = (
    "ear", "wheel", "license plate", "tail", "horn", "door handle",
    "handle", "shoe", "head", "face", "arm", "leg", "hand", "foot",
)
_CARRIERS = ("license plate", "sign", "screen", "label", "box", "banner", "board")
_THERMAL_RE = re.compile(
    r"\b(?:warm|warmer|warmest|hot|hottest|heat|thermal|infrared|cold|cooler)\b"
    r"|\bheat\s+signature\b|\bthermal\s+signature\b",
    re.I,
)
_DEPTH_RE = re.compile(
    r"\b(?:closest|nearest|farthest|furthest|closer|farther|in front of|behind)\b",
    re.I,
)


def _ordinal(text: str) -> dict[str, Any]:
    lowered = text.lower()
    # Ordinals in a reference clause ("the pier behind ... the fourth pier")
    # must not route the target into an ordinal family.
    relation = re.search(
        r"\b(?:behind|in front of|left of|right of|next to|beside|near|under|above|on)\b",
        lowered,
    )
    target_clause = lowered[: relation.start()] if relation else lowered
    direction = None
    if re.search(r"\b(?:from\s+)?left\s+to\s+right\b|\bfrom\s+(?:the\s+)?left\b", target_clause):
        direction = "left_to_right"
    elif re.search(r"\b(?:from\s+)?right\s+to\s+left\b|\bfrom\s+(?:the\s+)?right\b", target_clause):
        direction = "right_to_left"

    index = None
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", target_clause):
            index = value
            break
    if index is None:
        match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", target_clause)
        if match:
            index = int(match.group(1))
    if re.search(r"\bleftmost\b|\bfar\s+left\b", target_clause):
        index, direction = 1, "left_to_right"
    elif re.search(r"\brightmost\b|\bfar\s+right\b", target_clause):
        index, direction = 1, "right_to_left"

    enabled = index is not None and direction is not None
    return {
        "enabled": enabled,
        "type": "spatial" if enabled else None,
        "axis": "x" if enabled else None,
        "direction": direction if enabled else None,
        "index": index if enabled else None,
    }


def _quoted_text(text: str) -> str | None:
    match = re.search(r'["\u201c\u201d\']([^"\u201c\u201d\']+)["\u201c\u201d\']', text)
    return match.group(1).strip() if match else None


def _target_phrase(text: str) -> str:
    cleaned = re.sub(r"^(?:please\s+)?(?:find|locate|select|identify|choose)\s+", "", text, flags=re.I)
    cleaned = re.sub(r"^(?:from\s+)?(?:left\s+to\s+right|right\s+to\s+left)\s*,?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.I)
    return cleaned.strip(" .,;:!?")


def parse_query(query: str, query_id: str | None = None) -> dict[str, Any]:
    """Return a stable V2 schema without making model-dependent claims."""
    text = str(query or "").strip()
    lowered = text.lower()
    part = next((item for item in _PARTS if re.search(rf"\b{re.escape(item)}\b", lowered)), None)
    parent_match = re.search(r"\bof\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z -]{1,40})", lowered)
    parent = parent_match.group(1).strip() if part and parent_match else None
    expected_text = _quoted_text(text)
    carrier = next((item for item in _CARRIERS if re.search(rf"\b{re.escape(item)}\b", lowered)), None)
    thermal = bool(_THERMAL_RE.search(text))
    depth_match = _DEPTH_RE.search(text)
    depth_type = None
    if depth_match:
        token = depth_match.group(0).lower()
        depth_type = "absolute_ordinal" if token in {"closest", "nearest", "farthest", "furthest"} else "pairwise"

    return {
        "query_id": query_id,
        "raw_query": text,
        "target": {
            "class": _target_phrase(text),
            "attributes": [],
            "part": part,
            "parent_class": parent,
            "text_content": expected_text,
        },
        "count": 1,
        "ordinal": _ordinal(text),
        "depth": {"enabled": bool(depth_match), "type": depth_type, "scope": "target" if depth_match else None},
        "thermal": {"enabled": thermal, "scope": "unspecified" if thermal else None},
        "ocr": {"enabled": bool(expected_text and carrier), "carrier_class": carrier, "expected_text": expected_text},
        "relations": [],
    }
