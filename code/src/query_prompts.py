"""Query decomposition and candidate fusion for multi-prompt grounding."""

from __future__ import annotations

import re
from copy import deepcopy


ORDINAL_WORDS = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    "leftmost|rightmost"
)

RELATION_MARKERS = re.compile(
    r"\b(?:to the left of|to the right of|left of|right of|in front of|"
    r"behind|next to|near|beside|between|under|below|above|over|from the left|"
    r"from the right|from left|from right)\b",
    re.IGNORECASE,
)

# Conservative vocabulary: a class prompt is emitted only when the target noun
# is explicit. The original and stripped target phrase remain available for
# everything outside this vocabulary.
TARGET_CLASSES = (
    "traffic light", "fire hydrant", "parking meter", "sports ball",
    "baseball bat", "baseball glove", "tennis racket", "wine glass",
    "cell phone", "potted plant", "dining table", "stop sign",
    "motorcycle", "bicycle", "airplane", "aeroplane", "pedestrian",
    "refrigerator", "microwave", "keyboard", "backpack", "umbrella",
    "suitcase", "handbag", "sandwich", "broccoli", "toothbrush",
    "person", "people", "man", "woman", "boy", "girl", "child",
    "worker", "player", "rider", "animal", "dog", "cat", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bird",
    "car", "truck", "bus", "train", "boat", "vehicle", "scooter",
    "chair", "bench", "couch", "sofa", "bed", "table", "desk",
    "bottle", "cup", "glass", "bowl", "plate", "fork", "knife",
    "spoon", "banana", "apple", "orange", "pizza", "cake", "food",
    "sign", "screen", "monitor", "television", "tv", "laptop",
    "mouse", "remote", "phone", "book", "clock", "vase", "scissors",
    "bag", "box", "door", "window", "building", "tree", "plant",
    "shirt", "jacket", "coat", "hat", "helmet", "shoe", "hand",
    "head", "face", "arm", "leg", "towel", "flag", "pole",
)


def _clean(text: str) -> str:
    text = re.sub(r"[\s\t\r\n]+", " ", text).strip(" .,;:!?\t\r\n")
    return text


def _strip_instruction(text: str) -> str:
    text = re.sub(
        r"^(?:please\s+)?(?:find|select|locate|identify|choose|show me|point to)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Directional counting prefixes describe how to order target instances;
    # they are not target/reference relations. Remove them before splitting on
    # relation markers (which otherwise sees the leading "from left").
    text = re.sub(
        r"^(?:from\s+)?(?:left\s+to\s+right|right\s+to\s+left)\s*,?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _strip_positional_words(text: str) -> str:
    text = re.sub(rf"\b(?:the\s+)?(?:{ORDINAL_WORDS})\b", "", text, flags=re.I)
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:far\s+)?(?:left|right|top|bottom|middle|center)\b", "", text, flags=re.I)
    return _clean(text)


def _find_target_class(text: str) -> str | None:
    lowered = text.lower()
    matches = []
    for target_class in TARGET_CLASSES:
        match = re.search(rf"\b{re.escape(target_class)}\b", lowered)
        # A late class word often belongs to a prepositional reference, e.g.
        # "passage to the upper level of the building". Only trust explicit
        # class words near the beginning of the target phrase.
        if match and len(lowered[: match.start()].split()) <= 4:
            matches.append((match.start(), -len(target_class), target_class))
    return min(matches)[2] if matches else None


def generate_prompts(query: str, max_prompts: int = 4) -> list[dict[str, str]]:
    """Generate a conservative ordered prompt set for one referring query."""
    original = _clean(str(query))
    if not original:
        return []

    prompts: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(prompt_type: str, text: str):
        text = _clean(text)
        key = text.lower()
        if text and key not in seen and len(prompts) < max_prompts:
            prompts.append({"type": prompt_type, "text": text})
            seen.add(key)

    add("original", original)
    instructed = _strip_instruction(original)
    relation = RELATION_MARKERS.search(instructed)
    target_phrase = instructed[: relation.start()] if relation else instructed
    target_phrase = _strip_positional_words(target_phrase)
    target_phrase = re.sub(r"^(?:the|a|an)\s+", "", target_phrase, flags=re.I)
    add("target_phrase", target_phrase)

    target_class = _find_target_class(target_phrase or instructed)
    if target_class:
        add("head_noun", target_class)

    if relation:
        reference_phrase = instructed[relation.end() :]
        reference_phrase = _strip_positional_words(reference_phrase)
        reference_phrase = re.sub(r"^(?:the|a|an)\s+", "", reference_phrase, flags=re.I)
        add("reference", reference_phrase)

    return prompts


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def merge_prompt_candidates(
    prompt_results: list[dict],
    top_k: int = 10,
    per_prompt: int = 5,
    dedup_iou: float = 0.85,
) -> list[dict]:
    """Fuse prompt-specific candidates while retaining provenance and diversity."""
    clusters: list[dict] = []
    for prompt_result in prompt_results:
        prompt_type = prompt_result["prompt_type"]
        prompt_text = prompt_result["prompt"]
        # Reference detections are context, not valid target answers. They may
        # be visualized separately in a later relation-aware selector, but must
        # never consume the target shortlist used by the current Qwen chooser.
        if prompt_type == "reference":
            continue
        for candidate in prompt_result.get("candidates", [])[:per_prompt]:
            if not candidate.get("bbox"):
                continue
            enriched = deepcopy(candidate)
            enriched["source_modality"] = "visible"
            enriched["prompt_type"] = prompt_type
            enriched["prompt"] = prompt_text
            cluster = next(
                (
                    item
                    for item in clusters
                    if bbox_iou(item["representative"]["bbox"], enriched["bbox"])
                    >= dedup_iou
                ),
                None,
            )
            if cluster is None:
                clusters.append({"representative": enriched, "members": [enriched]})
            else:
                cluster["members"].append(enriched)
                if float(enriched.get("score", 0.0)) > float(
                    cluster["representative"].get("score", 0.0)
                ):
                    cluster["representative"] = enriched

    fused = []
    for cluster in clusters:
        members = cluster["members"]
        representative = deepcopy(cluster["representative"])
        prompt_types = sorted({item["prompt_type"] for item in members})
        prompts = sorted({item["prompt"] for item in members})
        agreement = min(max(len(prompt_types) - 1, 0) / 2.0, 1.0)
        original_bonus = 1.0 if "original" in prompt_types else 0.0
        model_score = max(float(item.get("score", 0.0)) for item in members)
        representative["model_score"] = model_score
        representative["agreement_count"] = len(prompt_types)
        representative["prompt_types"] = prompt_types
        representative["prompts"] = prompts
        representative["fusion_score"] = (
            0.65 * model_score + 0.20 * agreement + 0.15 * original_bonus
        )
        fused.append(representative)

    fused.sort(key=lambda item: item["fusion_score"], reverse=True)
    selected: list[dict] = []
    # Avoid allowing one generic prompt to monopolize the shortlist.
    type_limits = {"original": 3, "target_phrase": 3, "head_noun": 3, "reference": 1}
    type_counts = {key: 0 for key in type_limits}
    for candidate in fused:
        primary_type = candidate["prompt_type"]
        if type_counts.get(primary_type, 0) >= type_limits.get(primary_type, top_k):
            continue
        selected.append(candidate)
        type_counts[primary_type] = type_counts.get(primary_type, 0) + 1
        if len(selected) >= top_k:
            break

    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    return selected
