"""CPU-only linear candidate reranker with reproducible NumPy training."""

from __future__ import annotations

import re

import math

from .metrics import compute_iou


TOKEN_RE = re.compile(r"[a-z0-9]+")
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}

FEATURE_NAMES = [
    "detector_score", "inverse_source_rank", "cx", "cy", "width", "height",
    "area", "aspect_ratio", "edge_touch", "label_query_overlap",
    "x_rank", "y_rank", "area_rank", "left_x", "right_x", "top_y",
    "bottom_y", "center_distance", "small_area", "large_area",
    "ordinal_x_distance", "ordinal_right_x_distance", "query_length",
    "source_is_tiled", "source_is_small_3x3", "source_is_multiprompt",
]


def _tokens(text):
    return set(TOKEN_RE.findall(str(text).lower()))


def _ordinal(query):
    tokens = _tokens(query)
    for word, value in ORDINALS.items():
        if word in tokens:
            return value
    return 0


def feature_matrix(query, candidates):
    if not candidates:
        return []
    boxes = [item["bbox"] for item in candidates]
    scores = [float(item.get("score", 0.0)) for item in candidates]
    ranks = [float(item.get("source_rank", i + 1)) for i, item in enumerate(candidates)]
    cx = [(box[0] + box[2]) / 2 for box in boxes]
    cy = [(box[1] + box[3]) / 2 for box in boxes]
    width = [box[2] - box[0] for box in boxes]
    height = [box[3] - box[1] for box in boxes]
    area = [w * h for w, h in zip(width, height)]
    n = len(candidates)

    def normalized_rank(values):
        order = sorted(range(n), key=lambda index: (values[index], index))
        result = [0.0] * n
        for rank, index in enumerate(order):
            result[index] = rank / max(1, n - 1)
        return result

    x_rank = normalized_rank(cx)
    y_rank = normalized_rank(cy)
    area_rank = normalized_rank(area)
    tokens = _tokens(query)
    query_len = min(len(tokens), 20) / 20.0
    overlap = []
    for item in candidates:
        label_tokens = _tokens(item.get("label", ""))
        overlap.append(len(tokens & label_tokens) / max(1, len(label_tokens)))
    left = float(bool(tokens & {"left", "leftmost"}))
    right = float(bool(tokens & {"right", "rightmost"}))
    top = float(bool(tokens & {"top", "upper", "above"}))
    bottom = float(bool(tokens & {"bottom", "lower", "below", "under"}))
    center = float(bool(tokens & {"center", "middle"}))
    small = float(bool(tokens & {"small", "smallest", "tiny"}))
    large = float(bool(tokens & {"large", "largest", "big", "biggest"}))
    ordinal = _ordinal(query)
    desired = (ordinal - 1) / max(1, n - 1) if ordinal else 0.0
    from_right = float("from right" in str(query).lower())
    rows = []
    for i, box in enumerate(boxes):
        edge_touch = float(box[0] < 0.01 or box[1] < 0.01 or box[2] > 0.99 or box[3] > 0.99)
        center_distance = math.hypot(cx[i] - 0.5, cy[i] - 0.5) * center
        origin = str(candidates[i].get("source_origin", candidates[i].get("source", ""))).lower()
        rows.append([
            scores[i], 1.0 / max(ranks[i], 1), cx[i], cy[i], width[i], height[i],
            area[i], min(max(width[i] / max(height[i], 1e-6), 0), 10) / 10,
            edge_touch, overlap[i], x_rank[i], y_rank[i], area_rank[i],
            left * cx[i], right * cx[i], top * cy[i], bottom * cy[i], center_distance,
            small * area[i], large * area[i],
            -float(bool(ordinal)) * abs(x_rank[i] - desired),
            -from_right * abs((1.0 - x_rank[i]) - desired), query_len,
            float("tile" in origin), float("3x3" in origin), float("multiprompt" in origin),
        ])
    return rows


def training_arrays(annotations, cache_records, sample_ids):
    rows = []
    targets = []
    for sample_id in sample_ids:
        annotation = annotations[sample_id]
        candidates = cache_records.get(sample_id, {}).get("candidates", [])
        if not candidates:
            continue
        rows.extend(feature_matrix(annotation.get("query", ""), candidates))
        targets.extend(compute_iou(item["bbox"], annotation["bbox"]) for item in candidates)
    if not rows:
        raise ValueError("no cached candidates available for training")
    return rows, targets


def _solve_linear_system(matrix, vector):
    """Solve a small dense system with pivoted Gauss-Jordan elimination."""
    n = len(vector)
    augmented = [list(matrix[i]) + [float(vector[i])] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular training system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [augmented[i][-1] for i in range(n)]


def train_ridge_reranker(annotations, cache_records, sample_ids, regularization=10.0):
    x, y = training_arrays(annotations, cache_records, sample_ids)
    count = len(x)
    width = len(FEATURE_NAMES)
    mean = [sum(row[j] for row in x) / count for j in range(width)]
    scale = [math.sqrt(sum((row[j] - mean[j]) ** 2 for row in x) / count) for j in range(width)]
    scale = [value if value >= 1e-8 else 1.0 for value in scale]
    dimension = width + 1
    gram = [[0.0] * dimension for _ in range(dimension)]
    rhs = [0.0] * dimension
    for row, target in zip(x, y):
        design = [1.0] + [(row[j] - mean[j]) / scale[j] for j in range(width)]
        for i, value_i in enumerate(design):
            rhs[i] += value_i * target
            for j in range(i, dimension):
                gram[i][j] += value_i * design[j]
    for i in range(dimension):
        for j in range(i):
            gram[i][j] = gram[j][i]
        if i:
            gram[i][i] += float(regularization)
    weights = _solve_linear_system(gram, rhs)
    predictions = [
        weights[0] + sum(weights[j + 1] * (row[j] - mean[j]) / scale[j] for j in range(width))
        for row in x
    ]
    return {
        "schema_version": 1,
        "model_type": "standardized_ridge_iou_ranker",
        "feature_names": FEATURE_NAMES,
        "regularization": float(regularization),
        "training_samples": len(sample_ids),
        "training_candidates": len(y),
        "target_mean_iou": sum(y) / len(y),
        "train_rmse": math.sqrt(sum((prediction - target) ** 2 for prediction, target in zip(predictions, y)) / len(y)),
        "mean": mean,
        "scale": scale,
        "weights": weights,
    }


def score_candidates(query, candidates, model):
    x = feature_matrix(query, candidates)
    if not x:
        return []
    model_feature_names = model.get("feature_names", FEATURE_NAMES)
    if len(set(model_feature_names)) != len(model_feature_names):
        raise ValueError("model contains duplicate feature names")
    unknown = [name for name in model_feature_names if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"model contains unknown features: {unknown}")
    feature_indices = [FEATURE_NAMES.index(name) for name in model_feature_names]
    mean = model["mean"]
    scale = model["scale"]
    weights = model["weights"]
    width = len(model_feature_names)
    if len(mean) != width or len(scale) != width or len(weights) != width + 1:
        raise ValueError("model parameter dimensions do not match feature_names")
    return [
        weights[0] + sum(
            weights[j + 1] * (row[feature_indices[j]] - mean[j]) / scale[j]
            for j in range(width)
        )
        for row in x
    ]


def rerank_records(annotations, cache_records, model):
    output = {}
    for sample_id, annotation in annotations.items():
        candidates = cache_records.get(sample_id, {}).get("candidates", [])
        scores = score_candidates(annotation.get("query", ""), candidates, model)
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        ranked = []
        for rank, index in enumerate(order, 1):
            item = dict(candidates[int(index)])
            item["reranker_score"] = float(scores[int(index)])
            item["rank"] = rank
            ranked.append(item)
        output[sample_id] = {
            "bbox": ranked[0]["bbox"] if ranked else None,
            "candidates": ranked,
        }
    return output
