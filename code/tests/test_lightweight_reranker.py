import unittest

from src.candidate_cache import build_candidate_cache, validate_candidate_cache
from src.lightweight_reranker import (
    FEATURE_NAMES,
    feature_matrix,
    rerank_records,
    score_candidates,
    train_ridge_reranker,
)


class LightweightRerankerTests(unittest.TestCase):
    def test_cache_accepts_legacy_reranked_candidates(self):
        annotations = {"q": {"query": "left object", "source_image_id": 1}}
        source = {"q": {"reranked_candidates": [
            {"bbox": [0.1, 0.1, 0.2, 0.2], "score": 0.7, "rank": 2, "label": "object"}
        ]}}
        cache = build_candidate_cache(annotations, source, "fixture")
        self.assertTrue(validate_candidate_cache(cache, annotations))
        self.assertEqual(cache["records"]["q"]["candidates"][0]["source_rank"], 2)

    def test_features_encode_left_position(self):
        candidates = [
            {"bbox": [0.1, 0.1, 0.2, 0.2], "score": 0.5, "source_rank": 1},
            {"bbox": [0.8, 0.1, 0.9, 0.2], "score": 0.5, "source_rank": 2},
        ]
        matrix = feature_matrix("left object", candidates)
        self.assertLess(matrix[0][13], matrix[1][13])

    def test_train_and_rerank_prefers_learned_left_candidate(self):
        annotations = {}
        records = {}
        for index in range(20):
            sample_id = f"q{index}"
            annotations[sample_id] = {"query": "left object", "bbox": [0.1, 0.1, 0.2, 0.2]}
            records[sample_id] = {"candidates": [
                {"bbox": [0.8, 0.1, 0.9, 0.2], "score": 0.9, "source_rank": 1, "label": "object"},
                {"bbox": [0.1, 0.1, 0.2, 0.2], "score": 0.4, "source_rank": 2, "label": "object"},
            ]}
        model = train_ridge_reranker(annotations, records, list(annotations), regularization=0.1)
        ranked = rerank_records(annotations, records, model)
        self.assertEqual(ranked["q0"]["bbox"], [0.1, 0.1, 0.2, 0.2])

    def test_old_model_scores_only_its_named_feature_subset(self):
        candidates = [{
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "score": 0.75,
            "source_rank": 1,
        }]
        model = {
            "feature_names": FEATURE_NAMES[:2],
            "mean": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "weights": [0.0, 2.0, 3.0],
        }
        self.assertEqual(score_candidates("object", candidates, model), [4.5])


if __name__ == "__main__":
    unittest.main()
