import unittest

from src.rerank_v18 import query_route, rerank_candidates


class V18RoutingTests(unittest.TestCase):
    def test_relational_left_is_not_global_route(self):
        self.assertIsNone(query_route("the cup left of the pink towel"))

    def test_ordinal_from_right(self):
        self.assertEqual(
            query_route("Count the fourth monkey from right to left"),
            {"kind": "ordinal_x", "index": 4, "reverse": True},
        )

    def test_rightmost_is_hard_route(self):
        self.assertEqual(
            query_route("the rightmost white car"),
            {"kind": "extreme_x", "reverse": True},
        )

    def test_local_left_inside_right_region_is_not_global(self):
        self.assertIsNone(query_route(
            "The person on the left among two people on the right side of the image"
        ))

    def test_ordinal_reference_is_not_target_ordinal(self):
        self.assertIsNone(query_route(
            "The fence bar immediately right of the fourth stone pier from the left"
        ))

    def test_ordinal_selects_requested_unique_candidate(self):
        candidates = [
            {"bbox": [0.05, 0.1, 0.15, 0.3], "score": 0.9},
            {"bbox": [0.45, 0.1, 0.55, 0.3], "score": 0.8},
            {"bbox": [0.85, 0.1, 0.95, 0.3], "score": 0.7},
        ]
        ranked = rerank_candidates(
            {"query": "the second object from the right"}, candidates,
            weights={"model": 1.0, "position": 0.0, "depth": 0.0, "infrared": 0.0},
        )
        self.assertEqual(ranked[0]["bbox"], candidates[1]["bbox"])

    def test_extreme_only_leaves_ordinal_on_v17(self):
        candidates = [
            {"bbox": [0.05, 0.1, 0.15, 0.3], "score": 0.9},
            {"bbox": [0.45, 0.1, 0.55, 0.3], "score": 0.8},
        ]
        ranked = rerank_candidates(
            {"query": "the second object from the left"}, candidates,
            weights={"model": 1.0, "position": 0.0, "depth": 0.0, "infrared": 0.0},
            route_mode="extreme",
        )
        self.assertEqual(ranked[0]["bbox"], candidates[0]["bbox"])

    def test_ordinal_first_does_not_route_second(self):
        candidates = [
            {"bbox": [0.05, 0.1, 0.15, 0.3], "score": 0.9},
            {"bbox": [0.45, 0.1, 0.55, 0.3], "score": 0.8},
        ]
        ranked = rerank_candidates(
            {"query": "the second object from the left"}, candidates,
            weights={"model": 1.0, "position": 0.0, "depth": 0.0, "infrared": 0.0},
            route_mode="ordinal_first",
        )
        self.assertEqual(ranked[0]["bbox"], candidates[0]["bbox"])

    def test_ordinal_later_does_not_route_first(self):
        candidates = [
            {"bbox": [0.05, 0.1, 0.15, 0.3], "score": 0.8},
            {"bbox": [0.45, 0.1, 0.55, 0.3], "score": 0.9},
        ]
        ranked = rerank_candidates(
            {"query": "the first object from the left"}, candidates,
            weights={"model": 1.0, "position": 0.0, "depth": 0.0, "infrared": 0.0},
            route_mode="ordinal_later",
        )
        self.assertEqual(ranked[0]["bbox"], candidates[1]["bbox"])

    def test_ordinal_second_does_not_route_third(self):
        candidates = [
            {"bbox": [0.05, 0.1, 0.15, 0.3], "score": 0.9},
            {"bbox": [0.45, 0.1, 0.55, 0.3], "score": 0.8},
            {"bbox": [0.75, 0.1, 0.85, 0.3], "score": 0.7},
        ]
        ranked = rerank_candidates(
            {"query": "the third object from the left"}, candidates,
            weights={"model": 1.0, "position": 0.0, "depth": 0.0, "infrared": 0.0},
            route_mode="ordinal_second",
        )
        self.assertEqual(ranked[0]["bbox"], candidates[0]["bbox"])

    def test_ordinal_third_plus_does_not_route_second(self):
        candidates = [
            {"bbox": [0.05, 0.1, 0.15, 0.3], "score": 0.9},
            {"bbox": [0.45, 0.1, 0.55, 0.3], "score": 0.8},
            {"bbox": [0.75, 0.1, 0.85, 0.3], "score": 0.7},
        ]
        ranked = rerank_candidates(
            {"query": "the second object from the left"}, candidates,
            weights={"model": 1.0, "position": 0.0, "depth": 0.0, "infrared": 0.0},
            route_mode="ordinal_third_plus",
        )
        self.assertEqual(ranked[0]["bbox"], candidates[0]["bbox"])


if __name__ == "__main__":
    unittest.main()
