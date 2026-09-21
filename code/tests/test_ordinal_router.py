import unittest

from src.ordinal_router import (
    apply_ordinal_router, deduplicate_candidates, explicit_horizontal_ordinal,
    select_ordinal_candidate,
)


class OrdinalRouterTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"bbox": [0.05, 0.1, 0.15, 0.3], "score": 0.7},
            {"bbox": [0.40, 0.1, 0.50, 0.3], "score": 0.6},
            {"bbox": [0.80, 0.1, 0.90, 0.3], "score": 0.8},
        ]

    def test_gate_only_accepts_explicit_second_horizontal_queries(self):
        self.assertEqual(explicit_horizontal_ordinal("second person from left"), (2, "left"))
        self.assertEqual(explicit_horizontal_ordinal("2nd car from the right"), (2, "right"))
        self.assertIsNone(explicit_horizontal_ordinal("third person from left"))
        self.assertIsNone(explicit_horizontal_ordinal("second person"))

    def test_selects_second_from_each_direction(self):
        left = select_ordinal_candidate("second object from left", self.candidates)
        right = select_ordinal_candidate("second object from right", self.candidates)
        self.assertEqual(left["bbox"], [0.40, 0.1, 0.50, 0.3])
        self.assertEqual(right["bbox"], [0.40, 0.1, 0.50, 0.3])

    def test_near_duplicate_boxes_are_counted_once(self):
        duplicate = dict(self.candidates[0])
        duplicate["bbox"] = [0.051, 0.1, 0.151, 0.3]
        self.assertEqual(len(deduplicate_candidates(self.candidates + [duplicate])), 3)

    def test_ambiguous_query_preserves_base_result(self):
        annotations = {"q": {"query": "second object"}}
        cache = {"q": {"candidates": self.candidates}}
        base = {"q": {"bbox": self.candidates[0]["bbox"], "candidates": self.candidates}}
        output, stats = apply_ordinal_router(annotations, cache, base)
        self.assertEqual(output["q"]["bbox"], self.candidates[0]["bbox"])
        self.assertEqual(stats["eligible"], 0)


if __name__ == "__main__":
    unittest.main()
