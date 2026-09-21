import unittest

from src.boundary_refiner import apply_cached_boundary_refinements


class BoundaryRefinerTests(unittest.TestCase):
    def test_only_matching_high_score_refinement_is_applied(self):
        box = [0.1, 0.1, 0.5, 0.5]
        refined = [0.12, 0.12, 0.48, 0.48]
        base = {"q": {"bbox": box, "candidates": [{"bbox": box}]}}
        records = {"q": {"input_bbox": box, "refined_bbox": refined, "sam_score": 0.98}}
        output, stats = apply_cached_boundary_refinements(base, records)
        self.assertEqual(output["q"]["bbox"], refined)
        self.assertEqual(stats["changed"], 1)

    def test_stale_or_low_score_refinement_is_ignored(self):
        box = [0.1, 0.1, 0.5, 0.5]
        base = {"q": {"bbox": box, "candidates": [{"bbox": box}]}}
        stale = {"q": {"input_bbox": [0.2, 0.2, 0.4, 0.4], "refined_bbox": [0.1, 0.1, 0.4, 0.4], "sam_score": 1.0}}
        output, _ = apply_cached_boundary_refinements(base, stale)
        self.assertEqual(output["q"]["bbox"], box)
        low = {"q": {"input_bbox": box, "refined_bbox": [0.1, 0.1, 0.4, 0.4], "sam_score": 0.9}}
        output, _ = apply_cached_boundary_refinements(base, low)
        self.assertEqual(output["q"]["bbox"], box)


if __name__ == "__main__":
    unittest.main()
