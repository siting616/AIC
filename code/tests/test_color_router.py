import unittest

from src.color_router import apply_color_router, pixel_matches_color, query_color


class ColorRouterTests(unittest.TestCase):
    def test_query_color_normalizes_grey(self):
        self.assertEqual(query_color("the grey car"), "gray")
        self.assertIsNone(query_color("the large car"))

    def test_basic_pixel_color_buckets(self):
        self.assertTrue(pixel_matches_color("red", 255, 0, 0))
        self.assertTrue(pixel_matches_color("blue", 0, 0, 255))
        self.assertTrue(pixel_matches_color("black", 5, 5, 5))
        self.assertTrue(pixel_matches_color("white", 250, 250, 250))

    def test_safe_color_route_requires_margin(self):
        annotations = {"q": {"query": "blue object"}}
        raw = {"q": {"candidates": [
            {"bbox": [0.1, 0.1, 0.2, 0.2]}, {"bbox": [0.5, 0.1, 0.6, 0.2]}
        ]}}
        base = {"q": {"bbox": [0.1, 0.1, 0.2, 0.2], "candidates": [
            {"bbox": [0.1, 0.1, 0.2, 0.2]}, {"bbox": [0.5, 0.1, 0.6, 0.2]}
        ]}}
        features = {"q": {"color": "blue", "fractions": [0.1, 0.3]}}
        output, stats = apply_color_router(annotations, raw, features, base)
        self.assertEqual(output["q"]["bbox"], [0.5, 0.1, 0.6, 0.2])
        self.assertEqual(stats["changed"], 1)

    def test_unsafe_color_preserves_base(self):
        annotations = {"q": {"query": "green object"}}
        raw = {"q": {"candidates": [{"bbox": [0.1, 0.1, 0.2, 0.2]}, {"bbox": [0.5, 0.1, 0.6, 0.2]}]}}
        base = {"q": {"bbox": [0.1, 0.1, 0.2, 0.2], "candidates": raw["q"]["candidates"]}}
        output, stats = apply_color_router(annotations, raw, {"q": {"color": "green", "fractions": [0.0, 1.0]}}, base)
        self.assertEqual(output["q"]["bbox"], [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(stats["changed"], 0)


if __name__ == "__main__":
    unittest.main()
