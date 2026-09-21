import unittest

from src.rerank import position_score, query_intents


class RerankRulesTest(unittest.TestCase):
    def test_left_prefers_left_candidate(self):
        intents = query_intents("the person on the left")
        left = position_score([0.05, 0.2, 0.25, 0.8], intents)
        right = position_score([0.75, 0.2, 0.95, 0.8], intents)
        self.assertGreater(left, right)

    def test_right_prefers_right_candidate(self):
        intents = query_intents("the rightmost white car")
        left = position_score([0.05, 0.2, 0.25, 0.8], intents)
        right = position_score([0.75, 0.2, 0.95, 0.8], intents)
        self.assertGreater(right, left)

    def test_depth_intents(self):
        self.assertTrue(query_intents("the nearest person")["near"])
        self.assertTrue(query_intents("the farthest animal")["far"])


if __name__ == "__main__":
    unittest.main()
