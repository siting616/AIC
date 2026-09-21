import unittest

from scripts.run_qwen_pairwise_rescue import bbox_iou


class PairwiseRescueTests(unittest.TestCase):
    def test_iou_identical(self):
        self.assertEqual(bbox_iou([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]), 1.0)

    def test_iou_disjoint(self):
        self.assertEqual(bbox_iou([0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]), 0.0)


if __name__ == "__main__":
    unittest.main()
