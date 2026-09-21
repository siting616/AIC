import unittest

from src.candidate_merge import merge_candidate_lists, merge_tiled_records


class CandidateMergeTests(unittest.TestCase):
    def test_merge_deduplicates_and_respects_budget(self):
        base = [{"bbox": [0.1, 0.1, 0.2, 0.2], "score": 0.9}]
        extras = [
            {"bbox": [0.101, 0.1, 0.201, 0.2], "score": 0.8},
            {"bbox": [0.5, 0.5, 0.7, 0.7], "score": 0.7},
            {"bbox": [0.8, 0.8, 0.9, 0.9], "score": 0.6},
        ]
        merged = merge_candidate_lists(base, extras, max_candidates=2, dedup_iou=0.85)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1]["bbox"], [0.5, 0.5, 0.7, 0.7])

    def test_base_candidates_are_also_deduplicated(self):
        base = [
            {"bbox": [0.1, 0.1, 0.2, 0.2], "score": 0.9},
            {"bbox": [0.101, 0.1, 0.201, 0.2], "score": 0.8},
        ]
        merged = merge_candidate_lists(base, [], max_candidates=20, dedup_iou=0.85)
        self.assertEqual(len(merged), 1)

    def test_ground_truth_record_bbox_is_never_merged(self):
        base = {
            "schema_version": 1, "source_name": "base",
            "records": {"q": {"query": "object", "candidates": [
                {"bbox": [0.1, 0.1, 0.2, 0.2], "score": 0.9}
            ]}},
        }
        tiled = {"q": {
            "bbox": [0.7, 0.7, 0.9, 0.9],
            "new_candidates": [{"bbox": [0.4, 0.4, 0.5, 0.5], "score": 0.5}],
        }}
        merged = merge_tiled_records(base, tiled)
        boxes = [item["bbox"] for item in merged["records"]["q"]["candidates"]]
        self.assertIn([0.4, 0.4, 0.5, 0.5], boxes)
        self.assertNotIn([0.7, 0.7, 0.9, 0.9], boxes)

    def test_generic_candidates_field_can_be_merged(self):
        base = {"schema_version": 1, "source_name": "base", "records": {
            "q": {"query": "object", "candidates": [{"bbox": [0.1, 0.1, 0.2, 0.2]}]}
        }}
        source = {"q": {"candidates": [{"bbox": [0.5, 0.5, 0.6, 0.6], "score": 0.4}]}}
        merged = merge_tiled_records(base, source, candidate_field="candidates")
        self.assertEqual(len(merged["records"]["q"]["candidates"]), 2)
        self.assertEqual(merged["records"]["q"]["candidates"][1]["source_origin"], "tiled")


if __name__ == "__main__":
    unittest.main()
