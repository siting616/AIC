import unittest

from src.query_prompts import generate_prompts, merge_prompt_candidates


class QueryPromptTests(unittest.TestCase):
    def test_ordinal_query_keeps_original_and_extracts_target(self):
        prompts = generate_prompts(
            "the second person from the left wearing a red shirt"
        )
        values = {item["type"]: item["text"] for item in prompts}
        self.assertEqual(
            values["original"],
            "the second person from the left wearing a red shirt",
        )
        self.assertEqual(values["target_phrase"], "person")
        self.assertNotIn("head_noun", values)

    def test_relation_query_emits_reference_prompt(self):
        prompts = generate_prompts("the cup behind the blue bottle")
        values = {item["type"]: item["text"] for item in prompts}
        self.assertEqual(values["target_phrase"], "cup")
        self.assertNotIn("head_noun", values)
        self.assertEqual(values["reference"], "blue bottle")

    def test_duplicate_prompts_are_removed(self):
        prompts = generate_prompts("person")
        self.assertEqual(prompts, [{"type": "original", "text": "person"}])

    def test_directional_counting_prefix_is_not_a_reference_relation(self):
        prompts = generate_prompts("From left to right, the third stone pier")
        values = {item["type"]: item["text"] for item in prompts}
        self.assertEqual(values["target_phrase"], "stone pier")
        self.assertNotIn("reference", values)

    def test_late_reference_class_is_not_used_as_head_noun(self):
        prompts = generate_prompts("The passage to the upper level of the building")
        values = {item["type"]: item["text"] for item in prompts}
        self.assertNotIn("head_noun", values)

    def test_candidate_fusion_rewards_cross_prompt_agreement(self):
        results = [
            {
                "prompt_type": "original",
                "prompt": "red shirt person",
                "candidates": [
                    {"bbox": [0.1, 0.1, 0.3, 0.4], "score": 0.40},
                    {"bbox": [0.6, 0.1, 0.8, 0.4], "score": 0.45},
                ],
            },
            {
                "prompt_type": "head_noun",
                "prompt": "person",
                "candidates": [
                    {"bbox": [0.105, 0.1, 0.305, 0.4], "score": 0.38},
                ],
            },
        ]
        merged = merge_prompt_candidates(results, top_k=5)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["agreement_count"], 2)
        self.assertIn("original", merged[0]["prompt_types"])
        self.assertIn("head_noun", merged[0]["prompt_types"])

    def test_reference_candidates_do_not_enter_target_shortlist(self):
        results = [
            {
                "prompt_type": "original",
                "prompt": "cup behind bottle",
                "candidates": [{"bbox": [0.1, 0.1, 0.2, 0.2], "score": 0.2}],
            },
            {
                "prompt_type": "reference",
                "prompt": "bottle",
                "candidates": [{"bbox": [0.6, 0.1, 0.8, 0.5], "score": 0.9}],
            },
        ]
        merged = merge_prompt_candidates(results, top_k=5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["prompt_type"], "original")


if __name__ == "__main__":
    unittest.main()
