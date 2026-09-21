import unittest

from src.query_prompts import generate_prompts


class RescueSelectionTests(unittest.TestCase):
    def test_complex_query_has_multiple_detector_prompts(self):
        prompts = generate_prompts("the second red person from the left")
        detector = [prompt for prompt in prompts if prompt["type"] != "reference"]
        self.assertGreaterEqual(len(detector), 2)

    def test_reference_prompt_is_not_a_detector_target(self):
        prompts = generate_prompts("the person next to the car")
        detector = [prompt for prompt in prompts if prompt["type"] != "reference"]
        self.assertTrue(any(prompt["type"] == "reference" for prompt in prompts))
        self.assertTrue(all(prompt["text"] != "car" for prompt in detector))


if __name__ == "__main__":
    unittest.main()
