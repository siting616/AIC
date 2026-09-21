"""Unit tests for GroundingDINO threshold rescue and API compatibility."""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from src.baseline import GroundingDINOPredictor


class _FakeProcessorOld:
    def __init__(self):
        self.received = None

    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids,
        box_threshold,
        text_threshold,
        target_sizes,
    ):
        self.received = {
            "outputs": outputs,
            "input_ids": input_ids,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "target_sizes": target_sizes,
        }
        return ["old-api-result"]


class _FakeProcessorNew:
    def __init__(self):
        self.received = None

    def post_process_grounded_object_detection(
        self,
        outputs,
        threshold,
        text_threshold,
        target_sizes,
    ):
        self.received = {
            "outputs": outputs,
            "threshold": threshold,
            "text_threshold": text_threshold,
            "target_sizes": target_sizes,
        }
        return ["new-api-result"]


def _bare_predictor(processor):
    predictor = GroundingDINOPredictor.__new__(GroundingDINOPredictor)
    predictor.processor = processor
    return predictor


class _FakeInputs(dict):
    def to(self, device):
        return self


class GroundingDINOCompatibilityTests(unittest.TestCase):
    def test_old_transformers_api_uses_input_ids_and_box_threshold(self):
        processor = _FakeProcessorOld()
        predictor = _bare_predictor(processor)
        inputs = {"input_ids": "token-ids"}

        result = predictor._post_process(
            outputs="model-output",
            inputs=inputs,
            height=480,
            width=640,
            score_threshold=0.10,
            text_threshold=0.10,
        )

        self.assertEqual(result, ["old-api-result"])
        self.assertEqual(processor.received["input_ids"], "token-ids")
        self.assertEqual(processor.received["box_threshold"], 0.10)
        self.assertEqual(processor.received["target_sizes"], [(480, 640)])

    def test_new_transformers_api_uses_threshold(self):
        processor = _FakeProcessorNew()
        predictor = _bare_predictor(processor)

        result = predictor._post_process(
            outputs="model-output",
            inputs={"input_ids": "unused-token-ids"},
            height=720,
            width=1280,
            score_threshold=0.15,
            text_threshold=0.15,
        )

        self.assertEqual(result, ["new-api-result"])
        self.assertEqual(processor.received["threshold"], 0.15)
        self.assertEqual(processor.received["target_sizes"], [(720, 1280)])

    def test_primary_empty_triggers_rescue_with_expected_thresholds(self):
        predictor = GroundingDINOPredictor.__new__(GroundingDINOPredictor)
        predictor.Image = SimpleNamespace(
            open=Mock(
                return_value=SimpleNamespace(
                    convert=Mock(
                        return_value=SimpleNamespace(size=(640, 480))
                    )
                )
            )
        )
        predictor.device = "cpu"
        predictor.processor = Mock()
        predictor.processor.return_value = _FakeInputs(input_ids="token-ids")
        predictor.model = Mock(return_value="model-output")
        predictor.torch = SimpleNamespace(
            no_grad=Mock(return_value=nullcontext())
        )
        predictor.score_threshold = 0.15
        predictor.text_threshold = 0.15
        predictor.rescue_score_threshold = 0.10
        predictor.rescue_text_threshold = 0.10
        predictor.top_k = 5
        predictor._post_process = Mock(
            side_effect=[
                [{"boxes": []}],
                [{"boxes": []}],
            ]
        )

        detail = predictor.predict_detailed(
            {
                "visible_path": "unused.jpg",
                "query": "the person near the vehicle",
            }
        )

        self.assertTrue(detail["used_fallback"])
        self.assertEqual(detail["selection_mode"], "center_fallback")
        self.assertEqual(predictor._post_process.call_count, 2)
        primary_call = predictor._post_process.call_args_list[0].args
        rescue_call = predictor._post_process.call_args_list[1].args
        self.assertEqual(primary_call[-2:], (0.15, 0.15))
        self.assertEqual(rescue_call[-2:], (0.10, 0.10))


if __name__ == "__main__":
    unittest.main()
