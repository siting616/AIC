"""Baseline predictor interfaces for the AIC visual grounding task."""

from __future__ import annotations

from pathlib import Path

from .postprocess import DEFAULT_FALLBACK


class BasePredictor:
    """Base predictor interface."""

    def predict(self, sample):
        raise NotImplementedError

    def predict_detailed(self, sample):
        bbox = self.predict(sample)
        return {
            "bbox": bbox,
            "candidates": [
                {
                    "bbox": bbox,
                    "score": 1.0,
                    "label": "baseline",
                    "rank": 1,
                }
            ],
            "used_fallback": False,
        }


class OraclePredictor(BasePredictor):
    """Return ground-truth boxes. Use only for pipeline testing."""

    def predict(self, sample):
        if sample.get("bbox") is None:
            raise ValueError(
                "OraclePredictor cannot run on unlabeled competition data. "
                "Use a real model such as grounding_dino."
            )
        return sample["bbox"]


class CenterBoxPredictor(BasePredictor):
    """Return a fixed center box for submission-format testing."""

    def predict(self, sample):
        return DEFAULT_FALLBACK.copy()


class GroundingDINOPredictor(BasePredictor):
    """GroundingDINO predictor based on Hugging Face Transformers.

    This class is optional at runtime. Install the model dependencies in
    requirements_model.txt before setting baseline.name to "grounding_dino".
    """

    def __init__(self, config):
        self.config = config
        model_cfg = config.get("model", {}).get("grounding_dino", {})
        self.model_id = model_cfg.get("model_id", "IDEA-Research/grounding-dino-tiny")
        self.device = model_cfg.get("device", "cpu")
        self.score_threshold = float(model_cfg.get("score_threshold", 0.25))
        self.text_threshold = float(model_cfg.get("text_threshold", 0.20))
        self.rescue_score_threshold = float(
            model_cfg.get("rescue_score_threshold", 0.12)
        )
        self.rescue_text_threshold = float(
            model_cfg.get("rescue_text_threshold", 0.10)
        )
        self.top_k = max(1, int(model_cfg.get("top_k", 5)))
        if self.rescue_score_threshold > self.score_threshold:
            raise ValueError(
                "rescue_score_threshold must not exceed score_threshold"
            )
        if self.rescue_text_threshold > self.text_threshold:
            raise ValueError(
                "rescue_text_threshold must not exceed text_threshold"
            )

        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "GroundingDINOPredictor requires optional dependencies. "
                "Install them with: pip install -r requirements_model.txt"
            ) from exc

        self.torch = torch
        self.Image = Image
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()

    def predict_detailed(self, sample):
        image_path = Path(sample["visible_path"])
        image = self.Image.open(image_path).convert("RGB")
        width, height = image.size
        # A single image/query pair should be passed as one text sequence.
        # Some Transformers versions reject the nested ``[[query]]`` form
        # with "TextEncodeInput must be Union[...]".
        text_prompt = str(sample["query"])
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        results = self._post_process(
            outputs,
            inputs,
            height,
            width,
            self.score_threshold,
            self.text_threshold,
        )
        result = results[0]
        selection_mode = "primary_threshold"

        if len(result["boxes"]) == 0:
            rescue_results = self._post_process(
                outputs,
                inputs,
                height,
                width,
                self.rescue_score_threshold,
                self.rescue_text_threshold,
            )
            result = rescue_results[0]
            selection_mode = "rescue_threshold"
            if len(result["boxes"]) == 0:
                return {
                    "bbox": DEFAULT_FALLBACK.copy(),
                    "candidates": [],
                    "used_fallback": True,
                    "fallback_reason": "no_detection_above_rescue_threshold",
                    "selection_mode": "center_fallback",
                }

        scores = result["scores"]
        order = scores.argsort(descending=True)[: self.top_k].tolist()
        labels = result.get("text_labels", result.get("labels", []))
        candidates = []
        for rank, candidate_idx in enumerate(order, start=1):
            x1, y1, x2, y2 = result["boxes"][candidate_idx].tolist()
            label = ""
            if len(labels) > candidate_idx:
                raw_label = labels[candidate_idx]
                label = str(raw_label.item() if hasattr(raw_label, "item") else raw_label)
            candidates.append(
                {
                    "bbox": [
                        x1 / width,
                        y1 / height,
                        x2 / width,
                        y2 / height,
                    ],
                    "score": float(scores[candidate_idx].item()),
                    "label": label,
                    "rank": rank,
                }
            )
        return {
            "bbox": candidates[0]["bbox"],
            "candidates": candidates,
            "used_fallback": False,
            "selection_mode": selection_mode,
        }

    def _post_process(
        self,
        outputs,
        inputs,
        height,
        width,
        score_threshold,
        text_threshold,
    ):
        """Call either the old or new Transformers GroundingDINO API."""
        import inspect

        function = self.processor.post_process_grounded_object_detection
        parameters = inspect.signature(function).parameters
        kwargs = {
            "text_threshold": text_threshold,
            "target_sizes": [(height, width)],
        }
        if "threshold" in parameters:
            kwargs["threshold"] = score_threshold
        elif "box_threshold" in parameters:
            kwargs["box_threshold"] = score_threshold
        else:
            raise TypeError(
                "Unsupported GroundingDINO post-process signature: "
                f"{inspect.signature(function)}"
            )
        if "input_ids" in parameters:
            kwargs["input_ids"] = inputs["input_ids"]
        return function(outputs, **kwargs)

    def predict(self, sample):
        return self.predict_detailed(sample)["bbox"]


def build_predictor(config):
    """Build a predictor from configs/default.yaml."""
    name = config.get("baseline", {}).get("name", "oracle")
    if name == "oracle":
        return OraclePredictor()
    if name == "center":
        return CenterBoxPredictor()
    if name == "grounding_dino":
        return GroundingDINOPredictor(config)
    raise ValueError(f"Unsupported baseline name: {name}")
