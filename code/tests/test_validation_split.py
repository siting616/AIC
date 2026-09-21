import csv
import tempfile
import unittest
from pathlib import Path

from src.experiment_registry import append_registry
from src.validation_split import (
    create_grouped_split, dataset_fingerprint, image_group_key,
    subset_annotations, validate_split_manifest,
)


def fixture_annotations():
    data = {}
    for image_id in range(10):
        for query_index in range(3):
            data[f"q{image_id}_{query_index}"] = {
                "source_dataset": "fixture",
                "source_image_id": image_id,
                "visible": f"images/{image_id}.jpg",
                "query": f"object {query_index}",
                "bbox": [0.1, 0.1, 0.2, 0.2],
            }
    return data


class ValidationSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_has_no_image_leakage(self):
        annotations = fixture_annotations()
        first = create_grouped_split(annotations, 0.3, 42)
        second = create_grouped_split(annotations, 0.3, 42)
        self.assertEqual(first["splits"], second["splits"])
        self.assertTrue(validate_split_manifest(annotations, first))
        dev = subset_annotations(annotations, first, "dev")
        holdout = subset_annotations(annotations, first, "holdout")
        dev_groups = {image_group_key(k, v) for k, v in dev.items()}
        holdout_groups = {image_group_key(k, v) for k, v in holdout.items()}
        self.assertFalse(dev_groups & holdout_groups)
        self.assertEqual(len(holdout), 9)

    def test_manifest_rejects_changed_dataset(self):
        annotations = fixture_annotations()
        manifest = create_grouped_split(annotations, 0.3, 42)
        annotations["q0_0"]["query"] = "changed"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_split_manifest(annotations, manifest)

    def test_fingerprint_ignores_dictionary_order(self):
        annotations = fixture_annotations()
        reversed_annotations = dict(reversed(list(annotations.items())))
        self.assertEqual(dataset_fingerprint(annotations), dataset_fingerprint(reversed_annotations))

    def test_registry_writes_one_header_and_multiple_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiments.csv"
            append_registry(path, {"experiment": "a", "split": "dev"})
            append_registry(path, {"experiment": "b", "split": "holdout"})
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["experiment"] for row in rows], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
