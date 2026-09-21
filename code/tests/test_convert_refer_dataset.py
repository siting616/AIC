import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.convert_refer_dataset import convert, normalize_xywh


class ReferConverterTests(unittest.TestCase):
    def test_xywh_is_normalized_to_xyxy(self):
        self.assertEqual(normalize_xywh([10, 20, 30, 40], 100, 200), [0.1, 0.1, 0.4, 0.3])

    def test_split_filter_and_sentence_expansion(self):
        instances = {
            "images": [{"id": 7, "file_name": "scene.jpg", "width": 100, "height": 200}],
            "annotations": [{"id": 9, "image_id": 7, "bbox": [10, 20, 30, 40]}],
        }
        refs = [
            {"ref_id": 11, "ann_id": 9, "image_id": 7, "split": "val", "sentences": [
                {"sent_id": 21, "sent": "the red object"},
                {"sent_id": 22, "sent": "the object on the left"},
            ]},
            {"ref_id": 12, "ann_id": 9, "image_id": 7, "split": "train", "sentences": [
                {"sent_id": 23, "sent": "excluded"},
            ]},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            converted, skipped = convert(
                instances, refs, temp / "images", temp / "converted" / "val.json",
                "refcoco", ["val"],
            )
        self.assertEqual(skipped, 0)
        self.assertEqual(len(converted), 2)
        record = converted["refcoco_11_21"]
        self.assertEqual(record["bbox"], [0.1, 0.1, 0.4, 0.3])
        self.assertEqual(record["source_modalities"], ["visible"])
        self.assertEqual(record["visible"], record["infrared"])

    def test_cli_converts_json_refs(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "convert_refer_dataset.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            images = temp / "images"
            images.mkdir()
            (images / "scene.jpg").write_bytes(b"fixture")
            instances = temp / "instances.json"
            refs = temp / "refs.json"
            output = temp / "converted" / "val.json"
            instances.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "scene.jpg", "width": 200, "height": 100}],
                "annotations": [{"id": 2, "image_id": 1, "bbox": [20, 10, 40, 20]}],
            }), encoding="utf-8")
            refs.write_text(json.dumps([{
                "ref_id": 3, "ann_id": 2, "image_id": 1, "split": "val",
                "sentences": [{"sent_id": 4, "sent": "the object"}],
            }]), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script), "--instances", str(instances),
                 "--refs", str(refs), "--images-dir", str(images),
                 "--output", str(output), "--split", "val"],
                cwd=project_root, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["refcoco_3_4"]["bbox"], [0.1, 0.1, 0.3, 0.3])


if __name__ == "__main__":
    unittest.main()
