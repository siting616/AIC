import json
import tempfile
import unittest
from pathlib import Path

from src.dataset import AICDataset


class DatasetPathRemapTests(unittest.TestCase):
    def test_windows_paths_remap_to_cloud_image_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            images.mkdir()
            for name in ("visible.jpg", "infrared.jpg", "depth.jpg"):
                (images / name).write_bytes(b"fixture")
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps({"q": {
                "visible": r"C:\data\visible.jpg",
                "infrared": r"C:\data\infrared.jpg",
                "depth": r"C:\data\depth.jpg",
                "query": "object",
            }}), encoding="utf-8")
            dataset = AICDataset(str(root), str(annotations), str(images))
            sample = dataset[0]
            self.assertEqual(Path(sample["visible_path"]), (images / "visible.jpg").resolve())


if __name__ == "__main__":
    unittest.main()
