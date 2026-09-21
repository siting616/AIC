"""Dataset utilities for the AIC multimodal visual grounding task."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List


class AICDataset:
    """Read AIC samples from a JSON annotation file."""

    def __init__(self, root_dir: str, json_path: str, images_dir: str | None = None):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.images_dir = Path(images_dir).expanduser().resolve() if images_dir else None
        self.json_path = Path(json_path).expanduser()
        if not self.json_path.is_absolute():
            self.json_path = (self.root_dir / self.json_path).resolve()

        if not self.json_path.exists():
            raise FileNotFoundError(f"Annotation JSON not found: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as f:
            self.data: Dict[str, Dict[str, Any]] = json.load(f)

        if not isinstance(self.data, dict):
            raise ValueError(f"Annotation JSON must be an object: {self.json_path}")

        self.sample_ids: List[str] = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[idx]
        item = self.data[sample_id]

        for key in ("visible", "infrared", "depth", "query"):
            if key not in item:
                raise KeyError(f"Sample {sample_id} is missing required field: {key}")

        visible_path = self._resolve_path(item["visible"])
        infrared_path = self._resolve_path(item["infrared"])
        depth_path = self._resolve_path(item["depth"])

        for name, path in (
            ("visible", visible_path),
            ("infrared", infrared_path),
            ("depth", depth_path),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} image not found for sample {sample_id}: {path}"
                )

        return {
            "sample_id": sample_id,
            "visible_path": str(visible_path),
            "infrared_path": str(infrared_path),
            "depth_path": str(depth_path),
            "query": item["query"],
            "bbox": item.get("bbox"),
        }

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute() and (path.exists() or self.images_dir is None):
            return path
        rooted = path.resolve() if path.is_absolute() else (self.root_dir / path).resolve()
        if rooted.exists() or self.images_dir is None:
            return rooted
        # A Windows absolute path is not absolute to pathlib on Linux. Extract
        # its basename explicitly before remapping to the cloud image folder.
        filename = PureWindowsPath(str(path_value)).name
        mapped = self.images_dir / filename
        return mapped.resolve()

    def ground_truth_dict(self) -> Dict[str, List[float]]:
        return {
            sample_id: item.get("bbox")
            for sample_id, item in self.data.items()
        }
