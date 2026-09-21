import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.validation import evaluate_candidate_records, query_categories


class CandidateValidationTests(unittest.TestCase):
    def test_query_categories_are_multilabel(self):
        categories = query_categories("The second person from the left wearing a red shirt")
        self.assertIn("ordinal", categories)
        self.assertIn("horizontal_position", categories)
        self.assertIn("color", categories)
        self.assertIn("person", categories)
        self.assertIn("action_pose", categories)

    def test_oracle_acc_exposes_reranking_headroom(self):
        annotations = {
            "q1": {"query": "red object", "bbox": [0.5, 0.5, 0.8, 0.8]},
            "q2": {"query": "object", "bbox": [0.1, 0.1, 0.3, 0.3]},
        }
        candidates = {
            "q1": {"candidates": [
                {"bbox": [0.0, 0.0, 0.2, 0.2]},
                {"bbox": [0.5, 0.5, 0.8, 0.8]},
            ]},
            "q2": {"candidates": [{"bbox": [0.1, 0.1, 0.3, 0.3]}]},
        }
        result = evaluate_candidate_records(annotations, candidates, top_ks=(1, 2))
        self.assertEqual(result["overall"]["acc_at_05"], 0.5)
        self.assertEqual(result["overall"]["oracle_acc"]["2"], 1.0)

    def test_unlabeled_samples_are_skipped(self):
        result = evaluate_candidate_records(
            {"q1": {"query": "object"}}, {"q1": [0.1, 0.1, 0.2, 0.2]}
        )
        self.assertEqual(result["labeled_samples"], 0)
        self.assertEqual(result["skipped_unlabeled"], 1)

    def test_invalid_candidates_are_reported(self):
        result = evaluate_candidate_records(
            {"q1": {"query": "object", "bbox": [0.1, 0.1, 0.3, 0.3]}},
            {"q1": {"candidates": [
                {"bbox": [0.3, 0.1, 0.2, 0.3]},
                {"bbox": [0.1, 0.1, 0.3, 0.3]},
            ]}},
            top_ks=(1, 2),
        )
        self.assertEqual(result["overall"]["invalid_candidate_rate"], 0.5)
        self.assertEqual(result["overall"]["oracle_acc"]["2"], 1.0)

    def test_cli_writes_json_and_markdown_reports(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "evaluate_candidates.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            annotations = temp / "annotations.json"
            candidates = temp / "candidates.json"
            output = temp / "report"
            annotations.write_text(json.dumps({
                "q1": {"query": "the red object", "bbox": [0.5, 0.5, 0.8, 0.8]}
            }), encoding="utf-8")
            candidates.write_text(json.dumps({
                "q1": {"candidates": [
                    {"bbox": [0.0, 0.0, 0.2, 0.2]},
                    {"bbox": [0.5, 0.5, 0.8, 0.8]},
                ]}
            }), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script), "--annotations", str(annotations),
                 "--candidates", str(candidates), "--output-dir", str(output),
                 "--top-k", "1", "2", "--no-register"],
                cwd=project_root, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["overall"]["acc_at_05"], 0.0)
            self.assertEqual(metrics["overall"]["oracle_acc"]["2"], 1.0)
            self.assertIn("Candidate Validation Report", (output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
