import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "MedReason" / "evaluation"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MedReasonResultsDirectoryTest(unittest.TestCase):
    def test_default_relative_and_absolute_results_roots(self):
        scorer = load_module("medreason_scorer_test", EVALUATION_DIR / "scorer.py")
        with patch.dict("sys.modules", {"scorer": scorer}):
            evaluator = load_module("medreason_eval_test", EVALUATION_DIR / "eval.py")
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                data = Path(tmp) / "sample.jsonl"
                data.write_text("\n".join(json.dumps({
                    "question": "Question?", "options": {"A": "Correct", "B": "Wrong"},
                    "answer_idx": "A",
                }) for _ in range(2)))
                for root in (None, "custom_logs_root", str(Path(tmp) / "absolute_logs_root")):
                    with self.subTest(results_dir=root):
                        output = Path(root or "results") / "suite/run"
                        # An existing task directory may not have logs/result yet.
                        output.mkdir(parents=True)
                        argv = ["eval.py", "--eval_file", str(data), "--model_name", "test-model",
                                "--no-use_chat_template", "--task_floder", "suite/run"]
                        if root is not None:
                            argv += ["--results_dir", root]
                        def complete(**kwargs):
                            return SimpleNamespace(choices=[SimpleNamespace(text="The answer is A.")
                                                            for _ in kwargs["prompt"]])
                        client = SimpleNamespace(completions=SimpleNamespace(create=complete))
                        with patch("sys.argv", argv), patch.object(evaluator.openai, "Client", return_value=client):
                            evaluator.main()
                        logs = list((output / "logs").glob("*.json"))
                        results = list((output / "result").glob("*.json"))
                        self.assertEqual(len(logs), 1)
                        self.assertEqual(len(results), 1)
                        self.assertEqual(len(json.loads(logs[0].read_text())), 2)
                        self.assertEqual(json.loads(results[0].read_text())["normal"][0], 1.0)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
