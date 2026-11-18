import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from loraplusmseq.memory_stats import CudaMemoryTracker


class MemoryStatsTest(unittest.TestCase):
    def test_phase_averages_include_every_update_and_reset_peak(self):
        gib = 1024**3
        with tempfile.TemporaryDirectory() as tmp, \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.reset_peak_memory_stats") as reset, \
             patch("torch.cuda.max_memory_allocated", side_effect=[8*gib, 4*gib, 6*gib]), \
             patch("torch.cuda.memory_allocated") as current_memory:
            path = Path(tmp) / "memory_log.jsonl"
            tracker = CudaMemoryTracker("cuda:1", path)
            for step, phase in enumerate(("lora", "module", "lora"), start=1):
                tracker.begin(phase, step)
                tracker.end()
            self.assertEqual(reset.call_count, 3)
            self.assertTrue(all(str(call.args[0]) == "cuda:1" for call in reset.call_args_list))
            summary = tracker.metrics()
            self.assertEqual(summary["memory_samples"], 3)
            self.assertEqual(summary["max_memory_gib"], 8)
            self.assertEqual(summary["peak_memory_gib"], 6)
            self.assertEqual(summary["avg_peak_memory_gib"], 6)
            self.assertEqual(summary["memory_by_phase"]["lora"]["avg_peak_memory_gib"], 7)
            self.assertEqual(summary["memory_by_phase"]["module"]["max_memory_gib"], 4)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["step"] for row in records], [1, 2, 3])
            self.assertEqual(records[-1]["phase_memory"]["memory_samples"], 2)
            current_memory.assert_not_called()
            self.assertNotIn("current_memory_gib", json.dumps(summary))
            self.assertNotIn("current_memory_gib", path.read_text())

    def test_cpu_and_cancelled_window(self):
        with patch("torch.cuda.reset_peak_memory_stats") as reset:
            tracker = CudaMemoryTracker("cpu")
            tracker.begin("module", 1)
            tracker.cancel()
            self.assertEqual(tracker.summary()["memory_samples"], 0)
            tracker.begin("misa", 1)
            tracker.end()
            self.assertEqual(tracker.summary()["max_memory_gib"], 0)
            self.assertEqual(tracker.summary()["memory_samples"], 1)
            reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
