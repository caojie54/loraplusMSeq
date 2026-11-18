import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn

from loraplusmseq.module_selection import CompensationModuleManager
from loraplusmseq.seq_trainer import SequentialLoraPlusMTrainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Module()
        self.proj.base_layer = nn.Linear(3, 2, bias=False)
        self.proj.lora_A = nn.ModuleDict({"default": nn.Linear(3, 1, bias=False)})
        self.proj.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})
        self.proj.scaling = {"default": 1.0}

    def forward(self, input_ids):
        result = self.proj.base_layer(input_ids)
        result = result + self.proj.lora_B["default"](self.proj.lora_A["default"](input_ids))
        return SimpleNamespace(loss=result.square().mean())


class TrainerMemoryAndScoringTest(unittest.TestCase):
    def test_scoring_only_for_alpha_and_memory_for_partial_replay_block(self):
        for method in ("lora", "static_random", "dynamic_random", "alpha"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                model = TinyModel()
                manager = CompensationModuleManager(
                    model, ["proj"], method=method, param_ratio=1.0,
                    alpha_score="lora_effective_update_pressure",
                )
                if method == "alpha":
                    manager.score_from_lora_grads = Mock(wraps=manager.score_from_lora_grads)
                else:
                    manager.score_from_lora_grads = Mock(side_effect=AssertionError("Unused scores computed"))
                trainer = SequentialLoraPlusMTrainer(
                    model=model, train_dataset=[{"input_ids": torch.ones(3)} for _ in range(3)],
                    data_collator=lambda rows: {"input_ids": torch.stack([row["input_ids"] for row in rows])},
                    module_manager=manager, output_dir=tmp, batch_size=1, gradient_accumulation_steps=1,
                    num_train_epochs=1, learning_rate=1e-3, module_learning_rate=1e-3,
                    weight_decay=0, warmup_steps=0, lr_scheduler_type="constant",
                    selection_interval=2, logging_steps=2, seed=0, bf16=False,
                )
                metrics = trainer.train()
                self.assertEqual(manager.score_from_lora_grads.call_count, 3 if method == "alpha" else 0)
                phases = ["lora"] * 3 if method == "lora" else ["lora", "lora", "module", "module", "lora", "module"]
                records = [json.loads(line) for line in (Path(tmp) / "memory_log.jsonl").read_text().splitlines()]
                self.assertEqual([row["phase"] for row in records], phases)
                self.assertEqual([row["step"] for row in records], list(range(1, len(phases) + 1)))
                self.assertEqual(metrics["memory_samples"], metrics["train_updates"])
                self.assertEqual(metrics["memory_by_phase"]["lora"]["memory_samples"], 3)
                if method != "lora":
                    self.assertEqual(metrics["memory_by_phase"]["module"]["memory_samples"], 3)
                self.assertEqual(metrics["max_memory_gib"], 0)


if __name__ == "__main__":
    unittest.main()
