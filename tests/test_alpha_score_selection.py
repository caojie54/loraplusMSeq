import unittest

import torch
from torch import nn

from loraplusmseq.module_selection import CompensationModuleManager


class TinyLoraLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(3, 4, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
        self.scaling = {"default": 0.5}


class AlphaScoreSelectionTest(unittest.TestCase):
    def make_manager(self, score, candidate_ratio=0.0):
        model = nn.ModuleDict({name: TinyLoraLinear() for name in ("low", "mid", "high")})
        return CompensationModuleManager(
            model, ["low", "mid", "high"], method="alpha",
            total_model_params=36, param_ratio=1 / 3,
            alpha_score=score, alpha_candidate_ratio=candidate_ratio,
            alpha_uniform_mix=0.0, seed=0,
        )

    def test_effective_pressure_matches_explicit_dense_update(self):
        torch.manual_seed(0)
        manager = self.make_manager("lora_effective_update_pressure")
        for candidate in manager.candidates:
            module = candidate.module
            a = module.lora_A["default"].weight
            b = module.lora_B["default"].weight
            for missing_grad in (None, "a", "b"):
                a.grad = None if missing_grad == "a" else torch.randn_like(a)
                b.grad = None if missing_grad == "b" else torch.randn_like(b)
                ga = torch.zeros_like(a) if a.grad is None else a.grad
                gb = torch.zeros_like(b) if b.grad is None else b.grad
                expected = (0.5 * (gb @ a + b @ ga)).square().sum().item()
                for score in ("lora_effective_update_pressure", "lora_effective_update_pressure_min"):
                    manager.alpha_score = score
                    actual = manager.score_from_lora_grads()[candidate.name]
                    self.assertAlmostEqual(actual, expected, delta=max(1e-6, expected * 1e-5))

    def test_ranked_selection_and_history_reverse_for_min(self):
        scores = {"low": 1.0, "mid": 10.0, "high": 100.0}
        for base in ("lora_grad_norm", "lora_effective_update_pressure"):
            for suffix, expected, order in (
                ("", "high", ["high", "mid", "low"]),
                ("_min", "low", ["low", "mid", "high"]),
            ):
                with self.subTest(score=base + suffix):
                    manager = self.make_manager(base + suffix)
                    self.assertEqual(manager.select_by_alpha_scores(scores, step=1), [expected])
                    record = manager.selection_records[-1]
                    self.assertEqual(list(record["scores"]), order)
                    self.assertEqual(record["reason"], "seq_alpha_min" if suffix else "seq_alpha")

    def test_sampled_selection_uses_low_score_pool_for_min(self):
        scores = {"low": 1.0, "mid": 10.0, "high": 100.0}
        for base in ("lora_grad_norm", "lora_effective_update_pressure"):
            for suffix, expected_pool in (("", ["high", "mid"]), ("_min", ["low", "mid"])):
                with self.subTest(score=base + suffix):
                    manager = self.make_manager(base + suffix, candidate_ratio=2 / 3)
                    selected = manager.select_by_alpha_scores(scores, step=1)
                    record = manager.selection_records[-1]
                    sampling = record["alpha_sampling"]
                    pool = sampling["pool_top"]
                    self.assertEqual([item["name"] for item in pool], expected_pool)
                    self.assertGreater(pool[0]["norm_score"], pool[1]["norm_score"])
                    self.assertEqual(len(selected), 1)
                    self.assertIn(selected[0], expected_pool)
                    self.assertEqual(sampling["used_params"], 12)
                    self.assertEqual(sampling["score_order"], "ascending" if suffix else "descending")
                    self.assertEqual(record["reason"], "seq_alpha_min_sampled" if suffix else "seq_alpha_sampled")


if __name__ == "__main__":
    unittest.main()
