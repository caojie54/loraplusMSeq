"""Candidate discovery, phase control, and LoRA-pressure module selection."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn


@dataclass
class CandidateModule:
    """A PEFT LoRA linear module whose frozen base weight can be trained."""

    name: str
    short_name: str
    num_params: int
    weight_shape: List[int]
    module: nn.Module

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "short_name": self.short_name,
            "num_params": self.num_params,
            "weight_shape": self.weight_shape,
        }


def _matched_target(module_name: str, target_modules: Sequence[str]) -> Optional[str]:
    for target in target_modules:
        if module_name == target or module_name.endswith(f".{target}"):
            return target
    return None


def _target_match(module_name: str, target_modules: Sequence[str]) -> bool:
    return _matched_target(module_name, target_modules) is not None


def _is_peft_lora_linear(module: nn.Module) -> bool:
    base_layer = getattr(module, "base_layer", None)
    return (
        base_layer is not None
        and isinstance(base_layer, nn.Linear)
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
    )


def collect_candidate_modules(model: nn.Module, target_modules: Sequence[str]) -> List[CandidateModule]:
    """Collect LoRA-wrapped Linear modules whose original weights may compensate LoRA."""
    candidates: List[CandidateModule] = []
    for module_name, module in model.named_modules():
        matched_target = _matched_target(module_name, target_modules)
        if matched_target is None or not _is_peft_lora_linear(module):
            continue
        weight = module.base_layer.weight
        num_params = weight.numel()
        if getattr(module.base_layer, "bias", None) is not None:
            num_params += module.base_layer.bias.numel()
        candidates.append(
            CandidateModule(
                name=module_name,
                short_name=matched_target,
                num_params=num_params,
                weight_shape=list(weight.shape),
                module=module,
            )
        )
    return candidates


class CompensationModuleManager:
    """Owns compensation candidates and switches between LoRA and module phases."""

    def __init__(
        self,
        model: nn.Module,
        target_modules: Sequence[str],
        method: str,
        param_ratio: float = 0.005,
        total_model_params: Optional[int] = None,
        seed: int = 0,
        alpha_score: str = "lora_update_ratio",
        alpha_candidate_ratio: float = 0.0,
        alpha_sampling_temperature: float = 1.0,
        alpha_uniform_mix: float = 0.1,
        alpha_score_gamma: float = 1.0,
        alpha_group_norm: str = "none",
        history_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.target_modules = list(target_modules)
        self.method = method
        self.param_ratio = param_ratio
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.alpha_score = alpha_score
        self.alpha_candidate_ratio = float(alpha_candidate_ratio)
        self.alpha_sampling_temperature = float(alpha_sampling_temperature)
        self.alpha_uniform_mix = float(alpha_uniform_mix)
        self.alpha_score_gamma = float(alpha_score_gamma)
        self.alpha_group_norm = alpha_group_norm
        self.history_path = history_path
        self.candidates = collect_candidate_modules(model, target_modules)
        self.candidate_by_name = {candidate.name: candidate for candidate in self.candidates}
        self.managed_param_ids: Set[int] = set()
        self.selected_names: List[str] = []
        self.selection_step = -1
        self.selection_records: List[Dict[str, object]] = []
        self.total_candidate_params = sum(candidate.num_params for candidate in self.candidates)
        self.total_model_params = (
            int(total_model_params) if total_model_params is not None else self._count_original_model_params()
        )
        if self.enabled and (not math.isfinite(self.param_ratio) or self.param_ratio <= 0):
            raise ValueError("compensation_ratio must be a positive finite value when module replay is enabled.")
        if self.alpha_candidate_ratio < 0 or not math.isfinite(self.alpha_candidate_ratio):
            raise ValueError("alpha_candidate_ratio must be a non-negative finite value.")
        if self.alpha_candidate_ratio and self.alpha_candidate_ratio < self.param_ratio:
            raise ValueError("alpha_candidate_ratio must be >= compensation_ratio when enabled.")
        if self.alpha_sampling_temperature <= 0 or not math.isfinite(self.alpha_sampling_temperature):
            raise ValueError("alpha_sampling_temperature must be a positive finite value.")
        if not 0.0 <= self.alpha_uniform_mix <= 1.0:
            raise ValueError("alpha_uniform_mix must be in [0, 1].")
        if self.alpha_group_norm not in {"none", "global", "type"}:
            raise ValueError("alpha_group_norm must be one of: none, global, type.")
        self._register_managed_params()
        self.record_selection([], step=-1, reason="init")

    @property
    def enabled(self) -> bool:
        return self.method != "lora"

    def _register_managed_params(self) -> None:
        for candidate in self.candidates:
            base_layer = candidate.module.base_layer
            self.managed_param_ids.add(id(base_layer.weight))
            if getattr(base_layer, "bias", None) is not None:
                self.managed_param_ids.add(id(base_layer.bias))

    def _count_original_model_params(self) -> int:
        total = int(sum(param.numel() for name, param in self.model.named_parameters() if "lora_" not in name))
        if total > 0:
            return total
        return int(sum(param.numel() for param in self.model.parameters()))

    def save_candidates(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_modules": self.target_modules,
                    "method": self.method,
                    "param_ratio": self.param_ratio,
                    "alpha_candidate_ratio": self.alpha_candidate_ratio,
                    "alpha_sampling_temperature": self.alpha_sampling_temperature,
                    "alpha_uniform_mix": self.alpha_uniform_mix,
                    "alpha_score_gamma": self.alpha_score_gamma,
                    "alpha_group_norm": self.alpha_group_norm,
                    "ratio_denominator": "total_model_params",
                    "total_model_params": self.total_model_params,
                    "total_candidate_params": self.total_candidate_params,
                    "candidates": [candidate.to_json() for candidate in self.candidates],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def is_managed_param(self, param: nn.Parameter) -> bool:
        return id(param) in self.managed_param_ids

    def lora_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [(name, param) for name, param in self.model.named_parameters() if "lora_" in name]

    def selected_module_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        selected = set(self.selected_names)
        params: List[Tuple[str, nn.Parameter]] = []
        for candidate in self.candidates:
            if candidate.name not in selected:
                continue
            base_layer = candidate.module.base_layer
            params.append((f"{candidate.name}.base_layer.weight", base_layer.weight))
            if getattr(base_layer, "bias", None) is not None:
                params.append((f"{candidate.name}.base_layer.bias", base_layer.bias))
        return params

    def module_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        params: List[Tuple[str, nn.Parameter]] = []
        for candidate in self.candidates:
            base_layer = candidate.module.base_layer
            params.append((f"{candidate.name}.base_layer.weight", base_layer.weight))
            if getattr(base_layer, "bias", None) is not None:
                params.append((f"{candidate.name}.base_layer.bias", base_layer.bias))
        return params

    def selected_lora_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        selected = set(self.selected_names)
        params: List[Tuple[str, nn.Parameter]] = []
        for candidate in self.candidates:
            if candidate.name not in selected:
                continue
            lora_a = getattr(candidate.module, "lora_A", None)
            lora_b = getattr(candidate.module, "lora_B", None)
            if lora_a is not None:
                for adapter_name, layer in lora_a.items():
                    params.append((f"{candidate.name}.lora_A.{adapter_name}.weight", layer.weight))
            if lora_b is not None:
                for adapter_name, layer in lora_b.items():
                    params.append((f"{candidate.name}.lora_B.{adapter_name}.weight", layer.weight))
        return params

    def set_lora_phase(self) -> None:
        """Train only LoRA parameters while all original compensation weights are frozen."""
        for name, param in self.model.named_parameters():
            param.requires_grad_("lora_" in name)
            if "lora_" not in name:
                param.grad = None

    def set_module_phase(self) -> None:
        """Train selected original weights only; LoRA remains active in forward but frozen."""
        selected = set(self.selected_names)
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(False)
                param.grad = None

        for candidate in self.candidates:
            trainable = candidate.name in selected
            base_layer = candidate.module.base_layer
            base_layer.weight.requires_grad_(trainable)
            if not trainable:
                base_layer.weight.grad = None
            if getattr(base_layer, "bias", None) is not None:
                base_layer.bias.requires_grad_(trainable)
                if not trainable:
                    base_layer.bias.grad = None

    def freeze_all_compensation(self) -> None:
        for candidate in self.candidates:
            base_layer = candidate.module.base_layer
            base_layer.weight.requires_grad_(False)
            base_layer.weight.grad = None
            if getattr(base_layer, "bias", None) is not None:
                base_layer.bias.requires_grad_(False)
                base_layer.bias.grad = None

    def clear_all_grads(self) -> None:
        for param in self.model.parameters():
            param.grad = None

    def record_selection(
        self,
        selected_names: Iterable[str],
        step: int,
        reason: str,
        scores: Optional[Dict[str, float]] = None,
        extra_info: Optional[Dict[str, object]] = None,
    ) -> List[str]:
        self.selected_names = [name for name in selected_names if name in self.candidate_by_name]
        self.selection_step = step
        self._record_selection(step=step, reason=reason, scores=scores, extra_info=extra_info)
        return self.selected_names

    def select_random(self, step: int, reason: str) -> List[str]:
        selected = self._choose_random_by_budget()
        return self.record_selection([candidate.name for candidate in selected], step=step, reason=reason)

    def select_by_alpha_scores(self, scores: Dict[str, float], step: int) -> List[str]:
        select_min = self.alpha_score == "lora_grad_norm_min"
        if self._use_alpha_sampling():
            selected, extra_info = self._select_alpha_sampled(scores, select_min=select_min)
            return self.record_selection(
                [candidate.name for candidate in selected],
                step=step,
                reason="seq_alpha_min_sampled" if select_min else "seq_alpha_sampled",
                scores=scores,
                extra_info=extra_info,
            )

        missing_score = float("inf") if select_min else float("-inf")
        ranked = sorted(
            self.candidates,
            key=lambda candidate: scores.get(candidate.name, missing_score),
            reverse=not select_min,
        )
        selected = self._choose_ranked_by_budget(ranked)
        return self.record_selection(
            [candidate.name for candidate in selected],
            step=step,
            reason="seq_alpha_min" if select_min else "seq_alpha",
            scores=scores,
        )

    def _choose_random_by_budget(self) -> List[CandidateModule]:
        """Uniform random baseline: sample without replacement instead of taking model order."""
        if not self.candidates:
            return []
        shuffled_candidates = list(self.candidates)
        self.rng.shuffle(shuffled_candidates)
        return self._choose_ranked_by_budget(shuffled_candidates)

    def _choose_ranked_by_budget(self, ranked_candidates: Sequence[CandidateModule]) -> List[CandidateModule]:
        if not ranked_candidates:
            return []
        if not math.isfinite(self.param_ratio) or self.param_ratio <= 0:
            raise ValueError("compensation_ratio must be a positive finite value.")

        budget = self._budget_for_ratio(self.param_ratio)
        return self._choose_ranked_by_budget_limit(ranked_candidates, budget)

    def _budget_for_ratio(self, ratio: float) -> int:
        return max(1, int(self.total_model_params * ratio))

    def _choose_ranked_by_budget_limit(
        self,
        ranked_candidates: Sequence[CandidateModule],
        budget: int,
    ) -> List[CandidateModule]:
        selected: List[CandidateModule] = []
        used = 0
        for candidate in ranked_candidates:
            if selected and used + candidate.num_params > budget:
                continue
            selected.append(candidate)
            used += candidate.num_params
            if used >= budget:
                break
        return selected or ([ranked_candidates[0]] if ranked_candidates else [])

    def _use_alpha_sampling(self) -> bool:
        return (
            self.alpha_candidate_ratio > self.param_ratio
            and self.method == "alpha"
            and bool(self.candidates)
        )

    def _select_alpha_sampled(
        self,
        scores: Dict[str, float],
        select_min: bool = False,
    ) -> Tuple[List[CandidateModule], Dict[str, object]]:
        items = self._normalized_alpha_items(scores, select_min=select_min)
        if not items:
            return [], {"alpha_sampling": {"enabled": True, "empty_scores": True}}

        budget1 = self._budget_for_ratio(self.param_ratio)
        budget2 = self._budget_for_ratio(self.alpha_candidate_ratio)
        pool, pool_params = self._top_alpha_pool(items, budget2)
        sample_seed = self.rng.randrange(0, 2**63 - 1)
        selected_items, used_params = self._sequential_budgeted_sample(pool, budget1, sample_seed)
        selected = [item["candidate"] for item in selected_items]
        extra_info = {
            "alpha_sampling": {
                "enabled": True,
                "strategy": "ratio2_softmax_uniform_sequential_budget",
                "budget": int(budget1),
                "candidate_budget": int(budget2),
                "used_params": int(used_params),
                "pool_params": int(pool_params),
                "budget_fill_ratio": float(used_params / max(1, budget1)),
                "pool_fill_ratio": float(pool_params / max(1, budget2)),
                "candidate_ratio": float(self.alpha_candidate_ratio),
                "selection_ratio": float(self.param_ratio),
                "temperature": float(self.alpha_sampling_temperature),
                "uniform_mix": float(self.alpha_uniform_mix),
                "score_gamma": float(self.alpha_score_gamma),
                "group_norm": self.alpha_group_norm,
                "score_order": "ascending" if select_min else "descending",
                "score_transform": "-log(score)" if select_min else "log(score)",
                "sample_seed": int(sample_seed),
                "pool_count": int(len(pool)),
                "selected_count": int(len(selected_items)),
                "selected": [self._alpha_item_record(item, include_sampling=True) for item in selected_items],
                "pool_top": [self._alpha_item_record(item, include_sampling=False) for item in pool[:20]],
            }
        }
        return selected, extra_info

    def _normalized_alpha_items(
        self,
        scores: Dict[str, float],
        select_min: bool = False,
    ) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        eps = 1e-30
        has_positive_score = False
        for candidate in self.candidates:
            raw_score = float(scores.get(candidate.name, 0.0))
            if not math.isfinite(raw_score) or raw_score < 0.0:
                raw_score = 0.0
            has_positive_score = has_positive_score or raw_score > 0.0
            score_term = -math.log(raw_score + eps) if select_min else math.log(raw_score + eps)
            base_score = score_term - self.alpha_score_gamma * math.log(max(1, candidate.num_params))
            items.append(
                {
                    "candidate": candidate,
                    "name": candidate.name,
                    "short_name": candidate.short_name,
                    "raw_score": raw_score,
                    "num_params": int(candidate.num_params),
                    "base_score": float(base_score),
                    "norm_score": float(base_score),
                }
            )

        if not has_positive_score:
            for item in items:
                item["base_score"] = 0.0
                item["norm_score"] = 0.0
            return items

        if self.alpha_group_norm == "global":
            self._apply_alpha_group_norm(items, group_key=None)
        elif self.alpha_group_norm == "type":
            self._apply_alpha_group_norm(items, group_key="short_name")
        return items

    def _apply_alpha_group_norm(self, items: List[Dict[str, object]], group_key: Optional[str]) -> None:
        groups: Dict[str, List[float]] = {}
        for item in items:
            group = "all" if group_key is None else str(item[group_key])
            groups.setdefault(group, []).append(float(item["base_score"]))

        stats: Dict[str, Tuple[float, float]] = {}
        for group, values in groups.items():
            mean = sum(values) / max(1, len(values))
            if len(values) > 1:
                var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                std = math.sqrt(max(var, 1e-12))
            else:
                std = 1.0
            stats[group] = (mean, std)

        for item in items:
            group = "all" if group_key is None else str(item[group_key])
            mean, std = stats[group]
            z_score = (float(item["base_score"]) - mean) / std
            item["norm_score"] = float(max(-3.0, min(3.0, z_score)))

    def _top_alpha_pool(
        self,
        items: List[Dict[str, object]],
        budget: int,
    ) -> Tuple[List[Dict[str, object]], int]:
        sorted_items = sorted(items, key=lambda item: float(item["norm_score"]), reverse=True)
        pool: List[Dict[str, object]] = []
        used = 0
        for item in sorted_items:
            num_params = int(item["num_params"])
            if pool and used + num_params > budget:
                continue
            pool.append(item)
            used += num_params
            if used >= budget:
                break
        return pool or sorted_items[:1], used if pool else (int(sorted_items[0]["num_params"]) if sorted_items else 0)

    def _sequential_budgeted_sample(
        self,
        pool: List[Dict[str, object]],
        budget: int,
        seed: int,
    ) -> Tuple[List[Dict[str, object]], int]:
        if not pool or budget <= 0:
            return [], 0

        scores_tensor = torch.tensor([float(item["norm_score"]) for item in pool], dtype=torch.float64, device="cpu")
        costs = torch.tensor([int(item["num_params"]) for item in pool], dtype=torch.long, device="cpu")
        importance_probs = torch.softmax(scores_tensor / max(self.alpha_sampling_temperature, 1e-8), dim=0)
        uniform_mix = min(max(self.alpha_uniform_mix, 0.0), 1.0)
        mixed_probs = (1.0 - uniform_mix) * importance_probs + uniform_mix / max(1, len(pool))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        remaining = torch.ones(len(pool), dtype=torch.bool, device="cpu")

        selected: List[Dict[str, object]] = []
        used = 0
        while True:
            remaining_budget = budget - used
            feasible_mask = remaining & (costs <= remaining_budget)
            feasible_indices = torch.nonzero(feasible_mask, as_tuple=False).flatten()
            if feasible_indices.numel() == 0:
                break

            feasible_weights = mixed_probs[feasible_indices]
            feasible_weight_sum = feasible_weights.sum()
            if feasible_weight_sum <= 0:
                feasible_weights = torch.ones_like(feasible_weights) / feasible_weights.numel()
            else:
                feasible_weights = feasible_weights / feasible_weight_sum
            sampled_pos = torch.multinomial(
                feasible_weights,
                num_samples=1,
                replacement=False,
                generator=generator,
            ).item()
            selected_index = int(feasible_indices[sampled_pos].item())
            item = dict(pool[selected_index])
            item["importance_prob"] = float(importance_probs[selected_index].item())
            item["mixed_prob"] = float(mixed_probs[selected_index].item())
            item["conditional_draw_prob"] = float(feasible_weights[sampled_pos].item())
            item["remaining_budget_before_draw"] = int(remaining_budget)
            selected.append(item)
            used += int(item["num_params"])
            remaining[selected_index] = False

        if not selected and pool:
            item = dict(max(pool, key=lambda value: float(value["norm_score"])))
            item["importance_prob"] = 1.0
            item["mixed_prob"] = 1.0
            item["conditional_draw_prob"] = 1.0
            item["remaining_budget_before_draw"] = int(budget)
            selected.append(item)
            used = int(item["num_params"])
        return selected, used

    @staticmethod
    def _alpha_item_record(item: Dict[str, object], include_sampling: bool) -> Dict[str, object]:
        record: Dict[str, object] = {
            "name": str(item["name"]),
            "short_name": str(item["short_name"]),
            "raw_score": float(item["raw_score"]),
            "base_score": float(item["base_score"]),
            "norm_score": float(item["norm_score"]),
            "num_params": int(item["num_params"]),
        }
        if include_sampling:
            record.update(
                {
                    "importance_prob": float(item.get("importance_prob", 0.0)),
                    "mixed_prob": float(item.get("mixed_prob", 0.0)),
                    "conditional_draw_prob": float(item.get("conditional_draw_prob", 0.0)),
                    "remaining_budget_before_draw": int(item.get("remaining_budget_before_draw", 0)),
                }
            )
        return record

    def _record_selection(
        self,
        step: int,
        reason: str,
        scores: Optional[Dict[str, float]] = None,
        extra_info: Optional[Dict[str, object]] = None,
    ) -> None:
        selected_params = sum(self.candidate_by_name[name].num_params for name in self.selected_names)
        record: Dict[str, object] = {
            "step": int(step),
            "reason": reason,
            "selected_names": list(self.selected_names),
            "selected_short_names": [self.candidate_by_name[name].short_name for name in self.selected_names],
            "selected_params": int(selected_params),
            "selected_param_ratio": float(selected_params / max(1, self.total_model_params)),
            "selected_candidate_param_ratio": float(selected_params / max(1, self.total_candidate_params)),
            "ratio_denominator": "total_model_params",
        }
        if extra_info:
            record.update(extra_info)
        if scores is not None:
            reverse_scores = self.alpha_score != "lora_grad_norm_min"
            record["scores"] = {
                name: float(score)
                for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=reverse_scores)[:50]
            }
        self.selection_records.append(record)
        self._append_history(record)
        print(
            f"[loraplusMSeq] step={step} reason={reason} selected={len(self.selected_names)} "
            f"params={selected_params} ratio={record['selected_param_ratio']:.6f}",
            flush=True,
        )
        sampling_info = record.get("alpha_sampling")
        if isinstance(sampling_info, dict) and sampling_info.get("enabled"):
            print(
                "[loraplusMSeq] alpha sampling "
                f"pool={sampling_info.get('pool_count')} pool_params={sampling_info.get('pool_params')} "
                f"budget={sampling_info.get('budget')} used={sampling_info.get('used_params')} "
                f"tau={sampling_info.get('temperature')} uniform={sampling_info.get('uniform_mix')}",
                flush=True,
            )
        for name in self.selected_names[:20]:
            print(f"[loraplusMSeq]   {name}", flush=True)

    def _append_history(self, record: Dict[str, object]) -> None:
        if not self.history_path:
            return
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def score_from_lora_grads(self) -> Dict[str, float]:
        """Score modules by LoRA gradient pressure from the current effective batch."""
        scores: Dict[str, float] = {}
        for candidate in self.candidates:
            score = self._score_one_lora_module(candidate.module, candidate.num_params)
            if math.isfinite(score):
                scores[candidate.name] = score
        return scores

    def _score_one_lora_module(self, module: nn.Module, base_num_params: int) -> float:
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if not lora_a or not lora_b:
            return 0.0

        total = 0.0
        eps = 1e-6
        for adapter_name in list(lora_a.keys()):
            a_weight = lora_a[adapter_name].weight
            b_weight = lora_b[adapter_name].weight
            a_grad = a_weight.grad
            b_grad = b_weight.grad
            if a_grad is None and b_grad is None:
                continue

            if self.alpha_score in {"lora_grad_norm", "lora_grad_norm_min"}:
                a_term = torch.norm(a_grad.detach().float()).item() if a_grad is not None else 0.0
                b_term = torch.norm(b_grad.detach().float()).item() if b_grad is not None else 0.0
                denom = math.sqrt(max(1, a_weight.numel() + b_weight.numel()))
                total += math.sqrt(a_term * a_term + b_term * b_term) / denom
            elif self.alpha_score == "lora_effective_update_pressure":
                scaling = self._lora_scaling(module, adapter_name)
                total += self._lora_effective_update_energy(
                    a_weight=a_weight,
                    b_weight=b_weight,
                    a_grad=a_grad,
                    b_grad=b_grad,
                    scaling=scaling,
                ).item()
            elif self.alpha_score == "lora_update_ratio":
                a_term = 0.0
                b_term = 0.0
                if a_grad is not None:
                    a_term = torch.norm(a_grad.detach().float()).item() / (
                        torch.norm(a_weight.detach().float()).item() + eps
                    )
                if b_grad is not None:
                    b_term = torch.norm(b_grad.detach().float()).item() / (
                        torch.norm(b_weight.detach().float()).item() + eps
                    )
                total += a_term + b_term
            else:
                raise ValueError(f"Unsupported alpha_score: {self.alpha_score}")

        if self.alpha_score in {"lora_grad_norm", "lora_grad_norm_min", "lora_effective_update_pressure"}:
            return float(total)
        return float(total * math.log(max(2, base_num_params)))

    @staticmethod
    def _lora_scaling(module: nn.Module, adapter_name: str) -> float:
        scaling = getattr(module, "scaling", 1.0)
        if isinstance(scaling, dict):
            scaling = scaling.get(adapter_name, 1.0)
        return float(scaling)

    @staticmethod
    @torch.no_grad()
    def _lora_effective_update_energy(
        a_weight: torch.Tensor,
        b_weight: torch.Tensor,
        a_grad: Optional[torch.Tensor],
        b_grad: Optional[torch.Tensor],
        scaling: float,
    ) -> torch.Tensor:
        if a_grad is None and b_grad is None:
            return torch.zeros((), device=a_weight.device, dtype=torch.float32)

        a = a_weight.detach().float()
        b = b_weight.detach().float()
        grad_a = a_grad.detach().float() if a_grad is not None else None
        grad_b = b_grad.detach().float() if b_grad is not None else None
        energy = torch.zeros((), device=a_weight.device, dtype=torch.float32)

        if grad_b is not None:
            grad_b_t_grad_b = grad_b.t().matmul(grad_b)
            a_a_t = a.matmul(a.t())
            energy = energy + (grad_b_t_grad_b * a_a_t).sum()

        if grad_a is not None:
            b_t_b = b.t().matmul(b)
            grad_a_grad_a_t = grad_a.matmul(grad_a.t())
            energy = energy + (b_t_b * grad_a_grad_a_t).sum()

        if grad_a is not None and grad_b is not None:
            grad_b_t_b = grad_b.t().matmul(b)
            grad_a_a_t = grad_a.matmul(a.t())
            energy = energy + 2.0 * (grad_b_t_b * grad_a_a_t.t()).sum()

        energy = energy * float(scaling) * float(scaling)
        return torch.clamp(energy, min=0.0)


def set_lora_only_trainable(model: nn.Module) -> None:
    """Freeze all non-LoRA parameters before sequential training starts."""
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)
