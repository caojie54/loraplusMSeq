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


def _target_match(module_name: str, target_modules: Sequence[str]) -> bool:
    return module_name.split(".")[-1] in set(target_modules)


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
        if not _target_match(module_name, target_modules) or not _is_peft_lora_linear(module):
            continue
        weight = module.base_layer.weight
        num_params = weight.numel()
        if getattr(module.base_layer, "bias", None) is not None:
            num_params += module.base_layer.bias.numel()
        candidates.append(
            CandidateModule(
                name=module_name,
                short_name=module_name.split(".")[-1],
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
        top_k: int = 8,
        param_ratio: float = 0.0,
        seed: int = 0,
        alpha_score: str = "lora_update_ratio",
        history_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.target_modules = list(target_modules)
        self.method = method
        self.top_k = top_k
        self.param_ratio = param_ratio
        self.rng = random.Random(seed)
        self.alpha_score = alpha_score
        self.history_path = history_path
        self.candidates = collect_candidate_modules(model, target_modules)
        self.candidate_by_name = {candidate.name: candidate for candidate in self.candidates}
        self.managed_param_ids: Set[int] = set()
        self.selected_names: List[str] = []
        self.selection_step = -1
        self.selection_records: List[Dict[str, object]] = []
        self.total_candidate_params = sum(candidate.num_params for candidate in self.candidates)
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

    def save_candidates(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_modules": self.target_modules,
                    "method": self.method,
                    "top_k": self.top_k,
                    "param_ratio": self.param_ratio,
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
    ) -> List[str]:
        self.selected_names = [name for name in selected_names if name in self.candidate_by_name]
        self.selection_step = step
        self._record_selection(step=step, reason=reason, scores=scores)
        return self.selected_names

    def select_random(self, step: int, reason: str) -> List[str]:
        selected = self._choose_random_by_budget()
        return self.record_selection([candidate.name for candidate in selected], step=step, reason=reason)

    def select_by_alpha_scores(self, scores: Dict[str, float], step: int) -> List[str]:
        ranked = sorted(
            self.candidates,
            key=lambda candidate: scores.get(candidate.name, float("-inf")),
            reverse=True,
        )
        selected = self._choose_ranked_by_budget(ranked)
        return self.record_selection(
            [candidate.name for candidate in selected],
            step=step,
            reason="seq_alpha",
            scores=scores,
        )

    def _choose_random_by_budget(self) -> List[CandidateModule]:
        """Uniform random baseline: sample without replacement instead of taking model order."""
        if not self.candidates:
            return []
        if self.top_k > 0:
            return self.rng.sample(self.candidates, min(self.top_k, len(self.candidates)))

        shuffled_candidates = list(self.candidates)
        self.rng.shuffle(shuffled_candidates)
        return self._choose_ranked_by_budget(shuffled_candidates)

    def _choose_ranked_by_budget(self, ranked_candidates: Sequence[CandidateModule]) -> List[CandidateModule]:
        if not ranked_candidates:
            return []
        if self.top_k > 0:
            return list(ranked_candidates[: self.top_k])
        if self.param_ratio <= 0:
            return [ranked_candidates[0]]

        budget = max(1, int(self.total_candidate_params * self.param_ratio))
        selected: List[CandidateModule] = []
        used = 0
        for candidate in ranked_candidates:
            if selected and used + candidate.num_params > budget:
                continue
            selected.append(candidate)
            used += candidate.num_params
            if used >= budget:
                break
        return selected or [ranked_candidates[0]]

    def _record_selection(
        self,
        step: int,
        reason: str,
        scores: Optional[Dict[str, float]] = None,
    ) -> None:
        selected_params = sum(self.candidate_by_name[name].num_params for name in self.selected_names)
        record: Dict[str, object] = {
            "step": int(step),
            "reason": reason,
            "selected_names": list(self.selected_names),
            "selected_short_names": [self.candidate_by_name[name].short_name for name in self.selected_names],
            "selected_params": int(selected_params),
            "selected_param_ratio": float(selected_params / max(1, self.total_candidate_params)),
        }
        if scores is not None:
            record["scores"] = {
                name: float(score)
                for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:50]
            }
        self.selection_records.append(record)
        self._append_history(record)
        print(
            f"[loraplusMSeq] step={step} reason={reason} selected={len(self.selected_names)} "
            f"params={selected_params} ratio={record['selected_param_ratio']:.6f}",
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

            if self.alpha_score == "lora_grad_norm":
                a_term = torch.norm(a_grad.detach().float()).item() if a_grad is not None else 0.0
                b_term = torch.norm(b_grad.detach().float()).item() if b_grad is not None else 0.0
                denom = math.sqrt(max(1, a_weight.numel() + b_weight.numel()))
                total += math.sqrt(a_term * a_term + b_term * b_term) / denom
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

        if self.alpha_score == "lora_grad_norm":
            return float(total)
        return float(total * math.log(max(2, base_num_params)))


def set_lora_only_trainable(model: nn.Module) -> None:
    """Freeze all non-LoRA parameters before sequential training starts."""
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)

