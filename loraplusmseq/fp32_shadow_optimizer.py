"""AdamW wrapper that updates fp32 shadow copies of low-precision parameters."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch


NamedParameter = Tuple[str, torch.nn.Parameter]


class Fp32ShadowAdamW:
    """Optimize fp32 shadows, then write results back to model parameters.

    The model parameters stay in their original dtype for forward/backward.
    After backward, low-precision grads are copied to fp32 shadow parameters.
    AdamW updates the fp32 shadows and copies the result back to the original
    model parameters in their existing dtype.
    """

    def __init__(
        self,
        named_params: List[NamedParameter],
        lr: float,
        weight_decay: float,
        use_weight_decay: Callable[[str, torch.nn.Parameter], bool],
        include_frozen: bool = False,
    ) -> None:
        self.lp_to_hp: Dict[torch.nn.Parameter, torch.nn.Parameter] = {}
        self.hp_to_lp: Dict[torch.nn.Parameter, torch.nn.Parameter] = {}
        self.named_lp_params: List[NamedParameter] = []
        self.use_weight_decay = use_weight_decay

        param_groups = [
            {"params": [], "weight_decay": weight_decay},
            {"params": [], "weight_decay": 0.0},
        ]
        self._add_named_params_to_groups(named_params, param_groups, include_frozen=include_frozen)
        if not self.lp_to_hp:
            raise RuntimeError("No trainable parameters were provided to Fp32ShadowAdamW.")
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def add_named_params(self, named_params: List[NamedParameter], include_frozen: bool = False) -> int:
        return self._add_named_params_to_groups(
            named_params,
            self.optimizer.param_groups,
            include_frozen=include_frozen,
        )

    def hp_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.lp_to_hp.values())

    def hp_parameters_for_lp(self, named_params: List[NamedParameter]) -> List[torch.nn.Parameter]:
        return [self.lp_to_hp[param] for _, param in named_params if param in self.lp_to_hp]

    def lp_parameters(self) -> List[torch.nn.Parameter]:
        return [param for _, param in self.named_lp_params]

    def move_hp_params_to_cpu(self) -> int:
        moved = 0
        for hp_param in self.lp_to_hp.values():
            if hp_param.device.type == "cpu":
                continue
            hp_param.data = hp_param.data.to(device="cpu")
            moved += 1
        if moved and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return moved

    def move_hp_params_to_lp_devices(self, named_params: List[NamedParameter]) -> int:
        moved = 0
        for _, lp_param in named_params:
            hp_param = self.lp_to_hp.get(lp_param)
            if hp_param is None or hp_param.device == lp_param.device:
                continue
            hp_param.data = hp_param.data.to(device=lp_param.device, non_blocking=lp_param.device.type == "cuda")
            moved += 1
        return moved

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, lp_param in self.named_lp_params:
            lp_param.grad = None if set_to_none else torch.zeros_like(lp_param)
        for hp_param in self.lp_to_hp.values():
            hp_param.grad = None if set_to_none else torch.zeros_like(hp_param)
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def sync_lp_grads_to_hp(self, clear_lp_grads: bool = True) -> None:
        """Copy current model grads to fp32 shadow grads."""
        for lp_param, hp_param in self.lp_to_hp.items():
            if lp_param.grad is None:
                hp_param.grad = None
                continue
            hp_param.grad = lp_param.grad.detach().to(device=hp_param.device, dtype=torch.float32)
            if clear_lp_grads:
                lp_param.grad = None

    def step(self, sync_grads: bool = True) -> None:
        if sync_grads:
            self.sync_lp_grads_to_hp(clear_lp_grads=True)
        self.optimizer.step()
        self._update_lp_params()
        self._clean_hp_grad()

    def close(self) -> None:
        self.zero_grad(set_to_none=True)
        for hp_param in list(self.lp_to_hp.values()):
            if hp_param in self.optimizer.state:
                del self.optimizer.state[hp_param]
        for group in self.optimizer.param_groups:
            group["params"] = []
        self.lp_to_hp.clear()
        self.hp_to_lp.clear()
        self.named_lp_params.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_lp_states(self, named_params: List[NamedParameter]) -> int:
        reset_count = 0
        for _, lp_param in named_params:
            hp_param = self.lp_to_hp.get(lp_param)
            if hp_param is not None and hp_param in self.optimizer.state:
                del self.optimizer.state[hp_param]
                reset_count += 1
        return reset_count

    def _add_named_params_to_groups(
        self,
        named_params: List[NamedParameter],
        param_groups: List[Dict[str, object]],
        include_frozen: bool = False,
    ) -> int:
        added = 0
        for name, lp_param in named_params:
            if lp_param in self.lp_to_hp:
                continue
            if not include_frozen and not lp_param.requires_grad:
                continue
            hp_param = torch.nn.Parameter(lp_param.detach().clone().float(), requires_grad=True)
            self.lp_to_hp[lp_param] = hp_param
            self.hp_to_lp[hp_param] = lp_param
            self.named_lp_params.append((name, lp_param))
            group_index = 0 if self.use_weight_decay(name, lp_param) else 1
            param_groups[group_index]["params"].append(hp_param)
            added += 1
        return added

    @torch.no_grad()
    def _update_lp_params(self) -> None:
        for lp_param, hp_param in self.lp_to_hp.items():
            lp_param.data.copy_(hp_param.to(device=lp_param.device, dtype=lp_param.dtype).data)

    def _clean_hp_grad(self) -> None:
        for hp_param in self.lp_to_hp.values():
            hp_param.grad = None


def count_unique_params(named_params: Iterable[NamedParameter]) -> int:
    seen = set()
    total = 0
    for _, param in named_params:
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        total += param.numel()
    return int(total)
