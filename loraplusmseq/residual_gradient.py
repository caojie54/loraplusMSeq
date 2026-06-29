"""Residual-gradient projection hooks for selected base modules."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch

from .module_selection import CandidateModule
from .residual_scorer import _lora_weights, orth_basis_svd


@torch.no_grad()
def project_with_bases(grad: torch.Tensor, u_basis: torch.Tensor, v_basis: torch.Tensor) -> torch.Tensor:
    """Return (I - U U^T) grad (I - V V^T)."""
    projected = grad.detach().float()
    u_work = None
    v_work = None
    left_coeff = None
    right_coeff = None

    if u_basis is not None and u_basis.numel() > 0 and u_basis.shape[1] > 0:
        u_work = u_basis.to(device=projected.device, dtype=projected.dtype)
        left_coeff = u_work.t() @ projected
        projected = projected - u_work @ left_coeff

    if v_basis is not None and v_basis.numel() > 0 and v_basis.shape[1] > 0:
        v_work = v_basis.to(device=projected.device, dtype=projected.dtype)
        right_coeff = projected @ v_work
        projected = projected - right_coeff @ v_work.t()

    out = projected.to(dtype=grad.dtype)
    del projected, u_work, v_work, left_coeff, right_coeff
    return out


@torch.no_grad()
def lora_residual_bases(module, adapter_name: str = "default", rtol: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return orth(B) and orth(A.T) for a LoRA-wrapped linear module."""
    a_weight, b_weight = _lora_weights(module, adapter_name)
    u_basis = orth_basis_svd(b_weight.detach(), rtol=rtol)
    v_basis = orth_basis_svd(a_weight.detach().t(), rtol=rtol)
    return u_basis, v_basis


def attach_residual_grad_hooks(
    candidate_by_name: Dict[str, CandidateModule],
    selected_names: Iterable[str],
    adapter_name: str = "default",
    rtol: float = 1e-4,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Project selected base weight gradients to the LoRA residual subspace."""
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for name in selected_names:
        candidate = candidate_by_name.get(name)
        if candidate is None:
            continue

        module = candidate.module
        base_layer = module.base_layer
        base_layer.weight.requires_grad_(True)
        u_basis, v_basis = lora_residual_bases(module, adapter_name=adapter_name, rtol=rtol)

        def make_hook(u_cached: torch.Tensor, v_cached: torch.Tensor):
            def hook(grad: torch.Tensor) -> torch.Tensor:
                return project_with_bases(grad, u_cached, v_cached)

            return hook

        handles.append(base_layer.weight.register_hook(make_hook(u_basis, v_basis)))
    return handles


def remove_hooks(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()
