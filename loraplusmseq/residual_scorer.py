"""Small residual-gradient helpers for LoRA-wrapped Linear modules."""

from __future__ import annotations

from typing import Tuple

import torch


def _adapter_key(module, adapter_name: str) -> str:
    lora_a = getattr(module, "lora_A", None)
    lora_b = getattr(module, "lora_B", None)
    if lora_a is None or lora_b is None:
        raise AttributeError("LoRA module does not expose lora_A/lora_B")
    if adapter_name in lora_a and adapter_name in lora_b:
        return adapter_name
    keys = [key for key in lora_a.keys() if key in lora_b]
    if len(keys) == 1:
        return keys[0]
    raise KeyError(f"Cannot resolve LoRA adapter '{adapter_name}'. Available adapters: {keys}")


def _lora_weights(module, adapter_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    key = _adapter_key(module, adapter_name)
    return module.lora_A[key].weight, module.lora_B[key].weight


@torch.no_grad()
def orth_basis_svd(
    mat: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 0.0,
    accurate: bool = False,
) -> torch.Tensor:
    """Return an orthonormal basis for the numerical column space of mat."""
    m = mat.detach().to(dtype=torch.float32)
    if m.numel() == 0:
        return m.new_empty((m.shape[0], 0))

    kwargs = {}
    if accurate and m.is_cuda:
        kwargs["driver"] = "gesvd"

    u, singular_values, _ = torch.linalg.svd(m, full_matrices=False, **kwargs)
    if singular_values.numel() == 0:
        return m.new_empty((m.shape[0], 0))

    smax = singular_values[0]
    if float(smax.item()) == 0.0:
        return m.new_empty((m.shape[0], 0))

    threshold = torch.maximum(smax.new_tensor(atol), rtol * smax)
    rank = int((singular_values > threshold).sum().item())
    if rank == 0:
        return m.new_empty((m.shape[0], 0))
    return u[:, :rank].contiguous()
