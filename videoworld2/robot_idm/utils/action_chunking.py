from __future__ import annotations

import torch


def make_discount_weights(chunk_size: int, gamma: float, device: torch.device) -> torch.Tensor:
    steps = torch.arange(chunk_size, device=device, dtype=torch.float32)
    weights = gamma**steps
    return weights / weights.sum().clamp_min(1e-8)


def unfold_action_overlap(actions: torch.Tensor, execute_per_replan: int) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError("Expected [B, H, A] actions.")
    return actions[:, :execute_per_replan]
