from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .action_chunking import make_discount_weights


def discounted_gaussian_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 0.97,
) -> torch.Tensor:
    std = log_std.exp().clamp_min(1e-4)
    variance = std.square()
    nll = 0.5 * (((target - mean).square() / variance) + 2.0 * log_std + math.log(2.0 * math.pi))
    nll = nll.mean(dim=-1)
    weights = make_discount_weights(mean.size(1), gamma=gamma, device=mean.device)
    return (nll * weights.unsqueeze(0)).sum(dim=1).mean()


def action_mse(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(mean, target)


def jerk_metric(actions: torch.Tensor) -> torch.Tensor:
    if actions.size(1) < 3:
        return torch.zeros((), device=actions.device)
    jerk = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
    return jerk.square().mean()


def planner_accuracy(logits: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    return (pred == target_codes).float().mean()


def rollout_success(final_position: torch.Tensor, target_position: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    distance = (final_position - target_position).norm(dim=-1)
    return (distance <= threshold).float().mean()


def detach_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    detached: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            detached[key] = float(value.detach().cpu().item())
        else:
            detached[key] = float(value)
    return detached
