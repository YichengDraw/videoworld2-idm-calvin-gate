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


def sequence_jerk(actions: torch.Tensor) -> torch.Tensor:
    if actions.size(1) < 3:
        return torch.zeros(actions.size(0), device=actions.device)
    jerk = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
    return jerk.square().mean(dim=(1, 2))


def jerk_metric(actions: torch.Tensor) -> torch.Tensor:
    return sequence_jerk(actions).mean()


def planner_accuracy(logits: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    return (pred == target_codes).float().mean()


def rollout_success(final_position: torch.Tensor, target_position: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    distance = (final_position - target_position).norm(dim=-1)
    return (distance <= threshold).float().mean()


def detach_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    detached: dict[str, float] = {}
    for key, value in metrics.items():
        detached[key] = finite_metric_value(key, value)
    return detached


def finite_metric_value(key: str, value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"Non-finite metric {key}: {metric}")
    return metric


def ensure_finite_metrics(metrics: dict[str, Any], context: str = "metrics") -> dict[str, Any]:
    def _check(value: Any, path: str) -> None:
        if value is None or isinstance(value, (str, bool)):
            return
        if isinstance(value, dict):
            for child_key, child in value.items():
                _check(child, f"{path}.{child_key}")
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _check(child, f"{path}[{index}]")
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                for index, child in enumerate(value.detach().cpu().reshape(-1).tolist()):
                    _check(child, f"{path}[{index}]")
                return
            finite_metric_value(path, value)
            return
        if isinstance(value, (int, float)):
            finite_metric_value(path, value)

    try:
        _check(metrics, context)
    except ValueError as exc:
        raise ValueError(f"{context} contains {exc}") from exc
    return metrics
