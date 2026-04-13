from __future__ import annotations

import torch
from torch import nn

from .inverse_dynamics import HistoryAwareIDM


class DirectPolicy(nn.Module):
    def __init__(self, action_dim: int, chunk: int = 8, d_model: int = 256, depth: int = 4, n_heads: int = 4) -> None:
        super().__init__()
        self.policy = HistoryAwareIDM(
            action_dim=action_dim,
            chunk=chunk,
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            use_future_codes=False,
            use_past_actions=True,
        )

    def forward(
        self,
        state_tokens: torch.Tensor,
        future_code_embeds: torch.Tensor | None = None,
        past_action_hist: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
        noisy_code_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert embodiment_id is not None
        return self.policy(
            state_tokens=state_tokens,
            future_code_embeds=None,
            past_action_hist=past_action_hist,
            embodiment_id=embodiment_id,
        )


class MLPActionHead(nn.Module):
    def __init__(
        self,
        action_dim: int,
        chunk: int = 8,
        d_model: int = 256,
        hidden_dim: int = 512,
        num_embodiments: int = 8,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk = chunk
        self.embodiment_embedding = nn.Embedding(num_embodiments, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk * action_dim * 2),
        )

    def forward(
        self,
        state_tokens: torch.Tensor,
        future_code_embeds: torch.Tensor | None = None,
        past_action_hist: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
        noisy_code_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert embodiment_id is not None
        pooled_state = state_tokens.mean(dim=1)
        body = torch.cat([pooled_state, self.embodiment_embedding(embodiment_id)], dim=-1)
        outputs = self.mlp(body).reshape(state_tokens.size(0), self.chunk, self.action_dim, 2)
        mean = outputs[..., 0]
        log_std = outputs[..., 1].clamp(-5.0, 2.0)
        return mean, log_std
