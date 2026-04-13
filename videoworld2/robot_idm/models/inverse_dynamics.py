from __future__ import annotations

import torch
from torch import nn


class HistoryAwareIDM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        chunk: int = 8,
        d_model: int = 256,
        depth: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_future_codes: bool = True,
        use_past_actions: bool = True,
        num_embodiments: int = 8,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk = chunk
        self.use_future_codes = use_future_codes
        self.use_past_actions = use_past_actions
        self.query_tokens = nn.Parameter(torch.randn(chunk, d_model) * 0.02)
        self.action_hist_proj = nn.Linear(action_dim, d_model)
        self.embodiment_embedding = nn.Embedding(num_embodiments, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.mean_head = nn.Linear(d_model, action_dim)
        self.log_std_head = nn.Linear(d_model, action_dim)

    def forward(
        self,
        state_tokens: torch.Tensor,
        future_code_embeds: torch.Tensor | None,
        past_action_hist: torch.Tensor | None,
        embodiment_id: torch.Tensor,
        noisy_code_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = [state_tokens]
        if self.use_future_codes and future_code_embeds is not None:
            memory.append(noisy_code_embeds if noisy_code_embeds is not None else future_code_embeds)
        if self.use_past_actions and past_action_hist is not None:
            memory.append(self.action_hist_proj(past_action_hist))
        memory.append(self.embodiment_embedding(embodiment_id).unsqueeze(1))
        stacked_memory = torch.cat(memory, dim=1)
        queries = self.query_tokens.unsqueeze(0).expand(state_tokens.size(0), -1, -1)
        decoded = self.decoder(queries, stacked_memory)
        mean = self.mean_head(decoded)
        log_std = self.log_std_head(decoded).clamp(-5.0, 2.0)
        return mean, log_std

    @torch.no_grad()
    def sample(
        self,
        state_tokens: torch.Tensor,
        future_code_embeds: torch.Tensor | None,
        past_action_hist: torch.Tensor | None,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        mean, log_std = self(
            state_tokens=state_tokens,
            future_code_embeds=future_code_embeds,
            past_action_hist=past_action_hist,
            embodiment_id=embodiment_id,
        )
        noise = torch.randn_like(mean)
        return mean + noise * log_std.exp()
