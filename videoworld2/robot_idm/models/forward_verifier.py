from __future__ import annotations

import torch
from torch import nn

from videoworld2.robot_idm.utils.metrics import sequence_jerk


class ForwardVerifier(nn.Module):
    def __init__(
        self,
        action_dim: int,
        vocab_size: int,
        n_codes: int,
        d_model: int = 256,
        depth: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, d_model)
        self.query_tokens = nn.Parameter(torch.randn(n_codes, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, state_tokens: torch.Tensor, candidate_action_chunk: torch.Tensor) -> torch.Tensor:
        action_tokens = self.action_proj(candidate_action_chunk)
        memory = torch.cat([state_tokens, action_tokens], dim=1)
        queries = self.query_tokens.unsqueeze(0).expand(state_tokens.size(0), -1, -1)
        hidden = self.decoder(queries, memory)
        return self.output(hidden)

    @torch.no_grad()
    def rerank(
        self,
        state_tokens: torch.Tensor,
        candidate_action_chunks: torch.Tensor,
        target_codes: torch.Tensor,
        alpha: float = 1e-3,
        beta: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_candidates = candidate_action_chunks.shape[:2]
        flattened_actions = candidate_action_chunks.reshape(batch * num_candidates, *candidate_action_chunks.shape[2:])
        repeated_state = state_tokens.unsqueeze(1).expand(-1, num_candidates, -1, -1).reshape(
            batch * num_candidates,
            state_tokens.size(1),
            state_tokens.size(2),
        )
        logits = self.forward(repeated_state, flattened_actions)
        repeated_codes = target_codes.unsqueeze(1).expand(-1, num_candidates, -1).reshape(batch * num_candidates, -1)
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            repeated_codes.reshape(-1),
            reduction="none",
        ).reshape(batch * num_candidates, -1).mean(dim=1)
        jerk = sequence_jerk(flattened_actions)
        l2 = flattened_actions.square().mean(dim=(1, 2))
        score = -(ce + alpha * jerk + beta * l2)
        score = score.reshape(batch, num_candidates)
        best_idx = score.argmax(dim=1)
        chosen = candidate_action_chunks[torch.arange(batch, device=best_idx.device), best_idx]
        return chosen, score
