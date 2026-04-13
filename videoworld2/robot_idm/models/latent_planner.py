from __future__ import annotations

import torch
from torch import nn


class LatentPlanner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_codes: int,
        d_model: int = 256,
        depth: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_codes = n_codes
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.position_embedding = nn.Parameter(torch.randn(n_codes, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.output = nn.Linear(d_model, vocab_size)
        self.bos_token_id = vocab_size

    def _prepare_tokens(self, batch_size: int, target_codes: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if target_codes is None:
            tokens = torch.full((batch_size, self.n_codes), self.bos_token_id, device=device, dtype=torch.long)
        else:
            bos = torch.full((batch_size, 1), self.bos_token_id, device=device, dtype=torch.long)
            tokens = torch.cat([bos, target_codes[:, :-1]], dim=1)
        return tokens

    def forward(
        self,
        state_tokens: torch.Tensor,
        prev_code_tokens: torch.Tensor | None = None,
        target_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del prev_code_tokens
        tokens = self._prepare_tokens(state_tokens.size(0), target_codes, state_tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding.unsqueeze(0)
        decoded = self.decoder(hidden, state_tokens)
        return self.output(decoded)

    @torch.no_grad()
    def sample(self, state_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = state_tokens.size(0)
        generated = torch.full(
            (batch_size, self.n_codes),
            self.bos_token_id,
            dtype=torch.long,
            device=state_tokens.device,
        )
        for idx in range(self.n_codes):
            logits = self.forward(state_tokens, target_codes=generated)
            next_token = logits[:, idx].argmax(dim=-1)
            generated[:, idx] = next_token
        return generated
