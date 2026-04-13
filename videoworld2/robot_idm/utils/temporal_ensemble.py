from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TemporalEnsembler:
    chunk_size: int
    action_dim: int
    buffer: list[tuple[int, torch.Tensor]] = field(default_factory=list)

    def add(self, start_t: int, chunk: torch.Tensor) -> None:
        self.buffer.append((start_t, chunk.detach().cpu()))

    def merge(self, query_start: int) -> torch.Tensor:
        total = torch.zeros(self.chunk_size, self.action_dim)
        counts = torch.zeros(self.chunk_size, 1)
        for start_t, chunk in self.buffer:
            offset = start_t - query_start
            for idx in range(chunk.size(0)):
                target_idx = offset + idx
                if 0 <= target_idx < self.chunk_size:
                    total[target_idx] += chunk[idx]
                    counts[target_idx] += 1
        return total / counts.clamp_min(1.0)
