from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _normalize_clip(clip: torch.Tensor) -> torch.Tensor:
    clip = clip.float()
    if clip.max() > 1.5:
        clip = clip / 255.0
    return clip


def _channel_centroid(channel: torch.Tensor) -> torch.Tensor:
    batch, time, height, width = channel.shape
    ys = torch.linspace(0.0, 1.0, height, device=channel.device)
    xs = torch.linspace(0.0, 1.0, width, device=channel.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    weight = channel.clamp_min(1e-6)
    denom = weight.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    x = (weight * grid_x).sum(dim=(-1, -2), keepdim=True) / denom
    y = (weight * grid_y).sum(dim=(-1, -2), keepdim=True) / denom
    return torch.cat([x, y], dim=-1).squeeze(-2)


class SurrogateLocalTokenizer(nn.Module):
    def __init__(self, vocab_size: int, n_codes: int, embed_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_codes = n_codes
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(3, hidden_dim, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((n_codes, 1, 1)),
        )
        self.proj = nn.Linear(hidden_dim, vocab_size)

    @torch.no_grad()
    def encode_local_clip(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        clip = _normalize_clip(clip).permute(0, 2, 1, 3, 4)
        features = self.encoder(clip).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        logits = self.proj(features)
        codes = logits.argmax(dim=-1)
        embeds = self.embedding(codes)
        return {"codes": codes, "embeds": embeds}

    def code_embed(self, code_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(code_ids)


class MockGeometryTokenizer(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.vocab_size = 32
        self.n_codes = 4
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)

    def _bucket_angle(self, vector: torch.Tensor, offset: int) -> torch.Tensor:
        angle = torch.atan2(vector[..., 1], vector[..., 0])
        normalized = ((angle + math.pi) / (2.0 * math.pi)).clamp(0.0, 0.9999)
        return (normalized * 8).long() + offset

    def _bucket_scalar(self, scalar: torch.Tensor, offset: int) -> torch.Tensor:
        normalized = ((scalar + 1.0) / 2.0).clamp(0.0, 0.9999)
        return (normalized * 8).long() + offset

    @torch.no_grad()
    def encode_local_clip(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        clip = _normalize_clip(clip)
        agent_pos = _channel_centroid(clip[:, :, 2])
        target_pos = _channel_centroid(clip[:, :, 1])
        initial_vel = agent_pos[:, 1] - agent_pos[:, 0]
        final_vel = agent_pos[:, -1] - agent_pos[:, -2]
        radial_progress = (target_pos[:, 0] - agent_pos[:, 0]).norm(dim=-1) - (target_pos[:, -1] - agent_pos[:, -1]).norm(dim=-1)
        midpoint = agent_pos[:, agent_pos.size(1) // 2]
        curvature = agent_pos[:, -1] - 2.0 * midpoint + agent_pos[:, 0]

        codes = torch.stack(
            [
                self._bucket_angle(initial_vel, offset=0),
                self._bucket_angle(final_vel, offset=8),
                self._bucket_scalar(radial_progress, offset=16),
                self._bucket_scalar(curvature.norm(dim=-1).tanh() * curvature[..., 0].sign(), offset=24),
            ],
            dim=1,
        )
        return {"codes": codes, "embeds": self.embedding(codes)}

    def code_embed(self, code_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(code_ids)


class OfficialDLDMTokenizer(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        from videoworld2.latent_dynamics.discrete_video_latent_dynamic import CausalDiscreteVideoLatentDynamicTokenizer

        model_kwargs = cfg.get("official_kwargs", {})
        self.model = CausalDiscreteVideoLatentDynamicTokenizer(**model_kwargs)
        checkpoint_path = cfg.get("checkpoint_path")
        if checkpoint_path:
            state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
            state_dict = state.get("state_dict", state)
            if not isinstance(state_dict, dict):
                raise ValueError(f"Unsupported tokenizer checkpoint payload: {type(state_dict)}")
            cleaned: dict[str, Any] = {}
            for key, value in state_dict.items():
                if key.startswith("ema.network-"):
                    mapped = key[len("ema.network-") :].replace("-", ".")
                    cleaned.setdefault(mapped, value)
                elif key.startswith("network."):
                    cleaned[key[len("network.") :]] = value
                elif not key.startswith("ema."):
                    cleaned[key] = value
            expected = self.model.state_dict()
            filtered = {
                key: value
                for key, value in cleaned.items()
                if key in expected and hasattr(value, "shape") and expected[key].shape == value.shape
            }
            self.model.load_state_dict(filtered, strict=False)
        self.vocab_size = int(cfg.get("vocab_size", 1024))
        self.n_codes = int(cfg.get("n_codes", 8))

    @torch.no_grad()
    def encode_local_clip(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        clip = _normalize_clip(clip).permute(0, 2, 1, 3, 4)
        _, quant_info, (quant_codes, _), _ = self.model.encode(clip)
        if isinstance(quant_info, dict) and "indices" in quant_info:
            code_ids = quant_info["indices"]
        else:
            code_ids = quant_info
        code_ids = code_ids.flatten(start_dim=1).long()
        conv_dtype = self.model.post_quant_conv.conv3d.weight.dtype
        quant_codes = quant_codes.to(dtype=conv_dtype)
        quant = self.model.post_quant_conv(quant_codes)
        if getattr(self.model, "connector_type", "conv") == "llama":
            embeds = self.model.quant_to_dit_dim(quant)
        else:
            embeds = self.model.quant_to_dit_dim(quant).flatten(2).permute(0, 2, 1)
        return {"codes": code_ids, "embeds": embeds}

    def code_embed(self, code_ids: torch.Tensor) -> torch.Tensor:
        quant_codes = self.model.quantizer.indices_to_codes(code_ids)
        return quant_codes.flatten(start_dim=2).transpose(1, 2)


class DLDMLocalAdapter(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.backend = cfg.get("backend", "mock_geometry")
        if self.backend == "official":
            self.impl = OfficialDLDMTokenizer(cfg)
        elif self.backend == "surrogate":
            self.impl = SurrogateLocalTokenizer(
                vocab_size=int(cfg.get("vocab_size", 64)),
                n_codes=int(cfg.get("n_codes", 4)),
                embed_dim=int(cfg.get("embed_dim", 128)),
                hidden_dim=int(cfg.get("hidden_dim", 128)),
            )
        elif self.backend == "mock_geometry":
            self.impl = MockGeometryTokenizer(embed_dim=int(cfg.get("embed_dim", 128)))
        else:
            raise ValueError(f"Unsupported dLDM backend: {self.backend}")

        self.vocab_size = int(getattr(self.impl, "vocab_size"))
        self.n_codes = int(getattr(self.impl, "n_codes"))
        self.embed_dim = int(cfg.get("embed_dim", 128))

    @torch.no_grad()
    def encode_local_clip(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.impl.encode_local_clip(clip)

    def code_embed(self, code_ids: torch.Tensor) -> torch.Tensor:
        return self.impl.code_embed(code_ids)

    @torch.no_grad()
    def maybe_decode(self, first_frame: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        return first_frame
