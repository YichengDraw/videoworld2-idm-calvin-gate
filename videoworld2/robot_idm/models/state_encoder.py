from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class SmallCNNBackbone(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.net(images).flatten(1)
        return self.proj(hidden)


class ResNet18Backbone(nn.Module):
    def __init__(self, out_dim: int = 512, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = resnet18(weights=weights)
        except Exception:
            model = resnet18(weights=None)
        self.body = nn.Sequential(*list(model.children())[:-1])
        self.out_dim = model.fc.in_features
        self.proj = nn.Linear(self.out_dim, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.body(images).flatten(1)
        return self.proj(hidden)


class TextEncoder(nn.Module):
    def __init__(self, d_model: int, max_chars: int = 48) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.embedding = nn.Embedding(128, d_model)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        batch = len(texts)
        tokens = torch.zeros(batch, self.max_chars, dtype=torch.long, device=self.embedding.weight.device)
        for idx, text in enumerate(texts):
            encoded = [min(ord(ch), 127) for ch in text[: self.max_chars]]
            if encoded:
                tokens[idx, : len(encoded)] = torch.tensor(encoded, device=tokens.device)
        mask = tokens != 0
        embeds = self.embedding(tokens)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (embeds * mask.unsqueeze(-1)).sum(dim=1) / denom


class StateEncoder(nn.Module):
    def __init__(
        self,
        image_backbone: str = "small_cnn",
        d_model: int = 256,
        temporal_depth: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_proprio: bool = True,
        use_lang: bool = True,
        freeze_backbone: bool = False,
        num_embodiments: int = 8,
        pretrained_backbone: bool = False,
        proprio_dim: int = 4,
    ) -> None:
        super().__init__()
        if image_backbone == "resnet18":
            self.vision_backbone = ResNet18Backbone(out_dim=d_model, pretrained=pretrained_backbone)
        else:
            self.vision_backbone = SmallCNNBackbone(out_dim=d_model)
        if freeze_backbone:
            for parameter in self.vision_backbone.parameters():
                parameter.requires_grad = False

        self.use_proprio = use_proprio
        self.use_lang = use_lang
        self.proprio_proj = nn.Linear(proprio_dim, d_model) if use_proprio else None
        self.lang_encoder = TextEncoder(d_model=d_model) if use_lang else None
        self.embodiment_embedding = nn.Embedding(num_embodiments, d_model)
        self.history_position = nn.Parameter(torch.randn(64, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=temporal_depth)

    def forward(
        self,
        rgb_hist: torch.Tensor,
        proprio_hist: torch.Tensor | None,
        lang_texts: Sequence[str] | None,
        embodiment_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames = rgb_hist.shape[:2]
        flat_images = rgb_hist.reshape(batch * frames, *rgb_hist.shape[2:])
        vision_tokens = self.vision_backbone(flat_images).reshape(batch, frames, -1)
        tokens = [vision_tokens]

        if self.use_proprio and proprio_hist is not None and self.proprio_proj is not None:
            tokens.append(self.proprio_proj(proprio_hist))

        if self.use_lang and lang_texts is not None and self.lang_encoder is not None:
            lang_token = self.lang_encoder(lang_texts).unsqueeze(1)
            tokens.append(lang_token)

        embodiment_token = self.embodiment_embedding(embodiment_id).unsqueeze(1)
        tokens.append(embodiment_token)

        state_tokens = torch.cat(tokens, dim=1)
        state_tokens = state_tokens + self.history_position[: state_tokens.size(1)].unsqueeze(0)
        encoded = self.temporal(state_tokens)
        pooled = encoded.mean(dim=1)
        return encoded, pooled
