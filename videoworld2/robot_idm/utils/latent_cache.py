from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from videoworld2.robot_idm.data.collate import robot_idm_collate


class LatentCodeCache:
    def __init__(self, cache_path: str | Path) -> None:
        payload = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
        self.metadata = payload.get("metadata", {})
        records = payload.get("records", [])
        self._records = {
            self.make_key(record["episode_id"], int(record["t"])): record
            for record in records
        }

    @staticmethod
    def make_key(episode_id: str, t: int) -> str:
        return f"{episode_id}:{t}"

    def get(self, episode_id: str, t: int) -> dict[str, Any]:
        key = self.make_key(episode_id, t)
        if key not in self._records:
            raise KeyError(f"Latent code missing for {key}.")
        return self._records[key]

    def __contains__(self, item: tuple[str, int]) -> bool:
        episode_id, t = item
        return self.make_key(episode_id, t) in self._records

    @staticmethod
    def save(records: list[dict[str, Any]], output_path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata or {}, "records": records}, destination)
        return destination


def extract_code_cache(
    dataset,
    adapter,
    output_path: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
) -> Path:
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        return destination

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=robot_idm_collate,
    )
    adapter = adapter.to(device)
    adapter.eval()

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in data_loader:
            future_clip = batch["future_clip"].to(device)
            encoded = adapter.encode_local_clip(future_clip)
            codes = encoded["codes"].cpu()
            embeds = encoded["embeds"].cpu()
            for idx in range(codes.size(0)):
                records.append(
                    {
                        "episode_id": batch["episode_id"][idx],
                        "t": int(batch["t"][idx]),
                        "codes": codes[idx],
                        "embeds": embeds[idx],
                    }
                )

    metadata = {
        "backend": getattr(adapter, "backend", "unknown"),
        "n_codes": int(records[0]["codes"].numel()) if records else 0,
        "embed_dim": int(records[0]["embeds"].size(-1)) if records else 0,
    }
    return LatentCodeCache.save(records, destination, metadata=metadata)
