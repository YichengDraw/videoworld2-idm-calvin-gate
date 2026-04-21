from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from videoworld2.robot_idm.data.collate import robot_idm_collate
from videoworld2.robot_idm.utils.runtime import make_torch_generator, make_worker_init_fn


class LatentCodeCache:
    def __init__(
        self,
        cache_path: str | Path,
        expected_metadata: dict[str, Any] | None = None,
        expected_keys: set[str] | None = None,
        allow_legacy_metadata: bool = False,
    ) -> None:
        self.cache_path = Path(cache_path)
        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self.metadata = payload.get("metadata", {})
        records = payload.get("records", [])
        seen: set[str] = set()
        for record in records:
            key = self.make_key(record["episode_id"], int(record["t"]))
            if key in seen:
                raise ValueError(f"Duplicate latent cache record for {key} in {self.cache_path}.")
            seen.add(key)
        self._records = {
            self.make_key(record["episode_id"], int(record["t"])): record
            for record in records
        }
        if expected_metadata is not None:
            self.validate_metadata(expected_metadata, allow_legacy=allow_legacy_metadata)
        if expected_keys is not None:
            self.validate_keys(expected_keys)

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

    def validate_metadata(self, expected_metadata: dict[str, Any], allow_legacy: bool = False) -> None:
        if not self.metadata:
            if allow_legacy:
                return
            raise ValueError(f"Latent cache {self.cache_path} has no metadata; rebuild it with overwrite_cache=true.")
        if allow_legacy and any(key not in self.metadata for key in expected_metadata):
            return
        mismatches = []
        for key, expected_value in expected_metadata.items():
            if key not in self.metadata:
                mismatches.append((key, "<missing>", expected_value))
            elif self.metadata[key] != expected_value:
                mismatches.append((key, self.metadata[key], expected_value))
        if mismatches:
            details = "; ".join(f"{key}: cached={cached!r}, expected={expected!r}" for key, cached, expected in mismatches[:5])
            raise ValueError(f"Latent cache metadata mismatch in {self.cache_path}: {details}. Rebuild with overwrite_cache=true.")

    def validate_keys(self, expected_keys: set[str]) -> None:
        actual_keys = set(self._records)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)[:5]
            extra = sorted(actual_keys - expected_keys)[:5]
            raise ValueError(
                f"Latent cache key mismatch in {self.cache_path}: "
                f"sample missing={missing}, sample extra={extra}."
            )

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
    metadata: dict[str, Any] | None = None,
    seed: int = 7,
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
        generator=make_torch_generator(seed),
        worker_init_fn=make_worker_init_fn(seed),
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

    cache_metadata = {
        "backend": getattr(adapter, "backend", "unknown"),
        "n_codes": int(records[0]["codes"].numel()) if records else 0,
        "embed_dim": int(records[0]["embeds"].size(-1)) if records else 0,
    }
    if metadata:
        cache_metadata.update(metadata)
    return LatentCodeCache.save(records, destination, metadata=cache_metadata)
