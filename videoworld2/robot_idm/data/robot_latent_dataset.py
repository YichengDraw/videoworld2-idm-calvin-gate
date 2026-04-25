from __future__ import annotations

from pathlib import Path
from typing import Any

from videoworld2.robot_idm.data.robot_window_dataset import RobotWindowDataset, WindowSpec
from videoworld2.robot_idm.utils.latent_cache import LatentCodeCache


class RobotLatentWindowDataset(RobotWindowDataset):
    def __init__(
        self,
        manifest_path: str | Path,
        cache_path: str | Path,
        index_path: str | Path | None = None,
        spec: WindowSpec | None = None,
        image_size: int | None = None,
        limit_windows: int | None = None,
        rebuild_index: bool = False,
        expected_cache_metadata: dict[str, Any] | None = None,
        expected_cache_keys: set[str] | None = None,
    ) -> None:
        super().__init__(
            manifest_path=manifest_path,
            index_path=index_path,
            spec=spec,
            image_size=image_size,
            limit_windows=limit_windows,
            rebuild_index=rebuild_index,
        )
        if expected_cache_metadata is None:
            raise ValueError("RobotLatentWindowDataset requires expected_cache_metadata; call ensure_code_caches before constructing latent datasets.")
        expected_keys = expected_cache_keys or {LatentCodeCache.make_key(record["episode_id"], int(record["t"])) for record in self.index}
        self.cache = LatentCodeCache(
            cache_path,
            expected_metadata=expected_cache_metadata,
            expected_keys=expected_keys,
            allow_legacy_metadata=False,
            require_metadata=True,
        )

    def __getitem__(self, item: int) -> dict:
        sample = super().__getitem__(item)
        record = self.cache.get(sample["episode_id"], int(sample["t"]))
        sample["future_codes"] = record["codes"].long()
        sample["future_code_embeds"] = record["embeds"].float()
        return sample
