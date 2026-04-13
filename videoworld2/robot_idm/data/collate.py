from __future__ import annotations

from typing import Any

import torch


def robot_idm_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first_value = values[0]
        if isinstance(first_value, torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(first_value, (int, float, bool)):
            collated[key] = torch.as_tensor(values)
        else:
            collated[key] = values
    return collated
