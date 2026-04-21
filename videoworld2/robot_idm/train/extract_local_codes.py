from __future__ import annotations

import argparse

from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.train.common import ensure_code_caches
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.runtime import configure_determinism, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_determinism(int(cfg["training"].get("seed", 7)), deterministic=bool(cfg["training"].get("deterministic", True)))
    device = resolve_device(args.device)
    adapter = DLDMLocalAdapter(cfg["adapter"])
    ensure_code_caches(cfg, adapter=adapter, device=device)


if __name__ == "__main__":
    main()
