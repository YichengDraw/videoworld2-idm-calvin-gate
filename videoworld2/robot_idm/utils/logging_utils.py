from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, output_dir: str | Path, config: dict[str, Any], enabled: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.enabled = enabled
        self.wandb_run = None

        wandb_cfg = config.get("logging", {}).get("wandb", {})
        use_wandb = enabled and wandb_cfg.get("enabled", True)
        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_cfg.get("project", "vw2-idm"),
                    name=config.get("experiment_name"),
                    dir=str(self.output_dir),
                    mode=wandb_cfg.get("mode", "offline"),
                    config=config,
                    reinit=True,
                )
            except Exception:
                self.wandb_run = None

    def log(self, step: int, metrics: dict[str, float]) -> None:
        payload = {"step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
