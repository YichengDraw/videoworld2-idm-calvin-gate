# Rescued Artifacts

The original remote 4090 run produced checkpoints, latent caches, manifests, window indexes, configs, metrics logs, and offline-evaluation files. They are intentionally kept outside git because they are large run outputs.

Public summaries live in `results/`. The reproducibility boundary is:

- `results/phase1_result_provenance.json`: interpretation and blocker summary.
- `results/rescued_artifact_audit.json`: SHA256 audit for rescued source files and controller semantics inputs.
- `results/phase1_offline_metrics.json` and `.csv`: published offline metrics.

Recovered but non-versioned groups:

| Group | Files |
| --- | --- |
| Latent caches | `train_local_codes.pt`, `val_local_codes.pt` |
| Window indexes | `train_windows.json`, `val_windows.json` |
| CALVIN manifests | `train_manifest.json`, `val_manifest.json` |
| Controller checkpoints | `BC_vis`, `BC_vis_proprio`, `Pair_IDM_GTcode`, `History_IDM_GTcode`, `VW2_hidden_mlp_action_head` |
| Run metadata | `resolved_config.yaml`, `offline_eval.json`, `metrics.jsonl` |

Important boundaries:

- No standalone normalizer file was recovered.
- Rescued manifests reference remote Linux dataset roots.
- Latent caches do not contain raw CALVIN RGB, proprio, or action arrays.
- The rescued validation latent cache covers the configured `limit_val_windows: 1024` subset used by recovered offline evaluation.
- The rescued files preserve offline metrics but do not complete real CALVIN closed-loop adjudication.
