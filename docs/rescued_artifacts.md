# Rescued Artifacts

The original remote 4090 run produced checkpoints, latent caches, manifests, and offline evaluation files that were rescued to a local non-versioned directory after the instance was stopped.

These artifacts are intentionally **not committed** to this repository because they are large run outputs rather than source code.

The committed files in `results/` are the public, lightweight interpreted summaries. This inventory is historical artifact provenance for the non-versioned bundle, not the current source of truth for the README tables.
The current interpretation boundary is recorded in `results/phase1_result_provenance.json`, with a reproducible rescued-bundle audit in `results/rescued_artifact_audit.json`.

## Recovered groups

| Group | Files |
| --- | --- |
| Latent caches | `train_local_codes.pt`, `val_local_codes.pt` |
| Window indexes | `train_windows.json`, `val_windows.json` |
| CALVIN manifests | `train_manifest.json`, `val_manifest.json` |
| Controller checkpoints | `BC_vis`, `BC_vis_proprio`, `Pair_IDM_GTcode`, `History_IDM_GTcode`, `VW2_hidden_mlp_action_head` |
| Run metadata | `resolved_config.yaml`, `offline_eval.json`, `metrics.jsonl` for each recovered controller |

## Key recovered sizes

| Artifact | Size (bytes) |
| --- | ---: |
| `train_local_codes.pt` | `70,227,198` |
| `val_local_codes.pt` | `34,206,634` |
| `train_windows.json` | `238,664` |
| `val_windows.json` | `1,487,109` |
| `train_manifest.json` | `49,337` |
| `val_manifest.json` | `304,640` |
| `bc_vis_calvin_4090/best.pt` | `711,093,016` |
| `bc_vis_proprio_calvin_4090/best.pt` | `711,291,568` |
| `pair_idm_calvin_4090/best.pt` | `711,291,568` |
| `history_gt_calvin_4090/best.pt` | `711,358,482` |
| `vw2_hidden_mlp_action_head_calvin_4090/best.pt` | `324,263,030` |

## Important caveats

- No standalone normalizer file existed in the rescued output tree.
- The rescued CALVIN manifests referenced remote dataset roots during the original run.
- The rescued files preserved the completed offline metrics, but they do not by themselves complete the later real closed-loop CALVIN adjudication request.
- The latent caches do not contain raw CALVIN RGB, proprio, or action arrays. Local offline re-evaluation still needs those raw frame files or an equivalent remapped dataset root.
- The rescued window payloads match the current window-generation logic, but the rescued window files predate the current strict metadata guard and must be rebuilt before strict local reuse. The rescued validation latent cache covers the configured `limit_val_windows: 1024` subset used by the recovered offline evaluations, not the full 12,881-window validation index.
- The rescued CALVIN manifests contain same-root within-split span overlaps. The committed audit records 20 train overlap pairs and 8,383 validation overlap pairs; these are treated as CALVIN annotation-window weighting, not as independent raw-frame evidence.
- `planner_code_accuracy` is not applicable for the committed Phase 1 rows because none of them publishes a predicted-code planner branch.
- The Phase 1 action losses are raw-vector losses over the stored CALVIN action representation because no standalone action normalizer was recovered. Offline jerk is computed over the full predicted action chunk, not an executed closed-loop action stream.
- The locally observed `ldm_tokenizer_training_init_weights.pt` file is not accepted by the current official-tokenizer guard for fresh cache extraction because it misses encode-path parameters. A fresh official-tokenizer rerun needs a complete tokenizer checkpoint or an explicitly diagnostic partial-checkpoint override.
