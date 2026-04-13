# VW2-IDM Training Order

1. Build or point `train_manifest` and `val_manifest` to episodes in the standard robot format.
2. Extract local future-dynamics codes with `bash scripts/extract_local_robot_codes.sh configs/vw2_idm/data_calvin_mini.yaml`.
3. Train the one-step latent planner with `bash scripts/train_local_planner.sh configs/vw2_idm/planner_small.yaml`.
4. Train IDM variants:
   - `bash scripts/train_history_idm.sh configs/vw2_idm/exp_bc.yaml`
   - `bash scripts/train_history_idm.sh configs/vw2_idm/exp_pair_idm.yaml`
   - `bash scripts/train_history_idm.sh configs/vw2_idm/exp_gt_code_idm.yaml`
   - `bash scripts/train_history_idm.sh configs/vw2_idm/exp_pred_code_idm.yaml`
5. Train the optional verifier with `bash scripts/train_forward_verifier.sh configs/vw2_idm/verifier_small.yaml`.
6. Run the ablation bundle with `bash scripts/eval_vw2_idm.sh configs/vw2_idm/exp_history_ablation.yaml`.
7. Distill the direct policy with `bash scripts/distill_direct_policy.sh configs/vw2_idm/exp_distill.yaml`.

`configs/vw2_idm/data_mock_smoke.yaml` is the local smoke path used in this workspace because no CALVIN or LIBERO data is available here. Swap that data config for `data_calvin_mini.yaml` or `data_libero_object.yaml` after you build the corresponding manifests.
