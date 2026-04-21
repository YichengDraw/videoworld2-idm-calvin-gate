from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from videoworld2.robot_idm.data.robot_window_dataset import RobotWindowDataset, WindowSpec, build_window_index
from videoworld2.robot_idm.models.forward_verifier import ForwardVerifier
from videoworld2.robot_idm.models.inverse_dynamics import HistoryAwareIDM
from videoworld2.robot_idm.models.latent_planner import LatentPlanner
from videoworld2.robot_idm.train.common import sample_code_conditioning
from videoworld2.robot_idm.train.train_idm import build_trainable_policy
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.factory import validate_manifest_pair
from videoworld2.robot_idm.utils.latent_cache import LatentCodeCache
from videoworld2.robot_idm.utils.runtime import save_json


def _make_episode(path: Path, episode_id: str, length: int = 24) -> None:
    rgb_static = torch.rand(length, 3, 32, 32)
    proprio = torch.rand(length, 4)
    action = torch.rand(length, 2)
    torch.save(
        {
            "rgb_static": rgb_static,
            "proprio": proprio,
            "action": action,
            "lang": "",
            "task_id": 0,
            "embodiment_id": 0,
            "episode_id": episode_id,
        },
        path,
    )


class RobotIDMTests(unittest.TestCase):
    def test_local_window_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            episode_path = tmp_path / "episode.pt"
            _make_episode(episode_path, "episode_0")
            manifest_path = tmp_path / "manifest.json"
            save_json({"episodes": [{"path": str(episode_path), "episode_id": "episode_0"}]}, manifest_path)

            spec = WindowSpec(history_frames=4, past_action_hist=4, future_video_horizon=8, action_chunk=8, stride=4)
            index = build_window_index([{"path": str(episode_path), "episode_id": "episode_0"}], spec)
            self.assertEqual([entry["t"] for entry in index], [4, 8, 12])

            dataset = RobotWindowDataset(manifest_path=manifest_path, spec=spec)
            sample = dataset[1]
            self.assertEqual(sample["rgb_hist"].shape, (4, 3, 32, 32))
            self.assertEqual(sample["future_clip"].shape, (9, 3, 32, 32))
            self.assertEqual(sample["action_chunk"].shape, (8, 2))

    def test_latent_cache_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "codes.pt"
            records = [
                {"episode_id": "episode_0", "t": 4, "codes": torch.tensor([1, 2, 3]), "embeds": torch.randn(3, 8)},
                {"episode_id": "episode_0", "t": 8, "codes": torch.tensor([2, 3, 4]), "embeds": torch.randn(3, 8)},
            ]
            LatentCodeCache.save(records, cache_path, metadata={"embed_dim": 8})
            cache = LatentCodeCache(cache_path)
            record = cache.get("episode_0", 8)
            self.assertTrue(torch.equal(record["codes"], torch.tensor([2, 3, 4])))
            self.assertIn(("episode_0", 4), cache)

    def test_latent_cache_rejects_stale_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "codes.pt"
            records = [{"episode_id": "episode_0", "t": 4, "codes": torch.tensor([1]), "embeds": torch.randn(1, 4)}]
            LatentCodeCache.save(records, cache_path, metadata={"metadata_version": 2, "split": "train"})

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                LatentCodeCache(cache_path, expected_metadata={"metadata_version": 2, "split": "val"})

    def test_planner_output_shape(self) -> None:
        planner = LatentPlanner(vocab_size=32, n_codes=4, d_model=64, depth=2, n_heads=4)
        state_tokens = torch.randn(3, 6, 64)
        target_codes = torch.randint(0, 32, (3, 4))
        logits = planner(state_tokens, target_codes=target_codes)
        self.assertEqual(logits.shape, (3, 4, 32))

    def test_planner_causal_mask_blocks_future_target_leakage(self) -> None:
        torch.manual_seed(0)
        planner = LatentPlanner(vocab_size=32, n_codes=4, d_model=64, depth=2, n_heads=4, dropout=0.0)
        planner.eval()
        state_tokens = torch.randn(1, 6, 64)
        target_a = torch.tensor([[1, 2, 3, 4]])
        target_b = torch.tensor([[1, 2, 9, 4]])

        logits_a = planner(state_tokens, target_codes=target_a)
        logits_b = planner(state_tokens, target_codes=target_b)

        self.assertTrue(torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6))

    def test_idm_output_shape(self) -> None:
        idm = HistoryAwareIDM(action_dim=2, chunk=8, d_model=64, depth=2, n_heads=4)
        state_tokens = torch.randn(3, 6, 64)
        future_code_embeds = torch.randn(3, 4, 64)
        past_action_hist = torch.randn(3, 4, 2)
        embodiment_id = torch.tensor([0, 1, 0])
        mean, log_std = idm(state_tokens, future_code_embeds, past_action_hist, embodiment_id)
        self.assertEqual(mean.shape, (3, 8, 2))
        self.assertEqual(log_std.shape, (3, 8, 2))

    def test_verifier_score_path(self) -> None:
        verifier = ForwardVerifier(action_dim=2, vocab_size=32, n_codes=4, d_model=64, depth=2, n_heads=4)
        state_tokens = torch.randn(2, 6, 64)
        candidates = torch.randn(2, 3, 8, 2)
        target_codes = torch.randint(0, 32, (2, 4))
        chosen, scores = verifier.rerank(state_tokens, candidates, target_codes)
        self.assertEqual(chosen.shape, (2, 8, 2))
        self.assertEqual(scores.shape, (2, 3))

    def test_predicted_code_conditioning_requires_planner(self) -> None:
        cfg = {
            "idm": {
                "use_future_codes": True,
                "code_source": "predicted",
                "mixed_code_training": False,
            }
        }
        batch = {
            "future_codes": torch.randint(0, 8, (2, 4)),
            "future_code_embeds": torch.randn(2, 4, 16),
        }
        with self.assertRaisesRegex(ValueError, "planner checkpoint"):
            sample_code_conditioning(
                cfg=cfg,
                batch=batch,
                state_tokens=torch.randn(2, 3, 16),
                adapter=mock.Mock(),
                planner=None,
                training=False,
            )

    def test_manifest_pair_rejects_overlapping_calvin_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json(
                {"episodes": [{"episode_id": "train_0", "root": "/data/calvin", "start": 10, "end": 20}]},
                train_manifest,
            )
            save_json(
                {"episodes": [{"episode_id": "val_0", "root": "/data/calvin", "start": 18, "end": 30}]},
                val_manifest,
            )

            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_load_config_preserves_adapter_source_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shared_dir = root / "configs" / "shared"
            run_dir = root / "runs"
            shared_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)
            parent_config = shared_dir / "adapter_parent.yaml"
            child_config = run_dir / "experiment.yaml"
            parent_config.write_text(
                "adapter:\n"
                "  backend: official\n"
                "  checkpoint_path: ../../checkpoints/tokenizer.pt\n",
                encoding="utf-8",
            )
            child_config.write_text(
                f"extends:\n  - {parent_config.as_posix()}\n"
                "training:\n"
                "  seed: 7\n",
                encoding="utf-8",
            )

            cfg = load_config(child_config)

            self.assertEqual(cfg["adapter"]["_config_dir"], str(shared_dir.resolve()))

    def test_build_trainable_policy_respects_variant(self) -> None:
        device = torch.device("cpu")
        base_cfg = {
            "data": {"action_chunk": 8},
            "model": {"d_model": 64, "idm_depth": 2, "n_heads": 4, "num_embodiments": 8},
            "idm": {"variant": "history", "use_future_codes": True, "use_past_actions": True},
            "policy": {},
        }

        history_model, history_key = build_trainable_policy(base_cfg, action_dim=2, device=device)
        self.assertIsInstance(history_model, HistoryAwareIDM)
        self.assertEqual(history_key, "idm")

        bc_cfg = {**base_cfg, "idm": {**base_cfg["idm"], "variant": "bc", "use_future_codes": False, "use_past_actions": False}}
        bc_model, bc_key = build_trainable_policy(bc_cfg, action_dim=2, device=device)
        self.assertNotIsInstance(bc_model, HistoryAwareIDM)
        self.assertEqual(bc_key, "direct_policy")

        mlp_cfg = {**base_cfg, "policy": {"variant": "mlp", "hidden_dim": 128}}
        mlp_model, mlp_key = build_trainable_policy(mlp_cfg, action_dim=2, device=device)
        self.assertNotIsInstance(mlp_model, HistoryAwareIDM)
        self.assertEqual(mlp_key, "direct_policy")

    def test_mlp_eval_config_rejects_non_direct_checkpoint(self) -> None:
        from videoworld2.robot_idm.eval.eval_offline_idm import load_policy_bundle

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "bad.pt"
            torch.save({"idm": {}, "state_encoder": {}}, checkpoint_path)
            cfg = {
                "_meta": {"config_path": str(Path(tmp_dir) / "exp.yaml")},
                "adapter": {"backend": "mock_geometry", "embed_dim": 64},
                "data": {
                    "action_dim": 2,
                    "action_chunk": 8,
                    "proprio_dim": 4,
                    "use_proprio": True,
                    "use_lang": True,
                },
                "model": {
                    "image_backbone": "small_cnn",
                    "d_model": 64,
                    "temporal_depth": 2,
                    "idm_depth": 2,
                    "n_heads": 4,
                    "num_embodiments": 8,
                },
                "idm": {"variant": "history", "use_future_codes": True, "use_past_actions": True},
                "policy": {"variant": "mlp", "hidden_dim": 128},
                "evaluation": {},
            }

            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                load_policy_bundle(cfg, str(checkpoint_path), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
