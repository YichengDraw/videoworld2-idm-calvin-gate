from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from videoworld2.robot_idm.data.calvin_window_dataset import CalvinWindowDataset, _calvin_frame_fingerprint
from videoworld2.robot_idm.data.robot_window_dataset import RobotWindowDataset, WindowSpec, _file_fingerprint, build_window_index
from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter, _resolve_path_from_config_dir
from videoworld2.robot_idm.models.forward_verifier import ForwardVerifier
from videoworld2.robot_idm.models.inverse_dynamics import HistoryAwareIDM
from videoworld2.robot_idm.models.latent_planner import LatentPlanner
from videoworld2.robot_idm.train.common import (
    _manifest_source_fingerprint,
    ensure_code_caches,
    make_dataloaders,
    maybe_resume_training,
    resolve_config_path,
    sample_code_conditioning,
)
from videoworld2.robot_idm.train.train_idm import build_trainable_policy
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.factory import _resolve_path, validate_manifest_pair
from videoworld2.robot_idm.utils.latent_cache import LatentCodeCache
from videoworld2.robot_idm.utils.phase0 import prepare_phase0_overfit_cfg
from videoworld2.robot_idm.utils.runtime import configure_determinism, save_json


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

    def test_robot_manifest_relative_episode_path_resolves_from_manifest_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            episode_path = tmp_path / "episode.pt"
            _make_episode(episode_path, "episode_0")
            manifest_path = tmp_path / "manifest.json"
            save_json({"episodes": [{"path": "episode.pt", "episode_id": "episode_0"}]}, manifest_path)

            dataset = RobotWindowDataset(manifest_path=manifest_path, spec=WindowSpec())

            self.assertEqual(Path(dataset.episode_entries[0]["path"]), episode_path.resolve())
            self.assertEqual(dataset[0]["episode_id"], "episode_0")

    def test_calvin_manifest_relative_root_resolves_from_manifest_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            calvin_root = tmp_path / "calvin"
            calvin_root.mkdir()
            manifest_path = tmp_path / "manifest.json"
            save_json(
                {"episodes": [{"episode_id": "episode_0", "root": "calvin", "start": 0, "end": 20}]},
                manifest_path,
            )

            dataset = CalvinWindowDataset(manifest_path=manifest_path, spec=WindowSpec())

            self.assertEqual(Path(dataset.entry_by_id["episode_0"]["root"]), calvin_root.resolve())
            self.assertGreater(len(dataset), 0)

    def test_remote_posix_manifest_paths_are_not_rewritten_to_windows_drive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = tmp_path / "manifest.json"
            save_json(
                {"episodes": [{"episode_id": "episode_0", "root": "/remote/calvin", "start": 0, "end": 20}]},
                manifest_path,
            )

            dataset = CalvinWindowDataset(manifest_path=manifest_path, spec=WindowSpec())
            frame = _calvin_frame_fingerprint("/remote/calvin", 0)
            robot_file = _file_fingerprint("/remote/episode.pt")

            self.assertEqual(dataset.entry_by_id["episode_0"]["root"], "/remote/calvin")
            self.assertEqual(frame["path"], "/remote/calvin/episode_0000000.npz")
            self.assertEqual(robot_file["path"], "/remote/episode.pt")
            if not Path("/remote/calvin").is_absolute():
                with self.assertRaisesRegex(FileNotFoundError, "POSIX absolute path"):
                    dataset[0]

    def test_cache_source_fingerprint_preserves_remote_posix_manifest_paths(self) -> None:
        if Path("/remote/calvin").is_absolute():
            self.skipTest("POSIX roots are locally addressable on this platform")
        with tempfile.TemporaryDirectory() as tmp_dir:
            fingerprints = _manifest_source_fingerprint(
                {
                    "episodes": [
                        {"episode_id": "robot_0", "path": "/remote/episode.pt"},
                        {"episode_id": "calvin_0", "root": "/remote/calvin", "start": 0, "end": 20},
                    ]
                },
                Path(tmp_dir),
            )

            self.assertEqual(fingerprints[0]["file"]["path"], "/remote/episode.pt")
            self.assertEqual(fingerprints[1]["first_frame"]["path"], "/remote/calvin/episode_0000000.npz")
            self.assertEqual(fingerprints[1]["last_frame"]["path"], "/remote/calvin/episode_0000020.npz")

    def test_config_level_remote_posix_paths_fail_before_local_remap(self) -> None:
        if Path("/remote/config.json").is_absolute():
            self.skipTest("POSIX paths are native absolute paths on this platform")
        cfg = {"_meta": {"config_path": "C:/repo/configs/vw2_idm/exp.yaml"}}

        with self.assertRaisesRegex(ValueError, "POSIX absolute path"):
            resolve_config_path(cfg, "/remote/train_manifest.json")
        with self.assertRaisesRegex(ValueError, "POSIX absolute path"):
            _resolve_path("C:/repo/configs/vw2_idm/exp.yaml", "/remote/train_manifest.json")
        with self.assertRaisesRegex(ValueError, "POSIX absolute path"):
            _resolve_path_from_config_dir("/remote/tokenizer.pt", "C:/repo/configs/vw2_idm")

    def test_window_index_rejects_episode_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            episode_path = tmp_path / "episode.pt"
            _make_episode(episode_path, "episode_0")
            manifest_path = tmp_path / "manifest.json"
            index_path = tmp_path / "windows.json"
            save_json({"episodes": [{"path": str(episode_path), "episode_id": "episode_0"}]}, manifest_path)

            spec = WindowSpec(history_frames=4, past_action_hist=4, future_video_horizon=8, action_chunk=8, stride=4)
            RobotWindowDataset(manifest_path=manifest_path, index_path=index_path, spec=spec)
            stat = episode_path.stat()
            episode_path.write_bytes(episode_path.read_bytes() + b"changed")
            self.assertGreater(episode_path.stat().st_size, stat.st_size)

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                RobotWindowDataset(manifest_path=manifest_path, index_path=index_path, spec=spec)

    def test_dataloaders_reject_zero_window_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_episode = root / "train.pt"
            val_episode = root / "val.pt"
            _make_episode(train_episode, "train_0", length=6)
            _make_episode(val_episode, "val_0", length=6)
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json({"episodes": [{"path": str(train_episode), "episode_id": "train_0"}]}, train_manifest)
            save_json({"episodes": [{"path": str(val_episode), "episode_id": "val_0"}]}, val_manifest)
            cfg = {
                "_meta": {"config_path": str(root / "exp.yaml")},
                "data": {
                    "dataset_type": "standard",
                    "train_manifest": str(train_manifest),
                    "val_manifest": str(val_manifest),
                    "train_index": str(root / "train_index.json"),
                    "val_index": str(root / "val_index.json"),
                    "history_frames": 4,
                    "past_action_hist": 4,
                    "future_video_horizon": 8,
                    "action_chunk": 8,
                    "stride": 4,
                    "image_size": 32,
                },
                "training": {"batch_size": 2, "num_workers": 0},
            }

            with self.assertRaisesRegex(ValueError, "zero windows"):
                make_dataloaders(cfg, use_latent_cache=False)

    def test_phase0_overfit_cfg_explicitly_allows_same_train_val_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "configs" / "vw2_idm" / "exp.yaml"
            config_path.parent.mkdir(parents=True)
            base_cfg = {
                "_meta": {"config_path": str(config_path)},
                "data": {},
                "idm": {"use_future_codes": False},
                "training": {},
            }

            cfg = prepare_phase0_overfit_cfg(base_cfg, run_name="phase0_test", episodes=2, max_epochs=1)

            self.assertFalse(cfg["data"]["validate_split_disjoint"])
            self.assertEqual(cfg["data"]["train_manifest"], cfg["data"]["val_manifest"])
            train_loader, val_loader = make_dataloaders(cfg, use_latent_cache=False)
            self.assertGreater(len(train_loader.dataset), 0)
            self.assertGreater(len(val_loader.dataset), 0)

    def test_configure_determinism_enables_torch_deterministic_guards(self) -> None:
        previous_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            configure_determinism(17, deterministic=True)

            self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            if previous_env is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_env
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=True)

    def test_mock_adapter_initialization_is_stable(self) -> None:
        torch.manual_seed(123)
        first = DLDMLocalAdapter({"backend": "mock_geometry", "embed_dim": 16, "init_seed": 5})
        torch.manual_seed(999)
        second = DLDMLocalAdapter({"backend": "mock_geometry", "embed_dim": 16, "init_seed": 5})

        self.assertTrue(torch.equal(first.impl.embedding.weight, second.impl.embedding.weight))

    def test_official_adapter_rejects_partial_checkpoint_load(self) -> None:
        class FakeOfficialTokenizer(torch.nn.Module):
            def __init__(self, **kwargs) -> None:
                super().__init__()
                self.loaded = torch.nn.Linear(2, 2)
                self.missing = torch.nn.Linear(2, 2)

        fake_module = types.ModuleType("videoworld2.latent_dynamics.discrete_video_latent_dynamic")
        fake_module.CausalDiscreteVideoLatentDynamicTokenizer = FakeOfficialTokenizer
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "tokenizer.pt"
            model = FakeOfficialTokenizer()
            torch.save(
                {
                    "state_dict": {
                        "loaded.weight": model.loaded.weight.detach().clone(),
                        "loaded.bias": model.loaded.bias.detach().clone(),
                    }
                },
                checkpoint_path,
            )
            with mock.patch.dict("sys.modules", {"videoworld2.latent_dynamics.discrete_video_latent_dynamic": fake_module}):
                with self.assertRaisesRegex(ValueError, "missing parameters"):
                    DLDMLocalAdapter(
                        {
                            "backend": "official",
                            "checkpoint_path": str(checkpoint_path),
                            "embed_dim": 4,
                            "vocab_size": 8,
                            "n_codes": 2,
                        }
                    )

    def test_official_adapter_allows_decoder_only_missing_checkpoint_params(self) -> None:
        class FakeOfficialTokenizer(torch.nn.Module):
            def __init__(self, **kwargs) -> None:
                super().__init__()
                self.encoder = torch.nn.Linear(2, 2)
                self.decoder = torch.nn.Linear(2, 2)

        fake_module = types.ModuleType("videoworld2.latent_dynamics.discrete_video_latent_dynamic")
        fake_module.CausalDiscreteVideoLatentDynamicTokenizer = FakeOfficialTokenizer
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "tokenizer.pt"
            model = FakeOfficialTokenizer()
            torch.save(
                {
                    "state_dict": {
                        "encoder.weight": model.encoder.weight.detach().clone(),
                        "encoder.bias": model.encoder.bias.detach().clone(),
                    }
                },
                checkpoint_path,
            )
            with mock.patch.dict("sys.modules", {"videoworld2.latent_dynamics.discrete_video_latent_dynamic": fake_module}):
                adapter = DLDMLocalAdapter(
                    {
                        "backend": "official",
                        "checkpoint_path": str(checkpoint_path),
                        "embed_dim": 4,
                        "vocab_size": 8,
                        "n_codes": 2,
                    }
                )
            self.assertEqual(adapter.backend, "official")

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

    def test_manifest_pair_normalises_relative_and_absolute_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            calvin_root = root / "calvin"
            calvin_root.mkdir()
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json(
                {"episodes": [{"episode_id": "train_0", "root": "calvin", "start": 10, "end": 20}]},
                train_manifest,
            )
            save_json(
                {"episodes": [{"episode_id": "val_0", "root": str(calvin_root), "start": 18, "end": 30}]},
                val_manifest,
            )

            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_duplicate_ids_within_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json(
                {
                    "episodes": [
                        {"episode_id": "dup", "root": "/data/calvin_a", "start": 10, "end": 20},
                        {"episode_id": "dup", "root": "/data/calvin_b", "start": 30, "end": 40},
                    ]
                },
                train_manifest,
            )
            save_json(
                {"episodes": [{"episode_id": "val_0", "root": "/data/calvin_c", "start": 50, "end": 60}]},
                val_manifest,
            )

            with self.assertRaisesRegex(ValueError, "duplicate episode ids"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_ensure_code_caches_validates_existing_cache_even_if_other_split_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_episode = root / "train.pt"
            val_episode = root / "val.pt"
            _make_episode(train_episode, "train_0")
            _make_episode(val_episode, "val_0")
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json({"episodes": [{"path": str(train_episode), "episode_id": "train_0"}]}, train_manifest)
            save_json({"episodes": [{"path": str(val_episode), "episode_id": "val_0"}]}, val_manifest)

            train_cache = root / "train_codes.pt"
            val_cache = root / "val_codes.pt"
            LatentCodeCache.save(
                [{"episode_id": "train_0", "t": 4, "codes": torch.tensor([1]), "embeds": torch.randn(1, 4)}],
                train_cache,
                metadata={"metadata_version": 2, "split": "val"},
            )
            cfg = {
                "_meta": {"config_path": str(root / "exp.yaml")},
                "adapter": {"backend": "mock_geometry", "vocab_size": 8, "n_codes": 1, "embed_dim": 4},
                "data": {
                    "dataset_type": "standard",
                    "train_manifest": str(train_manifest),
                    "val_manifest": str(val_manifest),
                    "train_index": str(root / "train_index.json"),
                    "val_index": str(root / "val_index.json"),
                    "train_cache": str(train_cache),
                    "val_cache": str(val_cache),
                    "limit_train_windows": 1,
                    "limit_val_windows": 1,
                    "image_size": 32,
                    "action_chunk": 8,
                    "future_video_horizon": 8,
                    "history_frames": 4,
                    "past_action_hist": 4,
                    "stride": 4,
                },
                "training": {"seed": 7, "cache_batch_size": 2, "num_workers": 0},
            }
            adapter = mock.Mock(backend="mock_geometry", vocab_size=8, n_codes=1, embed_dim=4)

            with mock.patch("videoworld2.robot_idm.train.common.extract_code_cache") as extract:
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    ensure_code_caches(cfg, adapter=adapter, device=torch.device("cpu"))
                extract.assert_not_called()

    def test_resume_training_rejects_mismatched_module_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            torch.save({"model": {"wrong.weight": torch.zeros(1)}}, checkpoint_dir / "last.pt")

            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                maybe_resume_training(root, modules={"model": torch.nn.Linear(2, 1)})

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

    def test_predicted_eval_rejects_incomplete_planner_checkpoint(self) -> None:
        from videoworld2.robot_idm.eval.eval_offline_idm import load_policy_bundle
        from videoworld2.robot_idm.utils.factory import build_direct_policy, build_state_encoder

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cfg = {
                "_meta": {"config_path": str(root / "exp.yaml")},
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
                    "planner_depth": 2,
                    "n_heads": 4,
                    "num_embodiments": 8,
                },
                "idm": {
                    "variant": "bc",
                    "code_source": "predicted",
                    "planner_checkpoint": str(root / "planner.pt"),
                    "use_future_codes": True,
                    "use_past_actions": True,
                },
                "policy": {},
                "evaluation": {},
            }
            state_encoder = build_state_encoder(cfg)
            direct_policy = build_direct_policy(cfg, action_dim=2)
            policy_checkpoint = root / "policy.pt"
            torch.save(
                {
                    "state_encoder": state_encoder.state_dict(),
                    "direct_policy": direct_policy.state_dict(),
                    "model_metadata": {"checkpoint_key": "direct_policy", "policy_variant": ""},
                },
                policy_checkpoint,
            )
            torch.save({"state_encoder": state_encoder.state_dict(), "planner": {}}, root / "planner.pt")

            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_policy_bundle(cfg, str(policy_checkpoint), torch.device("cpu"))

    def test_closed_loop_rejects_non_mock_before_cache_work(self) -> None:
        from videoworld2.robot_idm.eval.eval_closed_loop import evaluate_closed_loop

        cfg = {
            "adapter": {"backend": "mock_geometry", "embed_dim": 16},
            "data": {"dataset_type": "calvin_static"},
            "training": {"seed": 7},
        }
        with mock.patch("videoworld2.robot_idm.eval.eval_closed_loop.ensure_code_caches") as ensure:
            with self.assertRaisesRegex(ValueError, "dataset_type=mock"):
                evaluate_closed_loop(cfg, checkpoint_path="unused.pt", device=torch.device("cpu"))
            ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
