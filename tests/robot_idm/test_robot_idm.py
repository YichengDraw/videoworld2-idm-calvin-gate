from __future__ import annotations

import os
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts import audit_rescued_artifacts, package_phase1_results
from videoworld2.robot_idm.data.calvin_window_dataset import CalvinWindowDataset, _calvin_frame_fingerprint
from videoworld2.robot_idm.data.robot_latent_dataset import RobotLatentWindowDataset
from videoworld2.robot_idm.data.robot_window_dataset import RobotWindowDataset, WindowSpec, _file_fingerprint, build_window_index
from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter, _resolve_path_from_config_dir
from videoworld2.robot_idm.models.forward_verifier import ForwardVerifier
from videoworld2.robot_idm.models.inverse_dynamics import HistoryAwareIDM
from videoworld2.robot_idm.models.latent_planner import LatentPlanner
from videoworld2.robot_idm.train.common import (
    _adapter_cache_metadata,
    _manifest_source_fingerprint,
    ensure_code_caches,
    make_dataloaders,
    maybe_resume_training,
    resolve_config_path,
    sample_code_conditioning,
)
from videoworld2.robot_idm.train.train_idm import build_trainable_policy
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.checkpoint import adapter_checkpoint_metadata, auxiliary_checkpoint_metadata, checkpoint_reference, find_resume_path, load_checkpoint, policy_checkpoint_metadata, teacher_policy_checkpoint_metadata
from videoworld2.robot_idm.utils.factory import _resolve_path, build_idm, build_state_encoder, validate_manifest_pair
from videoworld2.robot_idm.utils.latent_cache import LatentCodeCache
from videoworld2.robot_idm.utils.phase0 import prepare_phase0_overfit_cfg
from videoworld2.robot_idm.utils.runtime import configure_determinism, load_json, save_json


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


def _make_sample(action_value: float = 0.0) -> dict[str, object]:
    return {
        "rgb_hist": torch.zeros(1, 4, 3, 32, 32),
        "proprio_hist": torch.zeros(1, 4, 4),
        "lang": [""],
        "embodiment_id": torch.zeros(1, dtype=torch.long),
        "future_code_embeds": torch.zeros(1, 4, 64),
        "future_codes": torch.zeros(1, 4, dtype=torch.long),
        "past_action_hist": torch.zeros(1, 4, 2),
        "action_chunk": torch.full((1, 8, 2), action_value),
    }


def _make_single_sample(action_value: float = 0.0) -> dict[str, object]:
    return {
        "rgb_hist": torch.zeros(4, 3, 32, 32),
        "proprio_hist": torch.zeros(4, 4),
        "lang": "",
        "embodiment_id": torch.tensor(0),
        "future_code_embeds": torch.zeros(4, 64),
        "future_codes": torch.zeros(4, dtype=torch.long),
        "past_action_hist": torch.zeros(4, 2),
        "action_chunk": torch.full((8, 2), action_value),
        "meta": {
            "target": torch.tensor([0.5, 0.5]),
            "swirl": 0.0,
            "embodiment_gain": 1.0,
            "dt": 0.1,
            "action_scale": 1.0,
            "image_size": 32,
        },
    }


class _FakeLoader:
    def __init__(self, batches: list[dict[str, object]], sample: dict[str, object] | None = None, length: int | None = None) -> None:
        self._batches = batches
        self._sample = sample or (batches[0] if batches else _make_single_sample())
        self._length = len(batches) if length is None else length
        self.dataset = self

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._sample


class _FakeAdapter(torch.nn.Module):
    backend = "mock_geometry"
    vocab_size = 8
    n_codes = 4
    embed_dim = 64

    def encode_local_clip(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = clip.size(0)
        return {
            "codes": torch.zeros(batch, self.n_codes, dtype=torch.long, device=clip.device),
            "embeds": torch.zeros(batch, self.n_codes, self.embed_dim, device=clip.device),
        }

    def code_embed(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*codes.shape, self.embed_dim, device=codes.device)


class _FakeStateEncoder(torch.nn.Module):
    def forward(self, **kwargs):
        batch = kwargs["rgb_hist"].size(0)
        return torch.zeros(batch, 6, 64, device=kwargs["rgb_hist"].device), None


class _FakePolicy(torch.nn.Module):
    def __init__(self, mean_value: float = 0.0) -> None:
        super().__init__()
        self.mean_value = mean_value

    def forward(self, **kwargs):
        state_tokens = kwargs["state_tokens"]
        mean = torch.full((state_tokens.size(0), 8, 2), self.mean_value, device=state_tokens.device)
        log_std = torch.zeros_like(mean)
        return mean, log_std


class _FakeVerifier(torch.nn.Module):
    def rerank(self, state_tokens: torch.Tensor, candidate_actions: torch.Tensor, target_codes: torch.Tensor):
        return candidate_actions[:, 0], torch.zeros(candidate_actions.size(0), candidate_actions.size(1), device=candidate_actions.device)


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

    def test_calvin_manifest_rejects_duplicate_episode_ids_on_direct_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            calvin_root = tmp_path / "calvin"
            calvin_root.mkdir()
            manifest_path = tmp_path / "manifest.json"
            save_json(
                {
                    "episodes": [
                        {"episode_id": "dup", "root": "calvin", "start": 0, "end": 20},
                        {"episode_id": "dup", "root": "calvin", "start": 21, "end": 41},
                    ]
                },
                manifest_path,
            )

            with self.assertRaisesRegex(ValueError, "duplicate episode ids"):
                CalvinWindowDataset(manifest_path=manifest_path, spec=WindowSpec())

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
            self.assertEqual(fingerprints[1]["frame_range"], [0, 20])
            self.assertEqual(fingerprints[1]["frames"][0]["path"], "/remote/calvin/episode_0000000.npz")
            self.assertEqual(fingerprints[1]["frames"][-1]["path"], "/remote/calvin/episode_0000020.npz")
            self.assertEqual(len(fingerprints[1]["frames"]), 21)

    def test_calvin_cache_source_fingerprint_covers_interior_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "calvin"
            root.mkdir()
            for frame_idx in range(3):
                (root / f"episode_{frame_idx:07d}.npz").write_bytes(f"frame-{frame_idx}".encode("utf-8"))
            manifest_payload = {"episodes": [{"episode_id": "calvin_0", "root": "calvin", "start": 0, "end": 2}]}

            before = _manifest_source_fingerprint(manifest_payload, Path(tmp_dir))
            stat = (root / "episode_0000001.npz").stat()
            (root / "episode_0000001.npz").write_bytes(b"frame-1-mutated")
            os.utime(root / "episode_0000001.npz", ns=(stat.st_atime_ns, stat.st_mtime_ns))
            after = _manifest_source_fingerprint(manifest_payload, Path(tmp_dir))

            self.assertNotEqual(before, after)
            self.assertNotEqual(before[0]["frames"][1], after[0]["frames"][1])

    def test_calvin_window_index_rejects_interior_frame_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            root = tmp_path / "calvin"
            root.mkdir()
            for frame_idx in range(21):
                (root / f"episode_{frame_idx:07d}.npz").write_bytes(f"frame-{frame_idx:02d}".encode("utf-8"))
            manifest_path = tmp_path / "manifest.json"
            index_path = tmp_path / "windows.json"
            save_json({"episodes": [{"episode_id": "calvin_0", "root": str(root), "start": 0, "end": 20}]}, manifest_path)
            spec = WindowSpec(history_frames=4, past_action_hist=4, future_video_horizon=8, action_chunk=8, stride=4)
            CalvinWindowDataset(manifest_path=manifest_path, index_path=index_path, spec=spec, rebuild_index=True)

            target = root / "episode_0000010.npz"
            stat = target.stat()
            replacement = b"mutate10"
            self.assertEqual(len(target.read_bytes()), len(replacement))
            target.write_bytes(replacement)
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                CalvinWindowDataset(manifest_path=manifest_path, index_path=index_path, spec=spec)

    def test_calvin_window_index_rejects_tampered_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            root = tmp_path / "calvin"
            root.mkdir()
            for frame_idx in range(21):
                (root / f"episode_{frame_idx:07d}.npz").write_bytes(f"frame-{frame_idx:02d}".encode("utf-8"))
            manifest_path = tmp_path / "manifest.json"
            index_path = tmp_path / "windows.json"
            save_json(
                {"episodes": [{"episode_id": "calvin_0", "root": str(root), "start": 0, "end": 20}]},
                manifest_path,
            )
            spec = WindowSpec(history_frames=4, past_action_hist=4, future_video_horizon=8, action_chunk=8, stride=4)
            CalvinWindowDataset(manifest_path=manifest_path, index_path=index_path, spec=spec, rebuild_index=True)
            payload = load_json(index_path)
            payload["windows"] = [payload["windows"][0], payload["windows"][0]]
            save_json(payload, index_path)

            with self.assertRaisesRegex(ValueError, "contents mismatch"):
                CalvinWindowDataset(manifest_path=manifest_path, index_path=index_path, spec=spec)

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

        with tempfile.TemporaryDirectory() as tmp_dir:
            child_config = Path(tmp_dir) / "child.yaml"
            child_config.write_text("extends:\n  - /remote/base.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "POSIX absolute path"):
                load_config(child_config)

    def test_adapter_cache_metadata_uses_guarded_checkpoint_resolver(self) -> None:
        if Path("/remote/tokenizer.pt").is_absolute():
            self.skipTest("POSIX paths are native absolute paths on this platform")
        cfg = {
            "_meta": {"config_path": "C:/repo/configs/vw2_idm/exp.yaml"},
            "adapter": {
                "backend": "mock_geometry",
                "checkpoint_path": "/remote/tokenizer.pt",
                "vocab_size": 32,
                "n_codes": 4,
                "embed_dim": 128,
            },
        }
        adapter = types.SimpleNamespace(backend="mock_geometry", vocab_size=32, n_codes=4, embed_dim=128)

        with self.assertRaisesRegex(ValueError, "POSIX absolute path"):
            _adapter_cache_metadata(cfg, adapter)

    def test_adapter_cache_metadata_hashes_checkpoint_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "tokenizer.pt"
            checkpoint_path.write_bytes(b"abcdef")
            adapter = types.SimpleNamespace(backend="official", vocab_size=8, n_codes=4, embed_dim=16, init_seed=7)
            cfg = {
                "_meta": {"config_path": str(root / "config.yaml")},
                "adapter": {"backend": "official", "checkpoint_path": "tokenizer.pt"},
            }
            before = _adapter_cache_metadata(cfg, adapter)
            stat = checkpoint_path.stat()
            checkpoint_path.write_bytes(b"abcdeg")
            os.utime(checkpoint_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            after = _adapter_cache_metadata(cfg, adapter)

            self.assertEqual(before["adapter_checkpoint_size"], after["adapter_checkpoint_size"])
            self.assertEqual(before["adapter_checkpoint_mtime_ns"], after["adapter_checkpoint_mtime_ns"])
            self.assertNotEqual(before["adapter_checkpoint_sha256"], after["adapter_checkpoint_sha256"])

    def test_checkpoint_metadata_binds_adapter_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            checkpoint_dir = root / "checkpoints"
            config_dir.mkdir()
            checkpoint_dir.mkdir()
            (checkpoint_dir / "tok_a.pt").write_bytes(b"tokenizer-a")
            (checkpoint_dir / "tok_b.pt").write_bytes(b"tokenizer-b")
            cfg = {
                "_meta": {"config_path": str(config_dir / "exp.yaml")},
                "adapter": {
                    "backend": "official",
                    "checkpoint_path": "../checkpoints/tok_a.pt",
                    "_config_dir": str(config_dir),
                    "embed_dim": 16,
                    "vocab_size": 8,
                    "n_codes": 4,
                },
                "data": {"action_chunk": 8, "proprio_dim": 4, "use_proprio": True, "use_lang": True},
                "model": {"d_model": 64, "idm_depth": 2, "n_heads": 4},
                "idm": {"variant": "history", "use_future_codes": True, "use_past_actions": True, "code_source": "gt"},
                "policy": {},
            }
            cfg_swapped = {**cfg, "adapter": {**cfg["adapter"], "checkpoint_path": "../checkpoints/tok_b.pt"}}

            self.assertNotEqual(
                policy_checkpoint_metadata(cfg, "idm", 2)["adapter"],
                policy_checkpoint_metadata(cfg_swapped, "idm", 2)["adapter"],
            )
            self.assertNotEqual(
                auxiliary_checkpoint_metadata(cfg, "planner")["adapter"],
                auxiliary_checkpoint_metadata(cfg_swapped, "planner")["adapter"],
            )

    def test_checkpoint_config_paths_use_guarded_resolver(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        files = [
            repo_root / "videoworld2" / "robot_idm" / "eval" / "eval_offline_idm.py",
            repo_root / "videoworld2" / "robot_idm" / "eval" / "eval_ablation.py",
            repo_root / "videoworld2" / "robot_idm" / "train" / "train_idm.py",
            repo_root / "videoworld2" / "robot_idm" / "train" / "distill_policy.py",
        ]
        forbidden = 'Path(cfg["_meta"]["config_path"]).parent /'

        for path in files:
            self.assertNotIn(forbidden, path.read_text(encoding="utf-8"), msg=str(path))

    def test_explicit_resume_path_must_exist_and_be_locally_addressable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            torch.save({"epoch": 1}, checkpoint_dir / "last.pt")

            with self.assertRaisesRegex(FileNotFoundError, "Explicit resume checkpoint not found"):
                find_resume_path(root, explicit=str(root / "missing.pt"))

        if not Path("/remote/last.pt").is_absolute():
            with self.assertRaisesRegex(ValueError, "POSIX absolute path"):
                find_resume_path(root, explicit="/remote/last.pt")

    def test_cli_checkpoint_paths_are_guarded_before_torch_load(self) -> None:
        if Path("/remote/best.pt").is_absolute():
            self.skipTest("POSIX paths are native absolute paths on this platform")

        with self.assertRaisesRegex(ValueError, "POSIX absolute path"):
            load_checkpoint("/remote/best.pt")

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

    def test_window_index_rejects_tampered_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            episode_path = tmp_path / "episode.pt"
            _make_episode(episode_path, "episode_0")
            manifest_path = tmp_path / "manifest.json"
            index_path = tmp_path / "windows.json"
            save_json({"episodes": [{"path": str(episode_path), "episode_id": "episode_0"}]}, manifest_path)
            spec = WindowSpec(history_frames=4, past_action_hist=4, future_video_horizon=8, action_chunk=8, stride=4)
            RobotWindowDataset(manifest_path=manifest_path, index_path=index_path, spec=spec, rebuild_index=True)
            payload = load_json(index_path)
            payload["windows"] = [payload["windows"][0], payload["windows"][0]]
            save_json(payload, index_path)

            with self.assertRaisesRegex(ValueError, "contents mismatch"):
                RobotWindowDataset(manifest_path=manifest_path, index_path=index_path, spec=spec)

    def test_window_index_rejects_same_size_same_mtime_episode_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            episode_path = tmp_path / "episode.pt"
            _make_episode(episode_path, "episode_0")
            manifest_path = tmp_path / "manifest.json"
            index_path = tmp_path / "windows.json"
            save_json({"episodes": [{"path": str(episode_path), "episode_id": "episode_0"}]}, manifest_path)
            spec = WindowSpec(history_frames=4, past_action_hist=4, future_video_horizon=8, action_chunk=8, stride=4)
            RobotWindowDataset(manifest_path=manifest_path, index_path=index_path, spec=spec)
            before = episode_path.read_bytes()
            stat = episode_path.stat()
            replacement = bytes([before[0] ^ 1]) + before[1:]
            episode_path.write_bytes(replacement)
            os.utime(episode_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

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

    def test_official_adapter_rejects_missing_encode_buffer(self) -> None:
        class FakeOfficialTokenizer(torch.nn.Module):
            def __init__(self, **kwargs) -> None:
                super().__init__()
                self.encoder = torch.nn.Linear(2, 2)
                self.register_buffer("encode_scale", torch.ones(2))

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
                with self.assertRaisesRegex(ValueError, "parameters or buffers"):
                    DLDMLocalAdapter(
                        {
                            "backend": "official",
                            "checkpoint_path": str(checkpoint_path),
                            "embed_dim": 4,
                            "vocab_size": 8,
                            "n_codes": 2,
                        }
                    )

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

    def test_save_json_rejects_non_finite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "metrics.json"

            with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                save_json({"metrics": {"action_nll": float("nan")}}, output)

            self.assertFalse(output.exists())

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

    def test_verifier_rerank_uses_per_candidate_jerk(self) -> None:
        verifier = ForwardVerifier(action_dim=1, vocab_size=3, n_codes=2, d_model=8, depth=1, n_heads=1, dropout=0.0)
        state_tokens = torch.zeros(1, 2, 8)
        target_codes = torch.zeros(1, 2, dtype=torch.long)
        jerky = torch.tensor([[[1.0], [-1.0], [1.0], [-1.0], [1.0]]])
        smooth = torch.tensor([[[-1.0], [-0.5], [0.0], [0.5], [1.0]]])
        candidates = torch.stack([jerky, smooth], dim=1)

        with mock.patch.object(verifier, "forward", return_value=torch.zeros(2, 2, 3)):
            chosen, scores = verifier.rerank(state_tokens, candidates, target_codes, alpha=1.0, beta=0.0)

        self.assertTrue(torch.equal(chosen, smooth))
        self.assertGreater(float(scores[0, 1]), float(scores[0, 0]))

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

    def test_mixed_code_conditioning_requires_planner_when_pred_ratio_positive(self) -> None:
        cfg = {
            "idm": {
                "use_future_codes": True,
                "code_source": "gt",
                "mixed_code_training": True,
                "mixed_code_ratio_pred": 0.2,
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
                training=True,
            )

    def test_gt_code_training_does_not_load_unused_planner_checkpoint(self) -> None:
        from videoworld2.robot_idm.train.train_idm import load_planner_bundle

        cfg = {
            "_meta": {"config_path": "C:/repo/configs/vw2_idm/exp.yaml"},
            "idm": {"code_source": "gt", "mixed_code_training": False, "planner_checkpoint": "/remote/planner.pt"},
            "model": {"d_model": 64, "planner_depth": 2, "n_heads": 4},
        }
        with mock.patch("videoworld2.robot_idm.train.train_idm.load_checkpoint") as load:
            planner_encoder, planner = load_planner_bundle(cfg, DLDMLocalAdapter({"backend": "mock_geometry", "embed_dim": 64}), torch.device("cpu"))
            self.assertIsNone(planner_encoder)
            self.assertIsNone(planner)
            load.assert_not_called()

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

    def test_manifest_pair_allows_within_split_same_root_overlapping_calvin_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json(
                {
                    "episodes": [
                        {"episode_id": "train_0", "root": "/data/calvin_train", "start": 10, "end": 20},
                        {"episode_id": "train_1", "root": "/data/calvin_train", "start": 15, "end": 25},
                    ]
                },
                train_manifest,
            )
            save_json(
                {"episodes": [{"episode_id": "val_0", "root": "/data/calvin_val", "start": 15, "end": 25}]},
                val_manifest,
            )

            validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_shared_file_source_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shared_episode = root / "shared.pt"
            _make_episode(shared_episode, "shared")
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json({"episodes": [{"episode_id": "train_0", "path": "shared.pt"}]}, train_manifest)
            save_json({"episodes": [{"episode_id": "val_0", "path": str(shared_episode)}]}, val_manifest)

            with self.assertRaisesRegex(ValueError, "share episode source files"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_copied_file_source_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_episode = root / "train.pt"
            val_episode = root / "val.pt"
            _make_episode(train_episode, "train_0")
            val_episode.write_bytes(train_episode.read_bytes())
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json({"episodes": [{"episode_id": "train_0", "path": str(train_episode)}]}, train_manifest)
            save_json({"episodes": [{"episode_id": "val_0", "path": str(val_episode)}]}, val_manifest)

            with self.assertRaisesRegex(ValueError, "share episode source contents"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_re_serialized_episode_source_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_episode = root / "train.pt"
            val_episode = root / "val.pt"
            payload = {
                "rgb_static": torch.arange(24 * 3 * 4 * 4, dtype=torch.float32).view(24, 3, 4, 4),
                "proprio": torch.arange(24 * 4, dtype=torch.float32).view(24, 4),
                "action": torch.arange(24 * 2, dtype=torch.float32).view(24, 2),
                "lang": "",
                "task_id": 0,
                "embodiment_id": 0,
                "episode_id": "train_0",
            }
            torch.save(payload, train_episode)
            copied_payload = torch.load(train_episode, map_location="cpu", weights_only=False)
            copied_payload["episode_id"] = "val_0"
            torch.save(copied_payload, val_episode, _use_new_zipfile_serialization=False)
            self.assertNotEqual(train_episode.read_bytes(), val_episode.read_bytes())
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json({"episodes": [{"episode_id": "train_0", "path": str(train_episode)}]}, train_manifest)
            save_json({"episodes": [{"episode_id": "val_0", "path": str(val_episode)}]}, val_manifest)

            with self.assertRaisesRegex(ValueError, "share episode source contents"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_copied_calvin_roots_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_root = root / "train_calvin"
            val_root = root / "val_calvin"
            train_root.mkdir()
            val_root.mkdir()
            for frame_idx in range(3):
                payload = f"frame-{frame_idx}".encode("utf-8")
                (train_root / f"episode_{frame_idx:07d}.npz").write_bytes(payload)
                (val_root / f"episode_{frame_idx:07d}.npz").write_bytes(payload)
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json({"episodes": [{"episode_id": "train_0", "root": str(train_root), "start": 0, "end": 2}]}, train_manifest)
            save_json({"episodes": [{"episode_id": "val_0", "root": str(val_root), "start": 0, "end": 2}]}, val_manifest)

            with self.assertRaisesRegex(ValueError, "share CALVIN episode source contents"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_partially_overlapping_copied_calvin_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_root = root / "train_calvin"
            val_root = root / "val_calvin"
            train_root.mkdir()
            val_root.mkdir()
            for frame_idx in range(10):
                (train_root / f"episode_{frame_idx:07d}.npz").write_bytes(f"frame-{frame_idx}".encode("utf-8"))
            for frame_idx in range(5, 15):
                (val_root / f"episode_{frame_idx:07d}.npz").write_bytes(f"frame-{frame_idx}".encode("utf-8"))
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json({"episodes": [{"episode_id": "train_0", "root": str(train_root), "start": 0, "end": 9}]}, train_manifest)
            save_json({"episodes": [{"episode_id": "val_0", "root": str(val_root), "start": 5, "end": 14}]}, val_manifest)

            with self.assertRaisesRegex(ValueError, "share CALVIN frame contents"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_shifted_copied_calvin_frame_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_root = root / "train_calvin"
            val_root = root / "val_calvin"
            train_root.mkdir()
            val_root.mkdir()
            for frame_idx in range(10):
                (train_root / f"episode_{frame_idx:07d}.npz").write_bytes(f"frame-{frame_idx}".encode("utf-8"))
            for offset, source_idx in enumerate(range(5, 10), start=100):
                (val_root / f"episode_{offset:07d}.npz").write_bytes(f"frame-{source_idx}".encode("utf-8"))
            for frame_idx in range(105, 110):
                (val_root / f"episode_{frame_idx:07d}.npz").write_bytes(f"unique-{frame_idx}".encode("utf-8"))
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json({"episodes": [{"episode_id": "train_0", "root": str(train_root), "start": 0, "end": 9}]}, train_manifest)
            save_json({"episodes": [{"episode_id": "val_0", "root": str(val_root), "start": 100, "end": 109}]}, val_manifest)

            with self.assertRaisesRegex(ValueError, "share CALVIN frame contents"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_repacked_calvin_frame_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_root = root / "train_calvin"
            val_root = root / "val_calvin"
            train_root.mkdir()
            val_root.mkdir()

            def write_frame(path: Path, tag: int, *, reordered: bool = False) -> None:
                rgb = np.full((2, 2, 3), tag, dtype=np.float32)
                proprio = np.arange(4, dtype=np.float32) + tag
                action = np.arange(2, dtype=np.float32) + tag
                if reordered:
                    np.savez(path, actions=action, unused=np.array([999]), rgb_static=rgb, robot_obs=proprio)
                else:
                    np.savez(path, rgb_static=rgb, robot_obs=proprio, actions=action)

            write_frame(train_root / "episode_0000000.npz", 0)
            write_frame(train_root / "episode_0000001.npz", 1)
            write_frame(train_root / "episode_0000002.npz", 2)
            write_frame(val_root / "episode_0000100.npz", 100)
            write_frame(val_root / "episode_0000101.npz", 1, reordered=True)
            write_frame(val_root / "episode_0000102.npz", 102)
            self.assertNotEqual(
                (train_root / "episode_0000001.npz").read_bytes(),
                (val_root / "episode_0000101.npz").read_bytes(),
            )
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json({"episodes": [{"episode_id": "train_0", "root": str(train_root), "start": 0, "end": 2}]}, train_manifest)
            save_json({"episodes": [{"episode_id": "val_0", "root": str(val_root), "start": 100, "end": 102}]}, val_manifest)

            with self.assertRaisesRegex(ValueError, "share CALVIN frame contents"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_within_split_partial_calvin_frame_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_root_a = root / "train_calvin_a"
            train_root_b = root / "train_calvin_b"
            val_root = root / "val_calvin"
            train_root_a.mkdir()
            train_root_b.mkdir()
            val_root.mkdir()

            def write_frame(path: Path, tag: int, *, reordered: bool = False) -> None:
                rgb = np.full((2, 2, 3), tag, dtype=np.float32)
                proprio = np.arange(4, dtype=np.float32) + tag
                action = np.arange(2, dtype=np.float32) + tag
                if reordered:
                    np.savez(path, actions=action, rgb_static=rgb, unused=np.array([tag]), robot_obs=proprio)
                else:
                    np.savez(path, rgb_static=rgb, robot_obs=proprio, actions=action)

            for frame_idx in range(10):
                write_frame(train_root_a / f"episode_{frame_idx:07d}.npz", frame_idx)
            for frame_idx in range(100, 105):
                write_frame(train_root_b / f"episode_{frame_idx:07d}.npz", frame_idx)
            for offset, source_idx in enumerate(range(5, 10), start=105):
                write_frame(train_root_b / f"episode_{offset:07d}.npz", source_idx, reordered=True)
            for frame_idx in range(10):
                write_frame(val_root / f"episode_{frame_idx:07d}.npz", frame_idx + 1000)

            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json(
                {
                    "episodes": [
                        {"episode_id": "train_0", "root": str(train_root_a), "start": 0, "end": 9},
                        {"episode_id": "train_1", "root": str(train_root_b), "start": 100, "end": 109},
                    ]
                },
                train_manifest,
            )
            save_json({"episodes": [{"episode_id": "val_0", "root": str(val_root), "start": 0, "end": 9}]}, val_manifest)

            with self.assertRaisesRegex(ValueError, "duplicate CALVIN frame contents"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_manifest_pair_rejects_duplicate_file_source_within_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shared_episode = root / "shared.pt"
            val_episode = root / "val.pt"
            _make_episode(shared_episode, "shared")
            _make_episode(val_episode, "val_0")
            train_manifest = root / "train_manifest.json"
            val_manifest = root / "val_manifest.json"
            save_json(
                {
                    "episodes": [
                        {"episode_id": "train_0", "path": "shared.pt"},
                        {"episode_id": "train_1", "path": str(shared_episode)},
                    ]
                },
                train_manifest,
            )
            save_json({"episodes": [{"episode_id": "val_0", "path": str(val_episode)}]}, val_manifest)

            with self.assertRaisesRegex(ValueError, "duplicate episode source files"):
                validate_manifest_pair(train_manifest, val_manifest)

    def test_latent_dataset_requires_exact_cache_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            episode_path = root / "episode.pt"
            _make_episode(episode_path, "episode_0")
            manifest_path = root / "manifest.json"
            index_path = root / "windows.json"
            cache_path = root / "codes.pt"
            save_json({"episodes": [{"path": str(episode_path), "episode_id": "episode_0"}]}, manifest_path)
            LatentCodeCache.save(
                [{"episode_id": "episode_0", "t": 4, "codes": torch.tensor([1]), "embeds": torch.randn(1, 4)}],
                cache_path,
                metadata={"metadata_version": 2, "split": "other"},
            )

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                RobotLatentWindowDataset(
                    manifest_path=manifest_path,
                    cache_path=cache_path,
                    index_path=index_path,
                    spec=WindowSpec(),
                    expected_cache_metadata={"metadata_version": 2, "split": "train"},
                )

            with self.assertRaisesRegex(ValueError, "requires expected_cache_metadata"):
                RobotLatentWindowDataset(manifest_path=manifest_path, cache_path=cache_path, index_path=index_path, spec=WindowSpec())

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

    def test_resume_training_requires_all_module_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            torch.save({"epoch": 3, "global_step": 7, "best_metric": 1.0}, checkpoint_dir / "last.pt")

            with self.assertRaisesRegex(ValueError, "missing module states"):
                maybe_resume_training(root, modules={"model": torch.nn.Linear(2, 1)})

    def test_resume_training_requires_optimizer_state_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            model = torch.nn.Linear(2, 1)
            torch.save({"model": model.state_dict()}, checkpoint_dir / "last.pt")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            with self.assertRaisesRegex(ValueError, "missing optimizer state"):
                maybe_resume_training(root, modules={"model": model}, optimizer=optimizer)

    def test_training_epoch_rejects_empty_loader(self) -> None:
        from videoworld2.robot_idm.train.train_planner import run_epoch as planner_run_epoch
        from videoworld2.robot_idm.train.train_idm import run_epoch as idm_run_epoch
        from videoworld2.robot_idm.train.train_verifier import run_epoch as verifier_run_epoch

        state_encoder = torch.nn.Linear(1, 1)
        planner = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(list(state_encoder.parameters()) + list(planner.parameters()), lr=1e-3)

        with self.assertRaisesRegex(ValueError, "no samples"):
            planner_run_epoch([], state_encoder, planner, optimizer, torch.device("cpu"), training=False)

        with self.assertRaisesRegex(ValueError, "no samples"):
            idm_run_epoch(
                {"idm": {}, "training": {}},
                [],
                state_encoder,
                _FakePolicy(),
                _FakeAdapter(),
                None,
                None,
                optimizer,
                torch.device("cpu"),
                training=False,
            )

        with self.assertRaisesRegex(ValueError, "no samples"):
            verifier_run_epoch([], state_encoder, ForwardVerifier(2, 8, 4, d_model=8, depth=1, n_heads=1), optimizer, torch.device("cpu"), training=False)

    def test_eval_paths_reject_empty_outputs(self) -> None:
        from scripts import debug_action_stats, eval_oracle_replay
        from videoworld2.robot_idm.eval import eval_closed_loop, eval_offline_idm

        cfg = {
            "adapter": {"backend": "mock_geometry", "embed_dim": 64},
            "data": {"dataset_type": "mock", "action_chunk": 8},
            "training": {"seed": 7},
            "idm": {"variant": "bc", "code_source": "gt", "use_future_codes": False, "use_past_actions": False},
            "evaluation": {"num_rollouts": 4, "execute_per_replan": 2, "rollout_horizon": 4},
        }
        empty_loader = _FakeLoader([], sample=_make_sample(), length=0)
        empty_dataset = _FakeLoader([], sample=_make_single_sample(), length=0)
        policy_bundle = (_FakeAdapter(), _FakeStateEncoder(), _FakePolicy(), None, None, None, None)

        with mock.patch.object(eval_offline_idm, "DLDMLocalAdapter", return_value=_FakeAdapter()), \
            mock.patch.object(eval_offline_idm, "ensure_code_caches"), \
            mock.patch.object(eval_offline_idm, "make_dataloaders", return_value=(None, empty_loader)), \
            mock.patch.object(eval_offline_idm, "load_policy_bundle", return_value=policy_bundle):
            with self.assertRaisesRegex(ValueError, "no validation samples"):
                eval_offline_idm.evaluate_offline(dict(cfg), "checkpoint.pt", torch.device("cpu"))

        with mock.patch.object(eval_closed_loop, "DLDMLocalAdapter", return_value=_FakeAdapter()), \
            mock.patch.object(eval_closed_loop, "ensure_code_caches"), \
            mock.patch.object(eval_closed_loop, "build_train_val_datasets", return_value=(None, empty_dataset)), \
            mock.patch.object(eval_closed_loop, "load_policy_bundle", return_value=policy_bundle):
            with self.assertRaisesRegex(ValueError, "no rollouts"):
                eval_closed_loop.evaluate_closed_loop(dict(cfg), "checkpoint.pt", torch.device("cpu"))

        with mock.patch.object(debug_action_stats, "DLDMLocalAdapter", return_value=_FakeAdapter()), \
            mock.patch.object(debug_action_stats, "ensure_code_caches"), \
            mock.patch.object(debug_action_stats, "build_train_val_datasets", return_value=(empty_dataset, empty_dataset)), \
            mock.patch.object(debug_action_stats, "load_policy_bundle", return_value=policy_bundle):
            with self.assertRaisesRegex(ValueError, "no rollouts"):
                debug_action_stats.collect_debug_stats(dict(cfg), "checkpoint.pt", split="val", device=torch.device("cpu"), max_rollouts=4)

        with mock.patch.object(eval_oracle_replay, "prepare_mock_data_if_needed"), \
            mock.patch.object(eval_oracle_replay, "resolve_config_path", return_value=Path("manifest.json")), \
            mock.patch.object(eval_oracle_replay, "load_json", return_value={"episodes": []}):
            with self.assertRaisesRegex(ValueError, "at least one episode"):
                eval_oracle_replay.evaluate_oracle_replay({"data": {"dataset_type": "mock", "val_manifest": "manifest.json"}}, split="val")

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
                "  checkpoint_path: ../../checkpoints/tokenizer.pt\n"
                "data:\n"
                "  train_manifest: data/train_manifest.json\n",
                encoding="utf-8",
            )
            child_config.write_text(
                f"extends:\n  - {parent_config.as_posix()}\n"
                "adapter:\n"
                "  embed_dim: 4\n"
                "training:\n"
                "  seed: 7\n",
                encoding="utf-8",
            )

            cfg = load_config(child_config)

            self.assertEqual(cfg["adapter"]["_config_dir"], str(shared_dir.resolve()))
            self.assertEqual(resolve_config_path(cfg, cfg["data"]["train_manifest"]), shared_dir / "data" / "train_manifest.json")

    def test_config_path_source_uses_key_path_when_values_collide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            parent_dir = root / "parent"
            child_dir = root / "child"
            parent_dir.mkdir()
            child_dir.mkdir()
            parent_config = parent_dir / "base.yaml"
            child_config = child_dir / "run.yaml"
            parent_config.write_text("data:\n  train_manifest: shared/artifact.json\n", encoding="utf-8")
            child_config.write_text(
                f"extends:\n  - {parent_config.as_posix()}\n"
                "adapter:\n"
                "  checkpoint_path: shared/artifact.json\n",
                encoding="utf-8",
            )

            cfg = load_config(child_config)

            with self.assertRaisesRegex(ValueError, "Ambiguous relative config path"):
                resolve_config_path(cfg, cfg["data"]["train_manifest"])
            self.assertEqual(
                resolve_config_path(cfg, cfg["data"]["train_manifest"], key_path="data.train_manifest"),
                parent_dir / "shared" / "artifact.json",
            )

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

    def test_mlp_config_declares_direct_policy_conditioning(self) -> None:
        cfg = load_config("configs/vw2_idm/exp_vw2_hidden_mlp_action_head_calvin_4090.yaml")
        metadata = policy_checkpoint_metadata(cfg, "direct_policy", action_dim=7)

        self.assertEqual(metadata["policy_variant"], "mlp")
        self.assertEqual(metadata["checkpoint_key"], "direct_policy")
        self.assertEqual(metadata["idm_conditioning"]["variant"], "bc")
        self.assertFalse(metadata["idm_conditioning"]["use_future_codes"])
        self.assertFalse(metadata["idm_conditioning"]["use_past_actions"])

    def test_adapter_metadata_binds_tokenizer_semantics(self) -> None:
        metadata = adapter_checkpoint_metadata(
            {
                "adapter": {
                    "backend": "official",
                    "embed_dim": 32,
                    "vocab_size": 128,
                    "n_codes": 4,
                    "init_seed": 123,
                    "hidden_dim": 256,
                    "official_kwargs": {"patch_size": 8},
                    "allow_partial_checkpoint": True,
                }
            }
        )

        self.assertEqual(metadata["init_seed"], 123)
        self.assertEqual(metadata["hidden_dim"], 256)
        self.assertEqual(metadata["official_kwargs"], {"patch_size": 8})
        self.assertTrue(metadata["allow_partial_checkpoint"])

    def test_calvin_mini_config_uses_calvin_static_loader(self) -> None:
        cfg = load_config("configs/vw2_idm/data_calvin_mini.yaml")

        self.assertEqual(cfg["data"]["dataset_type"], "calvin_static")

    def test_rescue_audit_binds_cache_keys_and_controller_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_dir = root / "datasets" / "calvin_static"
            cache_dir = root / "cache"
            manifest_dir.mkdir(parents=True)
            cache_dir.mkdir()
            spec = WindowSpec(history_frames=4, past_action_hist=4, future_video_horizon=8, action_chunk=8, stride=4)
            train_manifest = {"episodes": [{"episode_id": "train_0", "root": "/remote/train", "start": 0, "end": 20}]}
            val_manifest = {"episodes": [{"episode_id": "val_0", "root": "/remote/val", "start": 100, "end": 120}]}
            save_json(train_manifest, manifest_dir / "train_manifest.json")
            save_json(val_manifest, manifest_dir / "val_manifest.json")
            train_windows = audit_rescued_artifacts.build_calvin_window_index(train_manifest["episodes"], spec)
            val_windows = audit_rescued_artifacts.build_calvin_window_index(val_manifest["episodes"], spec)
            save_json({"windows": train_windows}, cache_dir / "train_windows.json")
            save_json({"windows": val_windows}, cache_dir / "val_windows.json")
            torch.save({"records": [{"episode_id": item["episode_id"], "t": item["t"]} for item in train_windows]}, cache_dir / "train_local_codes.pt")
            torch.save({"records": [{"episode_id": item["episode_id"], "t": item["t"]} for item in val_windows]}, cache_dir / "val_local_codes.pt")
            controller_dirs = [
                "bc_vis_calvin_4090",
                "bc_vis_proprio_calvin_4090",
                "history_gt_calvin_4090",
                "pair_idm_calvin_4090",
                "vw2_hidden_mlp_action_head_calvin_4090",
            ]
            for controller in controller_dirs:
                controller_dir = root / "models" / controller
                checkpoint_dir = controller_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True)
                (controller_dir / "resolved_config.yaml").write_text(f"name: {controller}\n", encoding="utf-8")
                (controller_dir / "metrics.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
                save_json({"action_nll": 1.0, "action_mse": 2.0, "jerk": 3.0}, controller_dir / "offline_eval.json")
                torch.save({"model_metadata": {"controller": controller}}, checkpoint_dir / "best.pt")

            audit = audit_rescued_artifacts.audit_rescue(root)
            source_paths = {record["path"] for record in audit["source_files"]}

            self.assertFalse(audit["latent_cache"]["train_cache_has_duplicate_keys"])
            self.assertFalse(audit["latent_cache"]["val_cache_has_duplicate_keys"])
            self.assertTrue(audit["latent_cache"]["train_cache_keys_match_current_window_builder"])
            self.assertTrue(audit["latent_cache"]["val_cache_keys_match_configured_window_prefix"])
            self.assertIn("manifest_content_hash_boundary", audit)
            for controller in controller_dirs:
                self.assertIn(f"models/{controller}/resolved_config.yaml", source_paths)
                self.assertIn(f"models/{controller}/metrics.jsonl", source_paths)
                self.assertIn(f"models/{controller}/offline_eval.json", source_paths)
                self.assertIn(f"models/{controller}/checkpoints/best.pt", source_paths)

            duplicated_train_records = [
                {"episode_id": train_windows[0]["episode_id"], "t": train_windows[0]["t"]},
                {"episode_id": train_windows[0]["episode_id"], "t": train_windows[0]["t"]},
            ]
            torch.save({"records": duplicated_train_records}, cache_dir / "train_local_codes.pt")
            duplicate_audit = audit_rescued_artifacts.audit_rescue(root)
            self.assertTrue(duplicate_audit["latent_cache"]["train_cache_has_duplicate_keys"])
            self.assertFalse(duplicate_audit["latent_cache"]["train_cache_keys_match_current_window_builder"])

            torch.save({"records": [{"episode_id": item["episode_id"], "t": item["t"]} for item in train_windows]}, cache_dir / "train_local_codes.pt")
            shifted_val_records = [{"episode_id": item["episode_id"], "t": item["t"] + 1} for item in val_windows]
            torch.save({"records": shifted_val_records}, cache_dir / "val_local_codes.pt")
            shifted_audit = audit_rescued_artifacts.audit_rescue(root)
            self.assertFalse(shifted_audit["latent_cache"]["val_cache_has_duplicate_keys"])
            self.assertFalse(shifted_audit["latent_cache"]["val_cache_keys_match_configured_window_prefix"])
            self.assertFalse(shifted_audit["latent_cache"]["val_cache_is_configured_subset_of_full_val_index"])

    def test_phase1_packaging_requires_rescued_offline_eval_source(self) -> None:
        with mock.patch("sys.argv", ["package_phase1_results.py"]), mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                package_phase1_results.main()

        self.assertEqual(ctx.exception.code, 2)

    def test_phase1_packaging_marks_non_planner_accuracy_not_applicable(self) -> None:
        metrics = {
            "direct": {
                "action_nll": 1.0,
                "action_mse": 2.0,
                "jerk": 3.0,
                "planner_code_accuracy": 0.0,
            }
        }
        metadata = {
            "direct": {
                "conditioning": "direct_policy",
                "privileged_future_codes": False,
                "deployable_without_future_labels": True,
            }
        }

        packaged = package_phase1_results.package_metrics(metrics, metadata, "rescued_offline_eval_json")

        self.assertIsNone(packaged["direct"]["planner_code_accuracy"])

    def test_phase1_packaging_rejects_rescue_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            models_dir = root / "models"
            for rel_path in package_phase1_results.RESCUED_OFFLINE_EVALS.values():
                output_path = models_dir / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_json(
                    {
                        "action_nll": 1.0,
                        "action_mse": 2.0,
                        "jerk": 3.0,
                        "planner_code_accuracy": 0.0,
                    },
                    output_path,
                )
            audit_json = root / "audit.json"
            save_json(
                {
                    "source_files": [
                        {"path": f"models/{rel_path}", "sha256": "0" * 64}
                        for rel_path in package_phase1_results.RESCUED_OFFLINE_EVALS.values()
                    ]
                },
                audit_json,
            )

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                package_phase1_results.load_rescued_offline_metrics(models_dir, audit_json=audit_json)

    def test_phase1_packaging_validates_all_audited_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            controller_config = root / "models" / "direct" / "resolved_config.yaml"
            controller_config.parent.mkdir(parents=True)
            controller_config.write_text("policy:\n  variant: mlp\n", encoding="utf-8")
            audit_json = root / "audit.json"
            save_json(
                {
                    "source_files": [
                        {
                            "path": "models/direct/resolved_config.yaml",
                            "sha256": package_phase1_results._file_sha256(controller_config),
                        }
                    ]
                },
                audit_json,
            )

            package_phase1_results.validate_audited_source_files(root, audit_json)
            controller_config.write_text("policy:\n  variant: idm\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                package_phase1_results.validate_audited_source_files(root, audit_json)

    def test_phase1_packaging_rejects_non_finite_metrics(self) -> None:
        metrics = {
            "direct": {
                "action_nll": float("nan"),
                "action_mse": 2.0,
                "jerk": 3.0,
            }
        }
        metadata = {
            "direct": {
                "conditioning": "direct_policy",
                "privileged_future_codes": False,
                "deployable_without_future_labels": True,
            }
        }

        with self.assertRaisesRegex(ValueError, "Non-finite metric"):
            package_phase1_results.package_metrics(metrics, metadata, "rescued_offline_eval_json")

    def test_phase1_unaudited_packaging_marks_metric_origin(self) -> None:
        metrics = {
            "direct": {
                "action_nll": 1.0,
                "action_mse": 2.0,
                "jerk": 3.0,
                "planner_code_accuracy": 0.0,
            }
        }
        metadata = {
            "direct": {
                "conditioning": "direct_policy",
                "privileged_future_codes": False,
                "deployable_without_future_labels": True,
            }
        }

        packaged = package_phase1_results.package_metrics(metrics, metadata, "unaudited_rescued_offline_eval_json")
        prov = package_phase1_results.provenance(audited_rescue_hashes=False)

        self.assertEqual(packaged["direct"]["metric_origin"], "unaudited_rescued_offline_eval_json")
        self.assertFalse(prov["phase1_offline_metrics"]["audited_rescue_hashes"])

    def test_committed_non_planner_metrics_use_null_accuracy(self) -> None:
        phase0 = load_json("results/phase0_summaries.json")
        phase1 = load_json("results/phase1_offline_metrics.json")

        for row in phase1.values():
            self.assertIsNone(row["planner_code_accuracy"])
        for row in phase0.values():
            for section in ("offline", "closed_loop"):
                if section in row and "planner_code_accuracy" in row[section]:
                    self.assertIsNone(row[section]["planner_code_accuracy"])

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

    def test_idm_eval_rejects_semantic_variant_mismatch(self) -> None:
        from videoworld2.robot_idm.eval.eval_offline_idm import load_policy_bundle

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
                    "n_heads": 4,
                    "num_embodiments": 8,
                },
                "idm": {"variant": "history", "use_future_codes": True, "use_past_actions": True},
                "policy": {},
                "evaluation": {},
            }
            pair_cfg = {**cfg, "idm": {**cfg["idm"], "variant": "pair", "use_past_actions": False}}
            state_encoder = build_state_encoder(cfg)
            idm = build_idm(cfg, action_dim=2)
            checkpoint_path = root / "pair_as_history.pt"
            torch.save(
                {
                    "state_encoder": state_encoder.state_dict(),
                    "idm": idm.state_dict(),
                    "model_metadata": policy_checkpoint_metadata(pair_cfg, "idm", 2),
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_policy_bundle(cfg, str(checkpoint_path), torch.device("cpu"))

    def test_resume_rejects_semantic_variant_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            cfg = {
                "data": {"action_chunk": 8, "proprio_dim": 4, "use_proprio": True, "use_lang": True},
                "model": {"d_model": 64, "idm_depth": 2, "n_heads": 4, "num_embodiments": 8},
                "idm": {"variant": "history", "use_future_codes": True, "use_past_actions": True},
                "policy": {},
            }
            pair_cfg = {**cfg, "idm": {**cfg["idm"], "variant": "pair", "use_past_actions": False}}
            idm = build_idm(cfg, action_dim=2)
            torch.save(
                {
                    "epoch": 3,
                    "global_step": 3,
                    "best_metric": 1.23,
                    "idm": idm.state_dict(),
                    "model_metadata": policy_checkpoint_metadata(pair_cfg, "idm", 2),
                },
                checkpoint_dir / "last.pt",
            )

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                maybe_resume_training(root, modules={"idm": idm}, expected_metadata=policy_checkpoint_metadata(cfg, "idm", 2))

    def test_policy_metadata_rejects_conditioning_mismatch(self) -> None:
        from videoworld2.robot_idm.eval.eval_offline_idm import load_policy_bundle

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
                "idm": {"variant": "history", "code_source": "predicted", "planner_checkpoint": str(root / "planner.pt"), "use_future_codes": True, "use_past_actions": True},
                "policy": {},
                "evaluation": {},
            }
            train_cfg = {**cfg, "idm": {**cfg["idm"], "code_source": "gt", "planner_checkpoint": ""}}
            checkpoint_path = root / "history_gt.pt"
            state_encoder = build_state_encoder(cfg)
            idm = build_idm(cfg, action_dim=2)
            (root / "planner.pt").write_bytes(b"planner-a")
            torch.save(
                {
                    "state_encoder": state_encoder.state_dict(),
                    "idm": idm.state_dict(),
                    "model_metadata": policy_checkpoint_metadata(train_cfg, "idm", 2),
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_policy_bundle(cfg, str(checkpoint_path), torch.device("cpu"))

    def test_policy_metadata_rejects_planner_identity_mismatch(self) -> None:
        from videoworld2.robot_idm.eval.eval_offline_idm import load_policy_bundle

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            planner_a = root / "planner_a.pt"
            planner_b = root / "planner_b.pt"
            planner_a.write_bytes(b"planner-a")
            planner_b.write_bytes(b"planner-b")
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
                "idm": {"variant": "history", "code_source": "predicted", "planner_checkpoint": str(planner_b), "use_future_codes": True, "use_past_actions": True},
                "policy": {},
                "evaluation": {},
            }
            train_cfg = {**cfg, "idm": {**cfg["idm"], "planner_checkpoint": str(planner_a)}}
            checkpoint_path = root / "history_pred.pt"
            state_encoder = build_state_encoder(cfg)
            idm = build_idm(cfg, action_dim=2)
            torch.save(
                {
                    "state_encoder": state_encoder.state_dict(),
                    "idm": idm.state_dict(),
                    "model_metadata": policy_checkpoint_metadata(
                        train_cfg,
                        "idm",
                        2,
                        checkpoint_refs={"planner_checkpoint": checkpoint_reference(planner_a)},
                    ),
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_policy_bundle(cfg, str(checkpoint_path), torch.device("cpu"))

    def test_teacher_metadata_ignores_student_policy_variant(self) -> None:
        cfg = {
            "data": {"action_chunk": 8, "proprio_dim": 4, "use_proprio": True, "use_lang": True},
            "model": {"d_model": 64, "idm_depth": 2, "n_heads": 4, "num_embodiments": 8},
            "idm": {"variant": "history", "use_future_codes": True, "use_past_actions": True, "code_source": "gt"},
            "policy": {"variant": "mlp", "hidden_dim": 128},
        }
        teacher_cfg = {**cfg, "policy": {}}

        self.assertEqual(
            teacher_policy_checkpoint_metadata(cfg, 2),
            policy_checkpoint_metadata(teacher_cfg, "idm", 2),
        )

    def test_distill_teacher_uses_predicted_conditioning_when_configured(self) -> None:
        from videoworld2.robot_idm.train.distill_policy import teacher_future_code_embeds

        cfg = {"idm": {"use_future_codes": True, "code_source": "predicted", "mixed_code_training": False}}
        batch = {
            "future_codes": torch.zeros(2, 3, dtype=torch.long),
            "future_code_embeds": torch.zeros(2, 3, 4),
        }
        predicted_codes = torch.ones(2, 3, dtype=torch.long)
        predicted_embeds = torch.full((2, 3, 4), 5.0)
        planner = mock.Mock()
        planner.sample.return_value = predicted_codes
        adapter = mock.Mock()
        adapter.code_embed.return_value = predicted_embeds

        embeds = teacher_future_code_embeds(cfg, batch, torch.randn(2, 4, 8), adapter, planner)

        self.assertIs(embeds, predicted_embeds)
        planner.sample.assert_called_once()
        adapter.code_embed.assert_called_once_with(predicted_codes)

    def test_verifier_eval_loads_matching_verifier_encoder(self) -> None:
        from videoworld2.robot_idm.eval.eval_offline_idm import load_policy_bundle
        from videoworld2.robot_idm.utils.factory import build_direct_policy, build_verifier

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
                    "verifier_depth": 2,
                    "n_heads": 4,
                    "num_embodiments": 8,
                },
                "idm": {"variant": "bc", "code_source": "gt", "use_future_codes": False, "use_past_actions": False},
                "policy": {},
                "evaluation": {"verifier_checkpoint": str(root / "verifier.pt"), "use_verifier": True},
            }
            policy_encoder = build_state_encoder(cfg)
            direct_policy = build_direct_policy(cfg, action_dim=2)
            policy_checkpoint = root / "policy.pt"
            torch.save(
                {
                    "state_encoder": policy_encoder.state_dict(),
                    "direct_policy": direct_policy.state_dict(),
                    "model_metadata": policy_checkpoint_metadata(cfg, "direct_policy", 2),
                },
                policy_checkpoint,
            )

            verifier_encoder = build_state_encoder(cfg)
            verifier_state = verifier_encoder.state_dict()
            first_key = next(iter(verifier_state))
            verifier_state[first_key] = torch.ones_like(verifier_state[first_key])
            verifier = build_verifier(cfg, DLDMLocalAdapter(cfg["adapter"]), action_dim=2)
            torch.save(
                {
                    "state_encoder": verifier_state,
                    "verifier": verifier.state_dict(),
                    "model_metadata": auxiliary_checkpoint_metadata(cfg, "verifier", action_dim=2),
                },
                root / "verifier.pt",
            )

            _, loaded_policy_encoder, _, _, _, loaded_verifier_encoder, _ = load_policy_bundle(cfg, str(policy_checkpoint), torch.device("cpu"))

            self.assertIsNotNone(loaded_verifier_encoder)
            self.assertTrue(torch.equal(loaded_verifier_encoder.state_dict()[first_key], verifier_state[first_key]))
            self.assertFalse(torch.equal(loaded_policy_encoder.state_dict()[first_key], verifier_state[first_key]))

    def test_verifier_eval_marks_action_nll_not_applicable(self) -> None:
        from videoworld2.robot_idm.eval import eval_offline_idm

        cfg = {
            "adapter": {"backend": "mock_geometry", "embed_dim": 64},
            "data": {"dataset_type": "mock", "action_chunk": 8},
            "training": {"seed": 7, "gamma_discount": 0.97},
            "idm": {"variant": "bc", "code_source": "gt", "use_future_codes": False, "use_past_actions": False},
            "evaluation": {"use_verifier": True, "num_candidates": 2},
        }
        loader = _FakeLoader([_make_sample(action_value=1.0)], sample=_make_sample(action_value=1.0))
        policy_bundle = (_FakeAdapter(), _FakeStateEncoder(), _FakePolicy(mean_value=0.0), None, None, _FakeStateEncoder(), _FakeVerifier())

        with mock.patch.object(eval_offline_idm, "DLDMLocalAdapter", return_value=_FakeAdapter()), \
            mock.patch.object(eval_offline_idm, "ensure_code_caches"), \
            mock.patch.object(eval_offline_idm, "make_dataloaders", return_value=(None, loader)), \
            mock.patch.object(eval_offline_idm, "load_policy_bundle", return_value=policy_bundle):
            metrics = eval_offline_idm.evaluate_offline(dict(cfg), "checkpoint.pt", torch.device("cpu"))

        self.assertIsNone(metrics["action_nll"])
        self.assertGreaterEqual(metrics["action_mse"], 0.0)

    def test_offline_eval_rejects_non_finite_metrics(self) -> None:
        from videoworld2.robot_idm.eval import eval_offline_idm

        cfg = {
            "adapter": {"backend": "mock_geometry", "embed_dim": 64},
            "data": {"dataset_type": "mock", "action_chunk": 8},
            "training": {"seed": 7, "gamma_discount": 0.97},
            "idm": {"variant": "bc", "code_source": "gt", "use_future_codes": False, "use_past_actions": False},
            "evaluation": {},
        }
        loader = _FakeLoader([_make_sample(action_value=1.0)], sample=_make_sample(action_value=1.0))
        policy_bundle = (_FakeAdapter(), _FakeStateEncoder(), _FakePolicy(mean_value=float("nan")), None, None, None, None)

        with mock.patch.object(eval_offline_idm, "DLDMLocalAdapter", return_value=_FakeAdapter()), \
            mock.patch.object(eval_offline_idm, "ensure_code_caches"), \
            mock.patch.object(eval_offline_idm, "make_dataloaders", return_value=(None, loader)), \
            mock.patch.object(eval_offline_idm, "load_policy_bundle", return_value=policy_bundle):
            with self.assertRaisesRegex(ValueError, "Non-finite metric"):
                eval_offline_idm.evaluate_offline(dict(cfg), "checkpoint.pt", torch.device("cpu"))

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
                    "planner": {},
                    "model_metadata": auxiliary_checkpoint_metadata(cfg, "planner"),
                },
                root / "planner.pt",
            )
            torch.save(
                {
                    "state_encoder": state_encoder.state_dict(),
                    "direct_policy": direct_policy.state_dict(),
                    "model_metadata": policy_checkpoint_metadata(
                        cfg,
                        "direct_policy",
                        2,
                        checkpoint_refs={"planner_checkpoint": checkpoint_reference(root / "planner.pt")},
                    ),
                },
                policy_checkpoint,
            )

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
