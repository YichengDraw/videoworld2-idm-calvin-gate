from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from videoworld2.robot_idm.data.robot_window_dataset import RobotWindowDataset, WindowSpec, build_window_index
from videoworld2.robot_idm.models.forward_verifier import ForwardVerifier
from videoworld2.robot_idm.models.inverse_dynamics import HistoryAwareIDM
from videoworld2.robot_idm.models.latent_planner import LatentPlanner
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

    def test_planner_output_shape(self) -> None:
        planner = LatentPlanner(vocab_size=32, n_codes=4, d_model=64, depth=2, n_heads=4)
        state_tokens = torch.randn(3, 6, 64)
        target_codes = torch.randint(0, 32, (3, 4))
        logits = planner(state_tokens, target_codes=target_codes)
        self.assertEqual(logits.shape, (3, 4, 32))

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


if __name__ == "__main__":
    unittest.main()
