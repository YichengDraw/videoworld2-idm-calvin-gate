"""Robot IDM prototype package for VideoWorld2."""

from .data.robot_window_dataset import RobotWindowDataset, WindowSpec, build_window_index
from .utils.latent_cache import LatentCodeCache

__all__ = [
    "LatentCodeCache",
    "RobotWindowDataset",
    "WindowSpec",
    "build_window_index",
]
