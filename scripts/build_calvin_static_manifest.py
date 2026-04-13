from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoworld2.robot_idm.utils.runtime import save_json


def build_manifest(split_root: str | Path, limit_segments: int | None = None) -> dict[str, object]:
    split_root = Path(split_root).resolve()
    ann_path = split_root / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing CALVIN language annotations: {ann_path}")

    annotations = np.load(ann_path, allow_pickle=True).item()
    spans = list(annotations["info"]["indx"])
    langs = list(annotations["language"]["ann"])
    if limit_segments is not None:
        spans = spans[:limit_segments]
        langs = langs[:limit_segments]

    unique_langs = {lang: idx for idx, lang in enumerate(sorted(set(str(lang) for lang in langs)))}
    episodes = []
    for idx, (span, lang) in enumerate(zip(spans, langs)):
        start, end = int(span[0]), int(span[1])
        first_frame = split_root / f"episode_{start:07d}.npz"
        last_frame = split_root / f"episode_{end:07d}.npz"
        if not first_frame.exists() or not last_frame.exists():
            continue
        episodes.append(
            {
                "episode_id": f"{split_root.name}_{idx:06d}",
                "root": str(split_root),
                "start": start,
                "end": end,
                "lang": str(lang),
                "task_id": unique_langs[str(lang)],
                "embodiment_id": 0,
            }
        )

    return {
        "root": str(split_root),
        "episodes": episodes,
        "num_tasks": len(unique_langs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_root", type=str)
    parser.add_argument("output_manifest", type=str)
    parser.add_argument("--limit-segments", type=int, default=None)
    args = parser.parse_args()

    payload = build_manifest(args.split_root, limit_segments=args.limit_segments)
    save_json(payload, args.output_manifest)
    print(
        {
            "split_root": payload["root"],
            "episodes": len(payload["episodes"]),
            "num_tasks": payload["num_tasks"],
        }
    )


if __name__ == "__main__":
    main()
