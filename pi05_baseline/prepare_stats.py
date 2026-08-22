"""Compute PI0.5 qpos/action normalization statistics from RoboTwin HDF5."""

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def update_moments(
    values: np.ndarray,
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    for row in values:
        count += 1
        delta = row - mean
        mean += delta / count
        m2 += delta * (row - mean)
    return count, mean, m2


def compute_stats(dataset_root: Path) -> dict:
    episode_files = list(dataset_root.rglob("episode_*.hdf5"))
    if not episode_files:
        raise RuntimeError(f"No episode_*.hdf5 files found under {dataset_root}")

    action_count = 0
    action_mean = None
    action_m2 = None
    qpos_min = None
    qpos_max = None

    for episode_path in tqdm(episode_files, desc="PI0.5 normalization"):
        with h5py.File(episode_path, "r") as episode:
            actions = episode["actions_qpos"][()].astype(np.float64)
            qpos = episode["observations/state_qpos"][()].astype(np.float64)

        if action_mean is None:
            action_mean = np.zeros(actions.shape[-1], dtype=np.float64)
            action_m2 = np.zeros(actions.shape[-1], dtype=np.float64)
            qpos_min = np.full(qpos.shape[-1], np.inf, dtype=np.float64)
            qpos_max = np.full(qpos.shape[-1], -np.inf, dtype=np.float64)

        action_count, action_mean, action_m2 = update_moments(
            actions,
            action_count,
            action_mean,
            action_m2,
        )
        qpos_min = np.minimum(qpos_min, qpos.min(axis=0))
        qpos_max = np.maximum(qpos_max, qpos.max(axis=0))

    action_std = np.sqrt(action_m2 / max(action_count - 1, 1))
    action_std = np.maximum(action_std, 1e-4)

    qpos_range = qpos_max - qpos_min
    qpos_range = np.where(qpos_range < 1e-5, 2.0, qpos_range)
    qpos_midpoint = (qpos_min + qpos_max) / 2.0
    qpos_half_range = qpos_range / 2.0

    return {
        "action_mean": action_mean.astype(np.float32),
        "action_std": action_std.astype(np.float32),
        "qpos_min": qpos_min.astype(np.float32),
        "qpos_max": qpos_max.astype(np.float32),
        # The unchanged shared dataset computes (state-mean)/std. These
        # derived values make it exactly equal to min-max mapping to [-1, 1].
        "qpos_mean": qpos_midpoint.astype(np.float32),
        "qpos_std": qpos_half_range.astype(np.float32),
        "total_count": action_count,
        "mode": "qpos",
        "normalization_info": {
            "state": "min_max_minus1_to_1",
            "action": "mean_std",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing robotwin_processed* HDF5 directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output qpos_hybrid_stats.pkl path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compute_stats(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output_file:
        pickle.dump(stats, output_file)
    print(f"Saved PI0.5 normalization stats to {args.output}")


if __name__ == "__main__":
    main()
