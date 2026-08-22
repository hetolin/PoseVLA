import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from .model import get_model, reset_model


_norm_stats_cache = {}


def _load_normalization_stats():
    norm_path = os.environ["PI05_BASELINE_NORM_PATH"]
    if norm_path not in _norm_stats_cache:
        with open(norm_path, "rb") as stats_file:
            stats = pickle.load(stats_file)
        _norm_stats_cache[norm_path] = (
            np.asarray(stats["action_mean"], dtype=np.float32),
            np.asarray(stats["action_std"], dtype=np.float32),
            np.asarray(stats["qpos_min"], dtype=np.float32),
            np.asarray(stats["qpos_max"], dtype=np.float32),
        )
    return _norm_stats_cache[norm_path]


def _resize_with_pad(image, size=224):
    _, _, height, width = image.shape
    ratio = max(width / size, height / size)
    resized_height = max(1, int(height / ratio))
    resized_width = max(1, int(width / ratio))
    image = F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    pad_height = size - resized_height
    pad_width = size - resized_width
    return F.pad(
        image,
        (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ),
    )


def _convert_image(image):
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    return _resize_with_pad(tensor / 255.0).unsqueeze(0)


def encode_obs(observation, instruction):
    _, _, qpos_min, qpos_max = _load_normalization_stats()
    raw_state = np.asarray(
        observation["joint_action"]["vector"],
        dtype=np.float32,
    )[None, :]
    state_range = np.where(
        qpos_max - qpos_min < 1e-5,
        1.0,
        qpos_max - qpos_min,
    )
    normalized_state = 2.0 * (raw_state - qpos_min) / state_range - 1.0
    normalized_state = np.clip(
        normalized_state,
        -1.0,
        1.0,
    ).astype(np.float32)

    padded_state = np.zeros((1, 32), dtype=np.float32)
    padded_state[:, : normalized_state.shape[-1]] = normalized_state
    cameras = observation["observation"]
    return {
        "observation.images.top_head": _convert_image(
            cameras["head_camera"]["rgb"]
        ),
        "observation.images.hand_left": _convert_image(
            cameras["left_camera"]["rgb"]
        ),
        "observation.images.hand_right": _convert_image(
            cameras["right_camera"]["rgb"]
        ),
        "observation.state": padded_state,
        "task": [instruction],
    }


def eval(
    task_env,
    model,
    observation,
    action_type="pi05",
    use_prior=False,
):
    if action_type != "pi05":
        raise ValueError("PI0.5 baseline requires action_type='pi05'.")
    if use_prior:
        raise ValueError("PI0.5 baseline requires add_prior=false.")

    batch = encode_obs(observation, task_env.get_instruction())
    batch_numpy = {
        key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    action = model.get_action(batch_numpy)
    action_mean, action_std, _, _ = _load_normalization_stats()
    task_env.take_action(action * action_std + action_mean)
