import sys, os
try:
    from .model import *
except (ImportError, ValueError):
    from model import *
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


_norm_stats_cache = {}

def resize_with_pad(img, width, height, pad_value=0, mode="bilinear"):
    """Resize a (B, C, H, W) tensor while preserving aspect ratio, padding to
    the target (height, width) with ``pad_value``.

    Args:
        img: torch.Tensor of shape (B, C, H, W).
        width / height: target spatial size.
        pad_value: value used for the border padding.
        mode: interpolation mode. Use ``"nearest"`` to preserve discrete mask
            values (0 / 0.5 / 1.0); otherwise ``"bilinear"`` for RGB images.

    Returns:
        torch.Tensor of shape (B, C, height, width).
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, got {tuple(img.shape)}")
    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    rh = max(1, int(round(cur_h / ratio)))
    rw = max(1, int(round(cur_w / ratio)))
    kw = {"size": (rh, rw), "mode": mode}
    if mode != "nearest":
        kw["align_corners"] = False
    resized = F.interpolate(img, **kw)
    ph = max(0, height - rh)
    pw = max(0, width - rw)
    return F.pad(
        resized,
        (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2),
        value=pad_value,
    )


def pad_vector(vector, new_dim):
    """Pad the last dim of a numpy/torch vector to ``new_dim`` with zeros."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    shape[-1] = new_dim
    if isinstance(vector, torch.Tensor):
        new_vector = torch.zeros(shape, dtype=vector.dtype, device=vector.device)
        new_vector[..., : vector.shape[-1]] = vector
        return new_vector
    dtype = vector.dtype if hasattr(vector, "dtype") else np.float32
    new_vector = np.zeros(shape, dtype=dtype)
    new_vector[..., : vector.shape[-1]] = vector
    return new_vector


def convert_obs(img, target_size=224):
    """RGB numpy (H, W, 3) uint8 → torch tensor (1, 3, target_size, target_size) in [0, 1]."""
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    resized = resize_with_pad(
        tensor.unsqueeze(0),  # (1, 3, H, W)
        width=target_size,
        height=target_size,
        pad_value=0,
        mode="bilinear",
    )
    return resized  # (1, 3, target_size, target_size)


def _load_normalization_stats(action_type="qpos"):
    import pickle

    default_norm_path = (
        "/home/pub_data/hanyangyu/Datasets/robotwin_processed_random/"
        "global_stats_output_eep/qpos_mean_std_online.pkl"
    )
    norm_path = os.environ.get("POSEVLA_NORM_PATH", default_norm_path)
    cache_key = (action_type, norm_path)
    if cache_key in _norm_stats_cache:
        return _norm_stats_cache[cache_key]

    with open(norm_path, "rb") as f:
        stats = pickle.load(f)

    state_key = f"{action_type}_mean"
    state_std_key = f"{action_type}_std"
    result = (
        stats["action_mean"].astype(np.float32),
        stats["action_std"].astype(np.float32),
        stats[state_key].astype(np.float32),
        stats[state_std_key].astype(np.float32),
    )
    _norm_stats_cache[cache_key] = result
    return result


def _extract_gripper(joint_action, endpose, side):
    if f"{side}_gripper" in endpose:
        return np.asarray(endpose[f"{side}_gripper"]).reshape(-1)[0]
    if f"{side}_gripper" in joint_action:
        return np.asarray(joint_action[f"{side}_gripper"]).reshape(-1)[0]
    if side == "left" and "vector" in joint_action:
        return np.asarray(joint_action["vector"]).reshape(-1)[6]
    if side == "right" and "vector" in joint_action:
        return np.asarray(joint_action["vector"]).reshape(-1)[13]
    raise KeyError(f"Cannot find {side} gripper in observation['endpose'] or observation['joint_action']")


def _extract_state_vector(observation, action_type):
    joint_action = observation.get("joint_action", {})
    if action_type == "qpos":
        return np.asarray(joint_action["vector"], dtype=np.float32)

    if "vector" in joint_action and len(np.asarray(joint_action["vector"]).reshape(-1)) == 16:
        return np.asarray(joint_action["vector"], dtype=np.float32)

    endpose = observation.get("endpose", {})
    if "left_endpose" in endpose and "right_endpose" in endpose:
        left_endpose = np.asarray(endpose["left_endpose"], dtype=np.float32).reshape(-1)
        right_endpose = np.asarray(endpose["right_endpose"], dtype=np.float32).reshape(-1)
        left_gripper = _extract_gripper(joint_action, endpose, "left")
        right_gripper = _extract_gripper(joint_action, endpose, "right")
        return np.concatenate(
            [left_endpose, [left_gripper], right_endpose, [right_gripper]],
            axis=0,
        ).astype(np.float32)

    raise KeyError(
        "Robotwin EEP deployment needs either joint_action['vector'] with 16 dims "
        "or observation['endpose']['left_endpose'/'right_endpose'] plus grippers."
    )


def encode_obs(observation, instruction, action_type):  # Post-Process Observation
    head_img = observation["observation"]["head_camera"]["rgb"]  # (H, W, 3)
    left_img = observation["observation"]["left_camera"]["rgb"]  # (H, W, 3)
    right_img = observation["observation"]["right_camera"]["rgb"]  # (H, W, 3)

    action_mean, action_std, state_mean, state_std = _load_normalization_stats(action_type)
    raw_state = _extract_state_vector(observation, action_type)[np.newaxis, :]
    state_std_safe = np.where(state_std < 1e-5, 1.0, state_std).astype(np.float32)
    normalized_state = ((raw_state - state_mean) / state_std_safe).astype(np.float32)

    batch = {
        'observation.images.top_head': convert_obs(head_img).unsqueeze(0),
        'observation.images.hand_left': convert_obs(left_img).unsqueeze(0),
        'observation.images.hand_right': convert_obs(right_img).unsqueeze(0),
        'observation.state': pad_vector(normalized_state, new_dim=32),
        'task': [instruction],
    }

    return batch



_save_video_enabled = False
_video_save_dir = None
_episode_count = 0
_step_count = 0


def enable_video_saving(task_name=None, log_dir=None):
    """Prepare (but don't enable) on-disk prediction-frame logging.

    Keeps the dir/counters initialized so that `save_observation_images` is a
    no-op unless the user flips `_save_video_enabled` to True manually.
    """
    global _save_video_enabled, _video_save_dir, _episode_count, _step_count

    _save_video_enabled = False

    base_log_dir = log_dir or str(Path(__file__).resolve().parent / "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir_name = task_name or "default_task"
    _video_save_dir = Path(base_log_dir) / f"logs_{timestamp}" / "images" / task_dir_name
    try:
        _video_save_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Directory creation is best-effort; saving is disabled by default.
        pass

    _episode_count = 0
    _step_count = 0
    print(f"[Video Saving] Initialized (disabled). Would save to: {_video_save_dir}")


def reset_video_episode():
    """Reset per-episode counters for the predicted-frame logger."""
    global _episode_count, _step_count
    _episode_count += 1
    _step_count = 0
    if _save_video_enabled:
        print(f"[Video Saving] New episode: {_episode_count}")


def _create_prediction_grid(predicted_frames):
    """Build a horizontal PIL grid from predicted frames [C, T, H, W]."""
    num_frames = min(predicted_frames.shape[1], 8)
    frames_list = []
    for t in range(num_frames):
        frame = predicted_frames[:, t, :, :]  # [C, H, W]
        frame = frame.permute(1, 2, 0).detach().cpu().float()
        frame = torch.clamp(frame, 0, 1)
        frame_np = (frame.numpy() * 255).astype(np.uint8)
        frames_list.append(frame_np)
    return Image.fromarray(np.concatenate(frames_list, axis=1))


def save_observation_images(observation, predicted_frames=None):
    """Persist predicted video frames to disk (no-op unless enabled)."""
    global _step_count
    if not _save_video_enabled or predicted_frames is None:
        return
    try:
        if predicted_frames.dim() == 5:
            predicted_frames = predicted_frames.squeeze(0)
        grid_image = _create_prediction_grid(predicted_frames)
        filename = f"ep{_episode_count:03d}_step{_step_count:04d}.png"
        grid_image.save(_video_save_dir / filename)
        print(f"Saved: {filename}")
        _step_count += 1
    except Exception as e:
        print(f"Error saving: {e}")


def eval(TASK_ENV, model, observation, action_type='eep', use_prior=False,
         current_seg_mask=None):
    """x
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged

    """
    instruction = TASK_ENV.get_instruction()

    batch = encode_obs(
        observation, instruction, action_type,
    )  # Post-Process Observation

    batch_numpy = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):   
            batch_numpy[key] = value.cpu().numpy()
        else:

            batch_numpy[key] = value
    
    # Check if we need to generate new action chunk (queue is empty)
    is_new_chunk = len(model.action_cache) == 0
    
    action = model.get_action(batch_numpy) # Get Action according to observation chunk

    # ------------------------train use norm or not !!!!-----------------------------
    action_mean, action_std, _, _ = _load_normalization_stats(action_type)
    action = action * action_std + action_mean  # Denormalize Action
    
    # Only save predicted frames when generating new action chunk
    if is_new_chunk and _save_video_enabled:
        predicted_frames = getattr(model, 'latest_predicted_frames', None)
        # predicted_frames: [B, C, T, H, W] -> squeeze batch dim -> [C, T, H, W]
        predicted_frames = predicted_frames.squeeze(0) if predicted_frames.dim() == 5 else predicted_frames
        save_observation_images(observation, predicted_frames)

    if action_type == 'eep':
        TASK_ENV.take_action(action, action_type='ee')
    else:
        TASK_ENV.take_action(action)
