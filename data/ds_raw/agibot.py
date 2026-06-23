import os
import json
import random

import h5py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from embodied_pi0_action.utils.vis import get_history_indices, visualize_traj, visualize_2d_3d_all
from mapping_token import text_to_class_attr_dict_tokenizer, map_3d_label_to_string_tokenizer, BinTokenizer, repeat_attribute_to_match
from scipy.spatial.transform import Rotation
import torch
import embodied_pi0_action.data.normalize as normalize

DEFAULT_LEFT_HAND_INTRINSICS = np.array([[326.9194059551887, 0, 323.848876953125],
                                         [0, 432.5560302734375, 239.00411987304688],
                                         [0, 0, 1]])
DEFAULT_RIGHT_HAND_INTRINSICS = np.array([[328.54630380306605, 0, 318.35205078125],
                                         [0, 434.81304931640625, 237.24725341796875],
                                         [0, 0, 1]])
DEFAULT_HEAED_INTRINSICS = np.array([[322.47344970703125, 0, 325.61358642578125,],
                                         [0, 429.38614908854163, 239.24239095052081],
                                         [0, 0, 1]])

def project_points_to_2d(points, K):
    """
    points: (N,3), 在相机坐标系下
    K: (3,3), 内参矩阵
    返回: (N,2), 2D像素坐标
    """
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    valid = Z > 0  # 只保留可见点

    # 投影公式
    u = K[0,0] * X[valid]/Z[valid] + K[0,2]
    v = K[1,1] * Y[valid]/Z[valid] + K[1,2]
    pixels = np.stack([u, v], axis=1)  # (N_valid, 2)
    return pixels, valid          # 返回所有可见点的像素值和mask

# def project_points_to_2d_valid(points, K):
#     """
#     points: (N,3) 相机坐标系下的3D点
#     K: (3,3) 内参
#     返回: (N_valid,2)
#     """
#     X = points[:, 0]
#     Y = points[:, 1]
#     Z = points[:, 2]
#     valid_mask = Z > 0
#
#     if not np.any(valid_mask):
#         return np.empty((0,2))  # 没有valid点
#
#     X_valid = X[valid_mask]
#     Y_valid = Y[valid_mask]
#     Z_valid = Z[valid_mask]
#
#     u = K[0,0] * X_valid / Z_valid + K[0,2]
#     v = K[1,1] * Y_valid / Z_valid + K[1,2]
#     pixels = np.stack([u, v], axis=1)  # (N_valid, 2)
#     return pixels

def is_group_valid(action_2d, W, H):
    """
    actionl_in_head_2d: (N,2)一组点, 每个点[u, v]
    W: 图片宽, H: 高

    返回: bool, True=整组都有效, False=有点超出则全部无效
    """
    u = action_2d[:,0]
    v = action_2d[:,1]

    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    # 如果全部点都在范围内则有效，否则无效
    return np.all(valid_mask)

def pose_to_transformation_matrix_scipy(pose):
    # pose: (N,7): [x, y, z, qx, qy, qz, qw]
    t = pose[:, :3]  # (N,3)
    q = pose[:, 3:]  # (N,4) x,y,z,w
    rot = Rotation.from_quat(q)  # 批量处理
    R = rot.as_matrix()  # (N,3,3)

    N = pose.shape[0]
    mats = np.zeros((N, 4, 4), dtype=np.float64)
    mats[:, :3, :3] = R
    mats[:, :3, 3] = t
    mats[:, 3, 3] = 1
    return mats

def mat44_to_quat_trans(mat44):
    """
    输入: mat44 (N,4,4) torch.Tensor 或 np.ndarray
    输出: (N,7) torch.Tensor [q0, q1, q2, q3, tx, ty, tz]
    """
    if isinstance(mat44, torch.Tensor):
        mat44_np = mat44.cpu().numpy()
    else:
        mat44_np = mat44

    R_mats = mat44_np[:, :3, :3]   # (N,3,3)
    T_vecs = mat44_np[:, :3, 3]    # (N,3)
    quats = Rotation.from_matrix(R_mats).as_quat()  # (N,4) [x,y,z,w]
    # 通常你的格式是[w, x, y, z]，注意as_quat返回的是[x,y,z,w]
    quats = np.concatenate([quats[:, 3:4], quats[:, :3]], axis=1)  # 变成[w, x, y, z]
    out = np.concatenate([quats, T_vecs], axis=1)  # (N,7)
    return torch.from_numpy(out) if isinstance(mat44, torch.Tensor) else out


def pad_vector(vector, new_dim):
    """Can be (sequence_length x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = np.zeros(shape)
    new_vector[..., :current_dim] = vector
    return new_vector

class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """

    def __init__(self, cfg, sample_weights=dict, **kwargs) -> None:
        self.cfg = cfg
        self.action_type = kwargs.get("action_type", "joint")
        print(f"=====agibot action type: {self.action_type}")
        self.future_video = kwargs.get("future_video", False)
        print(f"=====agibot future_video: {self.future_video}")
        self.global_downsample_rate = kwargs.get("global_downsample_rate", 1)
        print(f"=====agibot global_downsample_rate: {self.global_downsample_rate}")
        self.tokenizer = kwargs.get("tokenizer", None)

        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIR = cfg.dataset.hdf5_dir #"data/datasets/agilex/rdt_data/"

        self.DATASET_NAMES = list(sample_weights.keys())
        SAMPLE_WEIGHTS = sample_weights

        # Load the config
        self.CHUNK_SIZE = cfg.dataset.action_chunk_size
        self.IMG_HISORY_SIZE = cfg.dataset.img_history_size
        self.STATE_DIM = cfg.dataset.state_dim

        # Load norm statistics
        self.use_self_norm = False
        if False:
        # if hasattr(cfg.dataset, 'mean_std_path'):
            self.use_self_norm = True
            # mean_std_path = cfg.dataset.mean_std_path
            mean_std_path = "mydata/agibot_beta_stats"

            # Check if the file exists
            if os.path.exists(mean_std_path):
                norm_stats = normalize.load(mean_std_path)
                self.qpos_mean = norm_stats["state"].mean[None, :] #(1,20)
                self.qpos_std = norm_stats["state"].std[None, :]
                self.act_mean = norm_stats["action"].mean #(1,20)
                self.act_std = norm_stats["action"].std
                print(f"===Load mean_std from {mean_std_path}")
                print(self.qpos_mean)
                print(self.qpos_std)
            else:
                # Raise a ValueError if the file does not exist
                raise ValueError(f"File does not exist: {mean_std_path}")

        # Video generation parameters
        self.num_video_frames = cfg.dataset.num_video_frames
        self.video_action_freq_ratio = 6 # Video:Action frequency ratio to keep video at 5hz
        # self.CHUNK_SIZE = self.num_video_frames * self.video_action_freq_ratio # Number of actions to predict (num_video_frames * video_action_freq_ratio)

        # 1. for weight sampling fetching when index is None in get_item()
        self.file_paths_dict = {}
        for dataset_name in self.DATASET_NAMES:
            self.file_paths_dict[dataset_name] = []

        dataset_json_path = Path(self.cfg.dev_dir) / 'config' / "dataset_meta" / "hdf5_agibot_list.json"
        if os.path.exists(dataset_json_path):
            print("Find previous saved hdf5 file list. Loading Now ===========================")
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                self.file_paths_dict = json.load(f)

            print(f"total_hdf5: {self.file_paths_dict['total_hdf5']}", )
            print(f"valid_hdf5: {self.file_paths_dict['total_valid_hdf5']}")
            print(f"valid_timesteps: {self.file_paths_dict['total_valid_timesteps']}")
        else:
            print("Not find previous saved hdf5 file list. Searching Now ===========================")
            for dataset_name in self.DATASET_NAMES:
                print(f"find hdf5 {dataset_name}. Waiting...")
                is_realworld = False if "sim" in dataset_name else True
                # self.file_paths_dict[dataset_name] = self.find_hdf5_files(f"{HDF5_DIR}/{dataset_name}", is_realworld)
                self.file_paths_dict[dataset_name] = self.find_match(f"{HDF5_DIR}/{dataset_name}", is_realworld)
                print(f"find hdf5 {dataset_name} Done!")


            # self.sample_weights_dataset = {}
            self.num_total_episode = 0
            self.num_valid_timestep = 0
            self.num_valid_episode = 0
            for dataset_name in self.DATASET_NAMES:
                is_realworld = False if "sim" in dataset_name else True
                episode_lens = []
                # for each dataset, get each episode's len
                for file_path in tqdm(self.file_paths_dict[dataset_name]):
                    valid, res = self.parse_hdf5_file_state_only(file_path, is_realworld)
                    _len = res['state'].shape[0] if valid else 0
                    episode_lens.append(_len)
                    self.num_valid_timestep += _len
                    self.num_total_episode += 1
                    if _len > 0 :
                        self.num_valid_episode += 1

                self.file_paths_dict[f"{dataset_name}_episode_weights"] = (np.array(episode_lens) / np.sum(episode_lens)).tolist()
                self.file_paths_dict[f"{dataset_name}_episode_timesteps"] = episode_lens
                self.file_paths_dict[f"{dataset_name}_total_timesteps"] = int(np.sum(episode_lens))

            print(f"total_hdf5: {self.num_total_episode}")
            print(f"valid_hdf5: {self.num_valid_episode}")
            print(f"valid_timesteps: {self.num_valid_timestep}")
            self.file_paths_dict["total_hdf5"] = int(self.num_total_episode)
            self.file_paths_dict["total_valid_hdf5"] = int(self.num_valid_episode)
            self.file_paths_dict["total_valid_timesteps"] = int(self.num_valid_timestep)

            with open(dataset_json_path, "w", encoding="utf-8") as f:
                json.dump(self.file_paths_dict, f, ensure_ascii=False, indent=4)


        dataset_weights = []
        for dataset_name in self.DATASET_NAMES:
            dataset_weights.append(SAMPLE_WEIGHTS[dataset_name])

        dataset_weights = np.array(dataset_weights)
        self.dataset_weights = dataset_weights / np.sum(dataset_weights)
        for i, dataset_name in enumerate(self.DATASET_NAMES):
            print(f"+ Dataset {dataset_name} weight: {self.dataset_weights[i]}")


    def __len__(self):
        # return len(self.file_paths)
        return sum(len(self.file_paths_dict[dataset_name]) for dataset_name in self.DATASET_NAMES)
        # return sum(len(v) for v in self.file_paths_dict.values())

    def get_dataset_name(self):
        return self.DATASET_NAMES

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                dataset_name = np.random.choice(self.DATASET_NAMES, p=self.dataset_weights)
                file_path = np.random.choice(self.file_paths_dict[dataset_name], p=self.file_paths_dict[f"{dataset_name}_episode_weights"])
                #file_path = np.str_('mydata/agibot_beta/proprio_stats/327/648642/proprio_stats.h5')
            else:
                for dataset_name in self.DATASET_NAMES:
                    dataset_len = len(self.file_paths_dict[dataset_name])
                    if index < dataset_len:
                        local_idx = index
                        file_path = self.file_paths_dict[dataset_name][local_idx]
                    # index超出了本数据集范围，则减去本数据集的长度，继续找下一个数据集
                    else:
                        index -= dataset_len
            is_realworld = False if "sim" in dataset_name else True
            valid, sample = self.parse_hdf5_file(file_path, dataset_name, is_realworld) \
                if not state_only else self.parse_hdf5_file_state_only(file_path, is_realworld)
            if valid:
                return sample

    def __getitem__(self, index):
        return self.get_item()

    def find_hdf5_files(self, dataset_root, is_realworld):

        if is_realworld:
            proprio_stats_dir = os.path.join(dataset_root, "proprio_stats")
        else:
            proprio_stats_dir = os.path.join(dataset_root, "meta_info")
        hdf5_files = []
        task_dirs = os.listdir(proprio_stats_dir)

        for task_dir in tqdm(task_dirs):
            if task_dir.endswith('.tar'):
                continue
            scenes = os.listdir(os.path.join(proprio_stats_dir, task_dir))
            for scene in scenes:
                if not scene.endswith('.tar'):  # 非压缩文件
                    scene_path = os.path.join(proprio_stats_dir, task_dir, scene)
                    # if os.path.isdir(scene_path) and len(scene) == 6:
                    if os.path.isdir(scene_path):
                        if is_realworld:
                            hdf5_file = os.path.join(scene_path, "proprio_stats.h5")
                        else:
                            hdf5_file = os.path.join(scene_path, "proprio_states.h5")
                        if os.path.exists(hdf5_file):
                            hdf5_files.append(hdf5_file)
        print(f"Total tar file: {len(hdf5_files)}")

        return hdf5_files

    def find_match(self, dataset_root, is_realworld):
        match_files = []

        observation_dir = os.path.join(dataset_root, "observations")

        task_dirs = os.listdir(observation_dir)

        for task_dir in tqdm(task_dirs):
            if task_dir.endswith('.tar'):
                continue
            scenes = os.listdir(os.path.join(observation_dir, task_dir))
            for scene in scenes:
                if not scene.endswith('.tar'):
                    video_file_dir = os.path.join(observation_dir, task_dir, scene)
                    if is_realworld:
                        state_file_path = os.path.join(observation_dir, task_dir, scene, "proprio_stats.h5")
                        state_file_path = state_file_path.replace("observations", "proprio_stats")
                    else:
                        state_file_path = os.path.join(observation_dir, task_dir, scene, "proprio_states.h5")
                        state_file_path = state_file_path.replace("observations", "meta_info")
                    if os.path.exists(state_file_path):
                        match_files.append(state_file_path)

        print(f"Total video-state match files: {len(match_files)}")
        return match_files

    def combine_views_single_np(self, head_img, left_img, right_img):
        """Combine three RGB images into a T-shaped layout (head full size, wrists half size)."""
        target_h, target_w = head_img.shape[:2]
        left_small = cv2.resize(left_img, (target_w // 2, target_h // 2), interpolation=cv2.INTER_LINEAR)
        right_small = cv2.resize(right_img, (target_w // 2, target_h // 2), interpolation=cv2.INTER_LINEAR)

        bottom = np.concatenate([left_small, right_small], axis=1)
        combined = np.concatenate([head_img, bottom], axis=0)
        return combined

    def process_data(self, cam_high, cam_left_wrist, cam_right_wrist, cam_high_future, cam_left_wrist_future, cam_right_wrist_future):
        # --- 1. 合并时间轴 (Current + Future) ---
        # 假设 cam_high[-1] 是当前帧，cam_high_future 是未来序列
        high_clip_raw = [cam_high[-1]] + list(cam_high_future)
        left_clip_raw = [cam_left_wrist[-1]] + list(cam_left_wrist_future)
        right_clip_raw = [cam_right_wrist[-1]] + list(cam_right_wrist_future)

        # --- 2. 分别对三个视角做增广 (每个视角参数独立，但时间轴一致) ---
        # 如果你希望三个视角参数也一样，就手动生成一套 M 传进去
        high_clip_aug = self.augment_video_clip(high_clip_raw)
        left_clip_aug = self.augment_video_clip(left_clip_raw)
        right_clip_aug = self.augment_video_clip(right_clip_raw)

        # --- 3. 重新拼接 T 型布局 ---
        combined_clip = []
        for h_img, l_img, r_img in zip(high_clip_aug, left_clip_aug, right_clip_aug):
            # 复用你之前的拼接逻辑
            # 这里的 combine_views_single_np 内部就不需要再做增广了，只负责 resize 和 concat
            combined = self.combine_views_single_np(h_img, l_img, r_img)
            combined_clip.append(combined)

        # --- 4. 拆分回 current 和 future ---
        current_image = combined_clip[0]
        future_images = np.stack(combined_clip[1:], axis=0)

        return current_image, future_images

    def augment_video_clip(self, clip, angle_range=(-5, 5), crop_ratio=0.9):
        """
        对一个视频序列（List of np.ndarray）施加相同的随机旋转和裁剪。
        """
        h, w = clip[0].shape[:2]

        # 1. 随机生成该序列唯一的参数
        angle = np.random.uniform(*angle_range)

        # 计算旋转矩阵
        # 为了减少黑边，我们可以在旋转的同时稍微放大一点（scale > 1.0）
        # 或者保持 1.0，后面手动裁剪
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

        # 2. 预计算裁剪边界
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        sy, sx = (h - new_h) // 2, (w - new_w) // 2

        augmented_clip = []
        for img in clip:
            # 旋转：borderMode=cv2.BORDER_CONSTANT (黑边) 或 BORDER_REPLICATE (边缘拉伸)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # 裁剪：切掉边缘可能出现的黑边
            cropped = rotated[sy:sy + new_h, sx:sx + new_w]
            # 缩放回原始尺寸
            rescaled = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            augmented_clip.append(rescaled)

        return augmented_clip

    def parse_hdf5_file(self, proprio_path, dataset_name, is_realworld, step_id=None):
        # proprio_path = / root / hetolin / datasets / agibot / proprio_stats / 327 / 648642 / "proprio_state".hdf5
        episode_id = proprio_path.split("/")[-2]
        task_id = proprio_path.split("/")[-3]
        dataset_root_path = Path(proprio_path).parents[3]
        if is_realworld:
            task_json_reformat_path = dataset_root_path / "task_info" / f"task_{task_id}_reformat.json"
            task_json_path = dataset_root_path / "task_info" / f"task_{task_id}.json"
        else:
            task_json_path = dataset_root_path / "meta_info" / task_id / episode_id / "task_info.json"

        if not os.path.exists(task_json_path):
            return False, None

        try:
            qpos, target_qpos = self.load_priprio_stats(proprio_path, is_realworld, self.action_type)
            # only support gripper
            if qpos.shape[-1] > 20:
                return False, None
        except Exception as e:
            return False, None

        num_steps = qpos.shape[0]
        # [Optional] We drop too-short episode
        if num_steps < 150:  # drop < 5s
            return False, None

        # [Optional] We skip the first few still steps
        EPS = 1e-2
        # Get the idx of the first qpos whose delta exceeds the threshold
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")

        # We randomly sample a timestep
        step_id = np.random.randint(first_idx - 1, num_steps) if (step_id is None) else step_id

        # Load the instruction
        task_name = self.get_task_name(task_json_path.as_posix(), is_realworld)
        subgoal_info = self.get_subgoal_info(task_json_reformat_path.as_posix(), episode_id, step_id)

        if random.random() > 0.5:
            instruction = self.get_task_instruction(task_json_path.as_posix(), is_realworld)# "put the laundry detergent into the box"
        else:
            instruction = self.get_task_instruction_finegrain(task_json_reformat_path.as_posix(), task_name, episode_id, step_id)  # "put the laundry detergent into the box"

        # Assemble the meta
        meta = {
            "dataset_name": dataset_name,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction,
            "subgoal_frame": subgoal_info.get("end_frame", -1) if subgoal_info else (num_steps-1),
        }

        # Parse the state and action
        state = qpos[step_id:step_id + 1] #(1,20)
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))
        indices = get_history_indices(step_id, self.IMG_HISORY_SIZE, 10)
        actions_history = target_qpos[indices]  # left pad first action until len=HISTORY_SIZE -1
        actions_future = target_qpos[step_id:step_id + self.CHUNK_SIZE * self.global_downsample_rate:self.global_downsample_rate]
        actions = np.concatenate([actions_history, actions_future], axis=0)
        if actions.shape[0] < self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1:  # future action is not enough
            # Pad the actions using the last action, right pad
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1 - actions.shape[0], 1))
            ], axis=0)

        if self.use_self_norm:
            # normalize (rows: samples, cols: features)
            state = (state - self.qpos_mean) / self.qpos_std
            actions = (actions - self.act_mean) / self.act_std

        state = pad_vector(state, self.STATE_DIM)
        state_indicator = pad_vector(np.ones_like(state_std), self.STATE_DIM)
        state_std = pad_vector(state_std, self.STATE_DIM)
        state_mean = pad_vector(state_mean, self.STATE_DIM)
        state_norm = pad_vector(state_norm, self.STATE_DIM)
        # If action's format is different from state's,
        # you may implement fill_in_action()
        actions = pad_vector(actions, self.STATE_DIM)
        # print(instruction, actions)

        if actions.shape[0] == 0:
            return False, None

        # Parse the images
        if is_realworld:
            video_root_path = dataset_root_path / "observations" / task_id / episode_id / "videos"
            depth_root_path = dataset_root_path / "observations" / task_id / episode_id / "depth"
            intrinsic_head_path = dataset_root_path / "parameters" / task_id / episode_id / "parameters" / "camera" / "head_intrinsic_params.json"
            intrinsic_left_path = dataset_root_path / "parameters" / task_id / episode_id / "parameters" / "camera" / "hand_left_intrinsic_params.json"
            intrinsic_right_path = dataset_root_path / "parameters" / task_id / episode_id / "parameters" / "camera" / "hand_right_intrinsic_params.json"
            extrinsic_head_path = dataset_root_path / "parameters" / task_id / episode_id / "parameters" / "camera" / "head_extrinsic_params_aligned.json"
            extrinsic_left_path = dataset_root_path / "parameters" / task_id / episode_id / "parameters" / "camera" / "hand_left_extrinsic_params_aligned.json"
            extrinsic_right_path = dataset_root_path / "parameters" / task_id / episode_id / "parameters" / "camera" / "hand_right_extrinsic_params_aligned.json"

        else:
            video_root_path = dataset_root_path / "observations" / task_id / episode_id / "video"
            depth_root_path = dataset_root_path / "observations" / task_id / episode_id / "depth"


        if os.path.exists(extrinsic_head_path) and os.path.exists(extrinsic_left_path) and os.path.exists(extrinsic_right_path):
            try:
                intrinsic_head = self.read_intrinsic(intrinsic_head_path, DEFAULT_HEAED_INTRINSICS)  # (3,3)
                intrinsic_left = self.read_intrinsic(intrinsic_left_path, DEFAULT_LEFT_HAND_INTRINSICS)
                intrinsic_right = self.read_intrinsic(intrinsic_right_path, DEFAULT_RIGHT_HAND_INTRINSICS)

                extrinsic_head = self.read_extrinsic(extrinsic_head_path)[step_id:step_id+1]  # cam2base #(1,4,4)
                extrinsic_left = self.read_extrinsic(extrinsic_left_path)[step_id:step_id+1] # cam2base
                extrinsic_right = self.read_extrinsic(extrinsic_right_path)[step_id:step_id+1] # cam2base
            except Exception as e:
                print(e)
                print(f"extrinsic not exist {extrinsic_head_path} {extrinsic_left_path} {extrinsic_right_path}")
                return False, None
        else:
            return False, None

        # gripper相对ee的z偏移0.227m
        T_gripper2ee = np.eye(4)
        T_gripper2ee[2, 3] = 0.227  # z方向平移

        action_step = 10
        action_l = pose_to_transformation_matrix_scipy(actions[::action_step, :7]) #(N,4,4)
        action_r = pose_to_transformation_matrix_scipy(actions[::action_step, 7:14])

        actionl_in_head = np.linalg.inv(extrinsic_head) @ (action_l @ T_gripper2ee) #(N,4,4)
        actionr_in_head = np.linalg.inv(extrinsic_head) @ (action_r @ T_gripper2ee)
        actionl_in_left = np.linalg.inv(extrinsic_left) @ (action_l @ T_gripper2ee)
        actionr_in_left = np.linalg.inv(extrinsic_left) @ (action_r @ T_gripper2ee)
        actionl_in_right = np.linalg.inv(extrinsic_right) @ (action_l @ T_gripper2ee)
        actionr_in_right = np.linalg.inv(extrinsic_right) @ (action_r @ T_gripper2ee)

        actionl_in_head_2d, _ = project_points_to_2d(actionl_in_head[:, :3, 3], intrinsic_head)
        actionr_in_head_2d, _ = project_points_to_2d(actionr_in_head[:, :3, 3], intrinsic_head)
        actionl_in_left_2d, _ = project_points_to_2d(actionl_in_left[:, :3, 3], intrinsic_left)
        actionr_in_left_2d, _ = project_points_to_2d(actionr_in_left[:, :3, 3], intrinsic_left)
        actionl_in_right_2d, _ = project_points_to_2d(actionl_in_right[:, :3, 3], intrinsic_right)
        actionr_in_right_2d, _ = project_points_to_2d(actionr_in_right[:, :3, 3], intrinsic_right)

        # Load history frames
        cam_high = self.parse_video(video_root_path, 'head', step_id, num_steps, indices, is_realworld)  # [] or (IMG_HISORY_SIZE, H, W, 3)
        cam_left_wrist = self.parse_video(video_root_path, 'hand_left', step_id, num_steps, indices, is_realworld)
        cam_right_wrist = self.parse_video(video_root_path, 'hand_right', step_id, num_steps, indices, is_realworld)
        head_depth = self.parse_depth(depth_root_path, step_id, indices, is_realworld)
        # head_depth = np.zeros(cam_high.shape[:3])

        if self.future_video:
            # Load future frames for video generation (like robotwin_dataset)
            future_indices = []
            for i in range(self.num_video_frames):
                future_frame_idx = step_id + (i + 1) * self.video_action_freq_ratio * self.global_downsample_rate
                future_indices.append(min(future_frame_idx, num_steps - 1))  # Clamp to valid range

            cam_high_future = self.parse_video(video_root_path, 'head', step_id, num_steps, future_indices, is_realworld, is_future=True)
            cam_left_wrist_future = self.parse_video(video_root_path, 'hand_left', step_id, num_steps, future_indices, is_realworld, is_future=True)
            cam_right_wrist_future = self.parse_video(video_root_path, 'hand_right', step_id, num_steps, future_indices, is_realworld, is_future=True)

            # Build subgoal combined image at annotated frame if available
            subgoal_image = None
            sg_frame = meta["subgoal_frame"]
            if sg_frame is not None and sg_frame >= 0:
                sg_frame = min(max(0, sg_frame), num_steps - 1)
                sg_frames = []
                for cam_key in ["head", "hand_left", "hand_right"]:
                    frame = self.parse_video(video_root_path, cam_key, sg_frame, num_steps, [], is_realworld)
                    if len(frame) == 0:
                        return False, None
                    # parse_video returns (IMG_HISORY_SIZE, H, W, 3), take the last valid frame
                    sg_frames.append(frame[-1])
                subgoal_image = self.combine_views_single_np(sg_frames[0], sg_frames[1], sg_frames[2])

            # # Build current combined image from first frame in history
            current_image, future_images = self.process_data(cam_high, cam_left_wrist, cam_right_wrist, cam_high_future, cam_left_wrist_future, cam_right_wrist_future)

        # some episode has only one video, or some episode has incomplete video,
        # padding zero for solving these two cases
        if len(cam_high) or len(cam_left_wrist) or len(cam_right_wrist):
            template_shape = []
            cams = [cam_high, cam_left_wrist, cam_right_wrist]
            for cam in cams:
                if len(cam):
                    template_shape = cam.shape

            if not len(cam_high):
                cam_high = np.zeros(template_shape).astype(np.uint8)
            if not len(cam_left_wrist):
                cam_left_wrist = np.zeros(template_shape).astype(np.uint8)
            if not len(cam_right_wrist):
                cam_right_wrist = np.zeros(template_shape).astype(np.uint8)


        if len(cam_high) and len(cam_left_wrist) and len(cam_right_wrist) and len(head_depth):
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist_mask = cam_high_mask.copy()

            depth_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            depth_left_wrist_mask = depth_high_mask.copy()
            depth_right_wrist_mask = depth_high_mask.copy()

            cam_high_det_mask = np.zeros(cam_high.shape[:3])  # (num, h, w)
            cam_left_wrist_det_mask = np.zeros(cam_high.shape[:3])  # (num, h, w)
            cam_right_wrist_det_mask = np.zeros(cam_high.shape[:3])  # (num, h, w)

            # from utils.vis import plot_all_images, plot_all_joints
            # plot_all_images(cam_high, cam_left_wrist, cam_right_wrist)
            # plot_all_joints(actions, state)

            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, h, w, 3)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, h, w, 3)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot

            # head
            multi_images = [cam_high[0].transpose(2,0,1), cam_left_wrist[0].transpose(2,0,1), cam_right_wrist[0].transpose(2,0,1)]
            multi_poses = []
            class_names = []
            multi_2ds = []

            _, H, W, _ = cam_high.shape

            class_names_each_view = []
            multi_poses_each_view = []
            multi_2ds_each_view = []

            # 一开始机械臂要出现在视野中
            if is_group_valid(actionl_in_head_2d[0:1,:], W, H):
                class_names_each_view += [f"left arm {i}" for i in range(len(actionl_in_head))]
                multi_poses_each_view.append(actionl_in_head)
                multi_2ds_each_view.append(actionl_in_head_2d)
            if is_group_valid(actionr_in_head_2d[0:1,:], W, H):
                class_names_each_view += [f"right arm {i}" for i in range(len(actionr_in_head))]
                multi_poses_each_view.append(actionr_in_head)
                multi_2ds_each_view.append(actionr_in_head_2d)
            class_names.append(class_names_each_view) #[(N1+N2,)]
            if len(multi_poses_each_view) > 0:
                multi_poses_each_view = np.concatenate(multi_poses_each_view, axis=0)  # [(N1+N2, 4, 4)]
                multi_poses_each_view = mat44_to_quat_trans(multi_poses_each_view)
                multi_poses.append(torch.from_numpy(multi_poses_each_view))

                multi_2ds_each_view = np.concatenate(multi_2ds_each_view, axis=0)  # [(N1+N2, 2)]
                multi_2ds.append(torch.from_numpy(multi_2ds_each_view))
            else:
                multi_poses.append([])
                multi_2ds.append([])

            class_names_each_view = []
            multi_poses_each_view = []
            multi_2ds_each_view = []
            # if is_group_valid(actionl_in_left_2d, W, H):
            if True:
                class_names_each_view += [f"left arm {i}" for i in range(len(actionl_in_left))]
                multi_poses_each_view.append(actionl_in_left)
                multi_2ds_each_view.append(actionl_in_left_2d)
            # if is_group_valid(actionr_in_left_2d, W, H):
            #     class_names_each_view += [f"right {i}" for i in range(len(actionr_in_left))]
            #     multi_poses_each_view.append(actionr_in_left)
            class_names.append(class_names_each_view)  # [(N1+N2,)]
            if len(multi_poses_each_view) > 0:
                multi_poses_each_view = np.concatenate(multi_poses_each_view, axis=0)  # [(N1+N2, 4, 4)]
                multi_poses_each_view = mat44_to_quat_trans(multi_poses_each_view)
                multi_poses.append(torch.from_numpy(multi_poses_each_view))

                multi_2ds_each_view = np.concatenate(multi_2ds_each_view, axis=0)  # [(N1+N2, 2)]
                multi_2ds.append(torch.from_numpy(multi_2ds_each_view))
            else:
                multi_poses.append([])
                multi_2ds.append([])

            class_names_each_view = []
            multi_poses_each_view = []
            multi_2ds_each_view = []
            # if is_group_valid(actionl_in_right_2d, W, H):
            # if True:
            #     class_names_each_view += [f"left {i}" for i in range(len(actionl_in_right))]
            #     multi_poses_each_view.append(actionl_in_right)
            #     multi_2ds_each_view.append(actionl_in_right_2d)
            # if is_group_valid(actionr_in_right_2d, W, H):
            if True:
                class_names_each_view += [f"right arm {i}" for i in range(len(actionr_in_right))]
                multi_poses_each_view.append(actionr_in_right)
                multi_2ds_each_view.append(actionr_in_right_2d)
            class_names.append(class_names_each_view)  # [(N1+N2,)]
            if len(multi_poses_each_view) > 0:
                multi_poses_each_view = np.concatenate(multi_poses_each_view, axis=0)  # [(N1+N2, 4, 4)]
                multi_poses_each_view = mat44_to_quat_trans(multi_poses_each_view)
                multi_poses.append(torch.from_numpy(multi_poses_each_view))

                multi_2ds_each_view = np.concatenate(multi_2ds_each_view, axis=0)  # [(N1+N2, 2)]
                multi_2ds.append(torch.from_numpy(multi_2ds_each_view))
            else:
                multi_poses.append([])
                multi_2ds.append([])


            # text_label = map_3d_label_to_string_tokenizer(class_names=class_names,
            #                                               multi_images=multi_images,
            #                                               multi_2ds=multi_2ds,
            #                                               multi_poses=multi_poses,
            #                                               multi_sizes=repeat_attribute_to_match(multi_poses, None),
            #                                               tokenizer=self.tokenizer,
            #                                               near2far=False)
            #
            #TODO: ablation: output pose directly
            # class_names = []
            # multi_poses = []
            # if len(action_l) > 0:
            #     class_names_v = []
            #     for i in range(action_l.shape[0]):
            #         class_names_v.append(f"left arm {i}")
            #         multi_poses.append(torch.from_numpy(mat44_to_quat_trans(action_l)))
            #     class_names.append(class_names_v)
            #
            # if len(action_r) > 0:
            #     class_names_v = []
            #     for i in range(action_r.shape[0]):
            #         class_names_v.append(f"right arm {i}")
            #         multi_poses.append(torch.from_numpy(mat44_to_quat_trans(action_r)))
            #     class_names.append(class_names_v)

            # images = {"image0": cam_high[0], "image1": cam_left_wrist[0], "image2": cam_right_wrist[0]}
            # intrinsics = {"image0": intrinsic_head, "image1": intrinsic_left, "image2": intrinsic_right}
            # res = text_to_class_attr_dict_tokenizer(text_label, self.tokenizer)
            # print(text_label)
            # visualize_2d_3d_all(images, res, res, intrinsics, meta["instruction"])

            data_dict = {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high, #(IMG_HISORY_SIZE,h,w,3)
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist, #np.zeros_like(cam_high),
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,

                "depth_high": head_depth, #(IMG_HISORY_SIZE,h,w)
                "depth_high_mask": depth_high_mask,
                "depth_left_wrist": np.zeros_like(head_depth),
                "depth_left_wrist_mask": depth_left_wrist_mask,
                "depth_right_wrist": np.zeros_like(head_depth),
                "depth_right_wrist_mask": depth_right_wrist_mask,
                "cam_high_det_mask": cam_high_det_mask,  # (IMG_HISORY_SIZE,h,w)
                "cam_left_wrist_det_mask": cam_left_wrist_det_mask,
                "cam_right_wrist_det_mask": cam_right_wrist_det_mask,
                # "text_labels": text_label,
                # "depth_priors": [head_depth, np.zeros_like(head_depth), np.zeros_like(head_depth)],
                "intrinsics": [intrinsic_head, intrinsic_left, intrinsic_right],
                "extrinsics": [extrinsic_head, extrinsic_left, extrinsic_right],
                "poses": multi_poses,
                "multi_images": multi_images,
                "multi_2ds": multi_2ds,
                "multi_poses": multi_poses,
                "class_names": class_names,
            }
            if self.future_video:
                data_dict.update({
                    "subgoal_image": subgoal_image,
                    "current_image": current_image,
                    "future_images": future_images,
                })

            return True, data_dict
        else:
            return False, None

    def parse_hdf5_file_state_only(self, proprio_path, is_realworld):
        # proprio_path = / root / hetolin / datasets / agibot / proprio_stats / 327 / 648642 / "proprio_state".hdf5
        episode_id = proprio_path.split("/")[-2]
        task_id = proprio_path.split("/")[-3]
        dataset_root_path = Path(proprio_path).parents[3]
        # task_json_reformat_path = dataset_root_path / "task_info" / f"task_{task_id}_reformat.json"
        # task_json_path = dataset_root_path / "task_info" / f"task_{task_id}.json"

        if is_realworld:
            task_json_reformat_path = dataset_root_path / "task_info" / f"task_{task_id}_reformat.json"
            task_json_path = dataset_root_path / "task_info" / f"task_{task_id}.json"
        else:
            task_json_path = dataset_root_path / "meta_info" / task_id / episode_id / "task_info.json"

        # if task json is not exist: using "" for instruction
        # if not os.path.exists(task_json_path):
        #     return False, None

        states_value, action_value = self.load_priprio_stats(proprio_path, is_realworld, self.action_type)

        qpos = states_value
        num_steps = qpos.shape[0]
        # [Optional] We drop too-short episode
        # if num_steps < 128:
        if num_steps < 150:  # drop < 5s
            return False, None

        # [Optional] We skip the first few still steps
        EPS = 1e-2
        # Get the idx of the first qpos whose delta exceeds the threshold
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            # raise ValueError("Found no qpos that exceeds the threshold.")
            print(f"Found {proprio_path} no qpos that exceeds the threshold.")
            return False, None

        # Parse the state and action
        state = qpos[first_idx - 1:]
        action = action_value[first_idx - 1:]

        # state = pad_vector(state, self.STATE_DIM)
        # action = pad_vector(action, self.STATE_DIM)

        # Return the resulting sample
        return True, {
            "state": state,
            "action": action
        }

    def parse_video(self, root_path, key, frame_id, num_steps, indices, is_realworld, is_future=False):
        # key: head, hand_left, hand_right
        # / root / hetolin / datasets / agibot / observations / 327 / 648642 / videos

        imgs = []
        if is_realworld:
            video_path = os.path.join(root_path, f"{key}_color_h264.mp4")
        else:
            video_path = os.path.join(root_path, f"{key}.mp4")

        # 有些任务的没有三个视角的视频，只存在一个视频
        if not os.path.exists(video_path):
            return []

        verbose = getattr(self.cfg, 'debug', False)

        # For future frames, use indices directly; for history, use indices + [frame_id]
        frame_indices = indices if is_future else (indices + [frame_id])

        for i in frame_indices:
            frame = self.read_images_from_video_opencv(video_path, i, num_steps, verbose=verbose)
            # not read valid frame
            if len(frame) == 0:
                return []

            imgs.append(frame)

        imgs = np.stack(imgs)
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)

        return imgs

    def parse_depth(self, root_path, frame_id, indices, is_realworld, is_future=False):
        depths = []
        # For future frames, use indices directly; for history, use indices + [frame_id]
        frame_indices = indices if is_future else (indices + [frame_id])

        for i in frame_indices:
            if is_realworld:
                frame_id_format = str(i).zfill(6)
                depth_file = (root_path / f"head_depth_{frame_id_format}.png").as_posix()
            else:
                depth_file = (root_path / f"{frame_id}" / "head.png").as_posix()

            #TODO: if depth file not exist, using zero-padding image instead of return []
            if not os.path.exists(depth_file):
                return []

            frame = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED) / 1000.  # (480, 640) unit: m
            frame[frame > 4.] = 0.0
            depths.append(frame)

        depths = np.stack(depths)
        if depths.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            depths = np.concatenate([
                np.tile(depths[:1], (self.IMG_HISORY_SIZE - depths.shape[0], 1, 1, 1)),
                depths
            ], axis=0)

        return depths

    def read_images_from_video_opencv(self, video_path, frame_id, episode_length, verbose=False):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if verbose: print(f"Error: Could not open video.{video_path}")
                return []

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not frame_count == episode_length:
                if verbose: print(f" {video_path} video len is {frame_count} but the episode len is {episode_length}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为 RGB
            else:
                if verbose: print(f"Error: Could not read frame {frame_id} in {video_path}.")
                return []

            cap.release()

        except Exception as e:
            print(f"Exception occurred while processing {video_path}: {e}")

        finally:
            if cap.isOpened():
                cap.release()  # 确保释放资源

        return frame

    def read_json(self, json_file_path):
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            return data
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
        except FileNotFoundError:
            print("The specified file was not found.")

    def read_intrinsic(self, json_file_path, default_intrinsics=None):
        cam_param = self.read_json(json_file_path)
        try:
            fx, fy, cx, cy = cam_param["intrinsic"]["fx"], cam_param["intrinsic"]["fy"], cam_param["intrinsic"]["ppx"], cam_param["intrinsic"]["ppy"]
        except Exception as e:
            # print(e, json_file_path)
            return default_intrinsics

        if fx > 500 or fy > 500 or cx > 400 or cy > 400:
            print(f"{json_file_path} have strange intrinsics, please check")
            return default_intrinsics

        intrinsic = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
        return intrinsic

    def read_extrinsic(self, json_file_path):
        cam_params_list = self.read_json(json_file_path)
        cam_transforms = []

        for cam_param in cam_params_list:
            extrinsic = np.eye(4)
            rotation_matrix = np.array(cam_param["extrinsic"]["rotation_matrix"]).reshape(3,3)
            translation_vector = np.array(cam_param["extrinsic"]["translation_vector"]).reshape(3,1)
            extrinsic[:3, :3] = rotation_matrix
            extrinsic[:3, 3:] = translation_vector

            cam_transforms.append(extrinsic)
        extrinsic = np.stack(cam_transforms, axis=0)

        return extrinsic


    def get_task_instruction(self, task_json_path: str, is_realworld: bool) -> dict:
        """Get task language instruction"""
        if not os.path.exists(task_json_path):
            return ""
        with open(task_json_path, "r") as f:
            task_info = json.load(f)
        if is_realworld:
            task_info = task_info[0]
        task_name = task_info["task_name"]
        task_init_scene = task_info["init_scene_text"]
        # task_instruction = f"{task_name}. {task_init_scene}"
        if task_name.endswith("."):
            task_instruction = f"{task_name}"
        else:
            task_instruction = f"{task_name}."
        # print(f"Get Task Instruction <{task_instruction}>")
        return task_instruction

    def get_task_name(self, task_json_path: str, is_realworld: bool) -> dict:
        """Get task language instruction"""
        with open(task_json_path, "r") as f:
            task_info = json.load(f)
        if is_realworld:
            task_info = task_info[0]
        task_name = task_info["task_name"]

        return task_name

    def get_subgoal_info(self, task_json_path: str, episode_id: str | int, step_id: int) -> dict | None:
        """Return the subgoal annotation for the given frame if available."""

        if not os.path.exists(task_json_path):
            return None

        with open(task_json_path, "r") as f:
            task_info = json.load(f)

        if not task_info or not isinstance(task_info, list):
            return None

        episode_dict = task_info[0] if isinstance(task_info[0], dict) else None
        if episode_dict is None:
            return None

        ep_key = str(episode_id)
        if ep_key not in episode_dict:
            return None

        action_config = episode_dict[ep_key].get("label_info", {}).get("action_config", [])
        for idx, label in enumerate(action_config):
            if step_id >= label.get("start_frame", -1) and step_id <= label.get("end_frame", -1):
                return {
                    #"subgoal_id": idx,
                    #"subgoal_text": label.get("action_text", ""),
                    #"skill": label.get("skill", ""),
                    #"start_frame": label.get("start_frame", -1),
                    "end_frame": label.get("end_frame", -1),
                }

        return None

    def get_task_instruction_finegrain(self, task_json_path, task_name, episode_id, step_id) -> dict:
        """Get task language instruction"""
        with open(task_json_path, "r") as f:
            task_info = json.load(f)

        action_text = ""
        if episode_id in task_info[0].keys():
            episode_info = task_info[0][episode_id]

            task_name = episode_info["task_name"]
            task_init_scene = episode_info["init_scene_text"]
            action_config = episode_info["label_info"]["action_config"]

            for label in action_config:
                if step_id >= label["start_frame"] and step_id <= label["end_frame"]:
                    action_text = label["action_text"]
                    if not action_text.endswith("."):
                        action_text = action_text + "."
                    # print(action_text)
                    break

        # task_instruction = f"{task_name}. {task_init_scene}"
        if task_name.endswith("."):
            task_instruction = f"{task_name} {action_text}"
        else:
            task_instruction = f"{task_name}. {action_text}"
        # print(f"Get Task Instruction: <{task_instruction}>")
        return task_instruction


    def load_priprio_stats(self, proprio_path, is_realworld, type="joint"):
        if is_realworld:
            with (h5py.File(proprio_path) as f):
                if type == "ee":
                    state_ee_trans = np.array(f["/state/end/position"]) #(N,2,3)
                    state_ee_trans_l = np.array(f["/state/end/position"])[:, 0, :]  # (N,3)
                    state_ee_trans_r = np.array(f["/state/end/position"])[:, 1, :]  # (N,3)

                    state_ee_quat = np.array(f["/state/end/orientation"])  # (N,2,4), xyzw format
                    state_ee_quat_l = np.array(f["/state/end/orientation"])[:, 0, :]  # (N,4), xyzw format
                    state_ee_quat_r = np.array(f["/state/end/orientation"])[:, 1, :]  # (N,4), xyzw format

                    action_ee_trans = np.array(f["/action/end/position"])  # (N,2,3)
                    action_ee_trans_l = np.array(f["/action/end/position"])[:, 0, :]  # (N,3)
                    action_ee_trans_r = np.array(f["/action/end/position"])[:, 1, :]  # (N,3)

                    action_ee_quat = np.array(f["/action/end/orientation"])  # (N,2,4), xyzw format
                    action_ee_quat_l = np.array(f["/action/end/orientation"])[:, 0, :]  # (N,4), xyzw format
                    action_ee_quat_r = np.array(f["/action/end/orientation"])[:, 1, :]  # (N,4), xyzw format

                    states_value = np.hstack(
                        [state_ee_trans_l, state_ee_quat_l,
                         state_ee_trans_r, state_ee_quat_r]).astype(np.float32)

                    action_value = np.hstack(
                        [action_ee_trans_l, action_ee_quat_l,
                         action_ee_trans_r, action_ee_quat_r]).astype(np.float32)
                else:
                    state_joint = np.array(f["state/joint/position"])
                    state_joint_l = state_joint[:, :7]
                    state_joint_r = state_joint[:, 7:]

                    state_effector = np.array(f["state/effector/position"])
                    # IMPORTANT: align with the action
                    if state_effector.shape[1] == 2:
                        # denom = state_effector.max(axis=0) - state_effector.min(axis=0)
                        # state_effector = (state_effector - state_effector.min(axis=0)) / np.where(denom == 0, 1, denom)
                        state_effector_l = state_effector[:, 0].reshape(-1, 1)
                        state_effector_r = state_effector[:, 1].reshape(-1, 1)
                    elif state_effector.shape[1] == 12:
                        state_effector_l = state_effector[:, :6]
                        state_effector_r = state_effector[:, 6:]
                    state_head = np.array(f["state/head/position"])
                    state_waist = np.array(f["state/waist/position"])

                    action_joint = np.array(f["action/joint/position"])
                    action_joint_l = action_joint[:, :7]
                    action_joint_r = action_joint[:, 7:]
                    action_effector = np.array(f["action/effector/position"])
                    if action_effector.shape[1] == 2:
                        action_effector_l = action_effector[:, 0].reshape(-1, 1)
                        action_effector_r = action_effector[:, 1].reshape(-1, 1)
                    elif action_effector.shape[1] == 12:
                        action_effector_l = action_effector[:, :6]
                        action_effector_r = action_effector[:, 6:]
                    action_head = np.array(f["action/head/position"])
                    action_waist = np.array(f["action/waist/position"])
                    action_velocity = np.array(f["action/robot/velocity"])

                    # states_value = np.hstack(
                    #     [state_joint, state_effector, state_head, state_waist]).astype(np.float32)
                    states_value = np.hstack(
                        [state_joint_l, state_effector_l,
                         state_joint_r, state_effector_r,
                         state_head, state_waist]).astype(np.float32)
                    assert (
                            action_joint.shape[0] == action_effector.shape[0]
                    ), f"shape of action_joint:{action_joint.shape};shape of action_effector:{action_effector.shape}"
                    # action_value = np.hstack(
                    #     [action_joint, action_effector, action_head, action_waist]).astype(np.float32)
                    action_value = np.hstack(
                        [action_joint_l, action_effector_l,
                         action_joint_r, action_effector_r,
                         action_head, action_waist]).astype(np.float32)
        else:
            with h5py.File(proprio_path) as f:
                state_joint = np.array(f["state/joint/position"])
                joint_names = f["state/joint"].attrs['name'].tolist()

            head_joint_names = [
                "joint_head_yaw",
                "joint_head_pitch",
            ]
            body_joint_names = [
                "joint_lift_body",
                "joint_body_pitch",
            ]
            arm_joint_names = [
                "Joint1_l",
                "Joint1_r",
                "Joint2_l",
                "Joint2_r",
                "Joint3_l",
                "Joint3_r",
                "Joint4_l",
                "Joint4_r",
                "Joint5_l",
                "Joint5_r",
                "Joint6_l",
                "Joint6_r",
                "Joint7_l",
                "Joint7_r",
            ]
            effector_joint_names = [
                "left_Left_1_Joint",
                # "left_Right_1_Joint"
                "right_Left_1_Joint",
                # "right_Right_1_Joint",
            ]

            # Get indices for arm and effector joints from the first frame
            head_joint_indices = [joint_names.index(name) for name in head_joint_names]
            body_joint_indices = [joint_names.index(name) for name in body_joint_names]
            arm_joint_indices = [joint_names.index(name) for name in arm_joint_names]
            arm_joint_l_indices = arm_joint_indices[0::2]
            arm_joint_r_indices = arm_joint_indices[1::2]
            arm_joint_indices = arm_joint_l_indices + arm_joint_r_indices
            effector_joint_indices = [joint_names.index(name) for name in effector_joint_names]

            # Extract joint positions for all frames
            state_head = state_joint[:, head_joint_indices]
            state_body = state_joint[:, body_joint_indices]
            state_arm = state_joint[:, arm_joint_indices]
            state_effector = state_joint[:, effector_joint_indices]

            # Get action from state
            action_head = state_head[1:]# - state_head[:-1]
            action_body = state_body[1:]# - state_body[:-1]
            action_arm = state_arm[1:]# - state_arm[:-1]
            action_effector = state_effector[1:]# - state_effector[:-1]

            # repeat the last frame of the action
            action_head = np.concatenate([action_head, action_head[-1:]])
            action_body = np.concatenate([action_body, action_body[-1:]])
            action_arm = np.concatenate([action_arm, action_arm[-1:]])
            action_effector = np.concatenate([action_effector, action_effector[-1:]])

            states_value = np.hstack(
                [state_arm, state_effector, state_head, state_body]
            ).astype(np.float32)
            assert (
                    action_arm.shape[0] == action_effector.shape[0]
            ), f"shape of action_arm:{action_arm.shape};shape of action_effector:{action_effector.shape}"
            action_value = np.hstack(
                [action_arm, action_effector, action_head, action_body]
            ).astype(np.float32)

        return states_value, action_value


if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)