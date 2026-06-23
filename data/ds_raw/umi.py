import os
import fnmatch
import json

import pickle
import h5py
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from embodied_pi0_action.utils.vis import get_history_indices, batch_overlay_gaussian_mask
import embodied_pi0_action.data.normalize as normalize
from PIL import Image, ImageDraw
from embodied_pi0_action.utils.transform_utils import convert_PosQuat2PosRotationMatrix_batch, dual_arm_poses_to_relative, get_relative_pose,get_relative_chunk_pose_dual_arm, add_rotation_noise


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

def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def make_gaussian_mask_on_resized(point, orig_w, orig_h, long_side=224, patch_size=14, sigma=1.0):
    """
    归一化点坐标, 原图尺寸, resize到长边, patch_size -> mask大小为 [resize_h//patch_size, resize_w//patch_size]
    """
    scale = long_side / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    num_patch_x = new_w // patch_size
    num_patch_y = new_h // patch_size

    # 归一化点对应在resize后图片里的像素坐标
    x_scaled = point[0] * orig_w * scale
    y_scaled = point[1] * orig_h * scale
    xg_float = x_scaled / patch_size
    yg_float = y_scaled / patch_size

    xx, yy = np.meshgrid(np.arange(num_patch_x), np.arange(num_patch_y))
    dist_sq = (xx - xg_float) ** 2 + (yy - yg_float) ** 2
    mask = np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)
    return mask  # shape [num_patch_y, num_patch_x]


def generate_masks(mask_keys, pred_label, orig_w, orig_h, long_side=224, patch_size=14):
    """
    返回：
    mask_dict = {
        'image0': mask,  # 每个 mask 是 [orig_h//patch_size, orig_w//patch_size]
    }
    """
    scale = long_side / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    num_patch_x = new_w // patch_size
    num_patch_y = new_h // patch_size

    mask_dict = {}
    # mask_keys = ["image0", "image1", "image2"]
    if isinstance(mask_keys, str):
        mask_keys = [mask_keys]

    for image_name in mask_keys:
        mask = np.zeros((num_patch_y, num_patch_x), dtype=np.float32)
        gt_pairs = pred_label[image_name] if image_name in pred_label.keys() else []
        for _, point in gt_pairs:
            # x_scaled = point[0] * orig_w * scale
            # y_scaled = point[1] * orig_h * scale
            # xg = int(np.clip(round(x_scaled / patch_size), 0, num_patch_x - 1))
            # yg = int(np.clip(round(y_scaled / patch_size), 0, num_patch_y - 1))
            # mask[yg, xg] = 1

            submask = make_gaussian_mask_on_resized(point, orig_w, orig_h, long_side, patch_size, sigma=1)
            # print(submask.shape, mask.shape)
            mask = np.maximum(mask, submask)
            # mask += submask  # 如果想累加所有点能量，用这一行
        # 归一化
        # mask = mask / mask.max() if mask.max() > 0 else mask
        # mask_dict[image_name] = mask
        mask_dict[image_name] = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return mask_dict

class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """

    def __init__(self, cfg, sample_weights=dict) -> None:
        self.cfg = cfg
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIR = cfg.dataset.hdf5_dir#"data/datasets/agilex/rdt_data/"

        # self.DATASET_NAMES = list(cfg.dataset.sample_weights.keys())
        self.DATASET_NAMES = list(sample_weights.keys())
        # self.DATASET_NAMES = [name for name in sample_weights.keys()
        #                       if 'xtrainer' in name.lower()]
        SAMPLE_WEIGHTS = sample_weights

        # Load the config
        self.CHUNK_SIZE = cfg.dataset.action_chunk_size
        self.IMG_HISORY_SIZE = cfg.dataset.img_history_size
        self.STATE_DIM = cfg.dataset.state_dim

        # todo herb - check if use self-calculated norm
        self.use_self_norm = False
        self.action_type = cfg.dataset.action_type
        print(f"****** action_type: {self.action_type} ******")
        self.root_noise_std = [0, 0, 0]
        if hasattr(cfg.dataset, 'root_noise_std'):
            self.root_noise_std = cfg.dataset.root_noise_std
        print(f"****** root_noise_std: {self.root_noise_std} ******")

        if hasattr(cfg.dataset, 'mean_std_path'):
            self.use_self_norm = True
            mean_std_path = cfg.dataset.mean_std_path

            # Check if the file exists
            if os.path.exists(mean_std_path):
                with open(mean_std_path, 'rb') as f:
                    norm_info = pickle.load(f)
                    self.qpos_mean = np.array(norm_info['qpos_mean'], dtype=np.float32)
                    self.qpos_std = np.array(norm_info['qpos_std'], dtype=np.float32)
                    self.act_mean = np.array(norm_info['action_mean'], dtype=np.float32)
                    self.act_std = np.array(norm_info['action_std'], dtype=np.float32)
                    print(f"===Load mean_std from {mean_std_path}")
                    print(self.qpos_mean)
                    print(self.qpos_std)
            else:
                # Raise a ValueError if the file does not exist
                raise ValueError(f"File does not exist: {mean_std_path}")

        # mean_std_dir = "mydata/xtrainer_mix_stats_pnp5"
        # # # Check if the file exists
        # norm_stats = normalize.load(mean_std_dir)
        # self.qpos_q02 = norm_stats["state"].q02[None, :]
        # self.qpos_q98 = norm_stats["state"].q98[None, :]
        # self.act_mean = norm_stats["action"].mean
        # self.act_std = norm_stats["action"].std
        # print("Load norm_stats.json")

        # 1. for weight sampling fetching when index is None in get_item()
        self.file_paths_dict = {}
        for dataset_name in self.DATASET_NAMES:
            self.file_paths_dict[dataset_name] = []

        # for dataset_name in self.DATASET_NAMES:
        #     for root, _, files in os.walk(f"{HDF5_DIR}/{dataset_name}"):
        #         for filename in fnmatch.filter(files, '*.hdf5'):
        #             file_path = os.path.join(root, filename)
        #             # self.file_paths.append(file_path)
        #             self.file_paths_dict[dataset_name].append(file_path)

        dataset_json_path = Path(self.cfg.dev_dir) / 'config' / "dataset_meta" / "hdf5_xtrainer_list_umi.json"
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
                # self.file_paths_dict[dataset_name] = self.find_hdf5_files(f"{HDF5_DIR}/{dataset_name}", is_realworld)
                # self.file_paths_dict[dataset_name] = self.find_match(f"{HDF5_DIR}/{dataset_name}", is_realworld)
                for root, _, files in os.walk(f"{HDF5_DIR}/{dataset_name}"):
                    for filename in fnmatch.filter(files, '*.hdf5'):
                        file_path = os.path.join(root, filename)
                        # self.file_paths.append(file_path)
                        self.file_paths_dict[dataset_name].append(file_path)

                print(f"find hdf5 {dataset_name} Done!")

            # self.sample_weights_dataset = {}
            self.num_total_episode = 0
            self.num_valid_timestep = 0
            self.num_valid_episode = 0
            for dataset_name in self.DATASET_NAMES:
                episode_lens = []
                # for each dataset, get each episode's len
                for file_path in tqdm(self.file_paths_dict[dataset_name]):
                    # file_path = file_path.replace("/apdcephfs_nj7/share_305204761/herbzhang/data", "mydata")
                    valid, res = self.parse_hdf5_file_state_only(file_path)
                    _len = res['state'].shape[0] if valid else 0
                    episode_lens.append(_len)
                    self.num_valid_timestep += _len
                    self.num_total_episode += 1
                    if _len > 0:
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

        # Load the config
        # self.CHUNK_SIZE = cfg.dataset.action_chunk_size
        # self.IMG_HISORY_SIZE = cfg.dataset.img_history_size
        # self.STATE_DIM = cfg.dataset.state_dim

        dataset_weights = []
        for dataset_name in self.DATASET_NAMES:
            dataset_weights.append(SAMPLE_WEIGHTS[dataset_name])

        dataset_weights = np.array(dataset_weights)
        self.dataset_weights = dataset_weights / np.sum(dataset_weights)
        for i, dataset_name in enumerate(self.DATASET_NAMES):
            print(f"+ Dataset {dataset_name} weight: {self.dataset_weights[i]}")

        # sample_weights = []
        # for dataset_name in self.DATASET_NAMES:
        #     sample_weights.append(SAMPLE_WEIGHTS[dataset_name])
        #
        # sample_weights = np.array(sample_weights)
        # self.sample_weights = sample_weights / np.sum(sample_weights)
        #
        # # 2. for index fetching when index is available in get_item()
        # self.file_paths = []
        # for dataset_name in self.DATASET_NAMES:
        #     for root, _, files in os.walk(f"{HDF5_DIR}/{dataset_name}"):
        #         for filename in fnmatch.filter(files, '*.hdf5'):
        #             file_path = os.path.join(root, filename)
        #             self.file_paths.append(file_path)
        #
        # # Get each episode's len
        # episode_lens = []
        # for file_path in self.file_paths:
        #     valid, res = self.parse_hdf5_file_state_only(file_path)
        #     _len = res['state'].shape[0] if valid else 0
        #     episode_lens.append(_len)
        # self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

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
                # dataset_name = np.random.choice(self.DATASET_NAMES, p=self.dataset_weights)
                # # file_path = np.random.choice(self.file_paths_dict[dataset_name])
                # file_path = np.random.choice(self.file_paths_dict[dataset_name], p=self.file_paths_dict[f"{dataset_name}_episode_weights"])

                # dataset_name = np.random.choice(self.DATASET_NAMES, p=self.sample_weights)
                # file_path = np.random.choice(self.file_paths_dict[dataset_name])
                # file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)

                dataset_name = np.random.choice(self.DATASET_NAMES, p=self.dataset_weights)
                full_paths = self.file_paths_dict[dataset_name]
                full_weights = self.file_paths_dict[f"{dataset_name}_episode_weights"]
                half_idx = len(full_paths) // 2
                half_paths = full_paths[:half_idx]
                half_weights = full_weights[:half_idx]
                half_weights = half_weights / np.sum(half_weights)
                file_path = np.random.choice(half_paths, p=half_weights)

            else:
                # file_path = self.file_paths[index]
                for dataset_name in self.DATASET_NAMES:
                    dataset_len = len(self.file_paths_dict[dataset_name])
                    if index < dataset_len:
                        local_idx = index
                        file_path = self.file_paths_dict[dataset_name][local_idx]
                    # index超出了本数据集范围，则减去本数据集的长度，继续找下一个数据集
                    else:
                        index -= dataset_len

            valid, sample = self.parse_hdf5_file(file_path, dataset_name) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = (index + 1) % len(self)

    def __getitem__(self, index):
        return self.get_item()

    def parse_hdf5_file(self, file_path, dataset_name):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """

        def compute_completeness(instruction_list):
            completion = np.zeros(len(instruction_list))

            # 标记指令变化的边界
            changes = np.where(instruction_list[:-1] != instruction_list[1:])[0] + 1
            changes = np.concatenate([[0], changes, [len(instruction_list)]])

            # 计算每段的完成度
            for i in range(len(changes) - 1):
                start, end = changes[i], changes[i + 1]
                completion[start:end] = np.linspace(0, 1, end - start)  # 线性增长

            completeness = np.round(completion, 4)

            # print(completeness)  # 输出: [0.0 0.5 1.   0.0 0.5 1.  ]
            return completeness

        with h5py.File(file_path, 'r') as f:
            # qpos = f['observations']['qpos'][:]
            # target_qpos = f['action'][:]

            if self.action_type == "joint":
                qpos = f['observations']['qpos'][:]
                target_qpos = f['action'][:]
                # qpos = f['observations']['qpos'][:][:, self.repeat_mask]
                # target_qpos = f['action_joint'][:][:, self.repeat_mask]
            elif self.action_type == "relative_chunk_ee":
                qpos = f['observations']['qpos'][:]
                target_qpos = f['action'][:]
            elif self.action_type == "relative_chunk_ee_RT":
                qpos = f['observations/eepos'][:].astype(np.float32)
                target_qpos = qpos.copy()
                # qpos = add_rotation_noise(qpos, noise_sigma=0.055, same_noise_for_both_arms=True)
                qpos = convert_PosQuat2PosRotationMatrix_batch(qpos)
                target_qpos[:, [7, 15]] = f['action'][:, [7, 15]].astype(np.float32)  # set gripper action value

            # from utils.vis import plot_all_joints
            # plot_all_joints(target_qpos, qpos)

            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            # if num_steps < 128:
            if num_steps < 50: # drop < 5s
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
            step_id = np.random.randint(first_idx - 1, num_steps)

            # instruction = "put the laundry detergent into the box"
            # instruction = "put the red bottle into the green box"
            # instruction = "put the circle shell into the base"
            # instruction = "put the object into the container"
            # instruction = "open the drawer and pick the pen into the drawer"

            # instruction = f["instruction"][:][0].decode('utf-8')

            raw_inst = f["instruction"][0]  # 取第一帧
            if isinstance(raw_inst, bytes):
                instruction = raw_inst.decode('utf-8')
            else:
                instruction = str(raw_inst)

            # if "stack_3_bowls" in file_path:
            #     instruction = "stack the green bowls"
            # if "spoon_fork_bowls" in file_path:
            #     instruction = "pick the spoon and fork into the bowl"
            # if "hang_mug" in file_path:
            #     instruction = "hang the mug"
            # if "open_drawer" in file_path:
            #     instruction = "open the drawer and pick the pen into the drawer"
            # if "fold_towel" in file_path:
            #     instruction = "fold the towel"

            # if "bowl" in file_path:
            #     instruction = "stack the green bowls"
            # if "cup" in file_path:
            #     instruction = "hang the mug"
            # if "drawer" in file_path:
            #     instruction = "open the drawer and pick the pen into the drawer"
            # if "towel" in file_path:
            #     instruction = "fold the towel"

            # Assemble the meta
            meta = {
                "dataset_name": dataset_name,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            # completeness = compute_completeness(f['instruction'][:])
            # print([x.decode('utf-8') for x in f['instruction'][:]])

            # Parse the state and action
            state = qpos[step_id:step_id + 1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))
            # actions = target_qpos[step_id:step_id + self.CHUNK_SIZE]
            indices = get_history_indices(step_id, self.IMG_HISORY_SIZE, 10)
            # left_id = max(step_id - self.IMG_HISORY_SIZE + 1, 0)
            # actions = target_qpos[left_id:step_id + self.CHUNK_SIZE]
            actions_history = target_qpos[indices] # left pad first action until len=HISTORY_SIZE -1
            actions_future = target_qpos[step_id:step_id + self.CHUNK_SIZE]
            actions = np.concatenate([actions_history, actions_future], axis=0)
            if actions.shape[0] < self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1: # future action is not enough
                # Pad the actions using the last action, right pad
                actions = np.concatenate([
                    actions,
                    # np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1))
                    np.tile(actions[-1:], (self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1 - actions.shape[0], 1))
                ], axis=0)
            actions = actions[-self.CHUNK_SIZE:, :]

            # Define the standard deviation for each dimension
            noise_std = np.array(self.root_noise_std, dtype=np.float32)
            root_noise = np.random.normal(0, noise_std).astype(np.float32)

            # Apply noise to the specified slices of the state and actions arrays
            if self.action_type == "relative_chunk_ee" or self.action_type == "relative_chunk_ee_rot_noise":
                state[:, 0:3] += root_noise
                state[:, 10:13] += root_noise

                assert len(self.act_mean.shape) == len(self.act_std.shape) == 2
                assert self.act_mean.shape[0] == self.act_std.shape[0] == self.CHUNK_SIZE
                assert self.act_mean.shape[1] == self.act_std.shape[1]

                # w/ relative quat
                actions = get_relative_chunk_pose_dual_arm(actions)
            elif self.action_type == "relative_chunk_ee_RT":
                state[:, 0:3] += root_noise
                state[:, 10:13] += root_noise

                actions = dual_arm_poses_to_relative(actions)

            # herb: here means use on-line normalization, note that here use global mean/std
            if self.use_self_norm:
                # normalize (rows: samples, cols: features)
                state = (state - self.qpos_mean) / self.qpos_std
                actions = (actions - self.act_mean) / self.act_std

            # # mask = make_bool_mask(6, -1, 6, -1)
            # mask = make_bool_mask(14)
            # mask = np.asarray(mask)
            # # actions -= np.where(mask, state, 0)
            # # state = (state - self.qpos_mean) / self.qpos_std
            # # actions = (actions - self.act_mean) / self.act_std
            # # state = np.where(mask, state, (state - self.qpos_mean) / self.qpos_std)
            # # actions = np.where(mask, actions, (actions - self.act_mean) / self.act_std)
            #
            # state_normalized = 2.0 * (state - self.qpos_q02) / (self.qpos_q98 - self.qpos_q02 + 1e-6) - 1.0
            # state_normalized = np.clip(state_normalized, -1, 1)
            # state = np.where(mask, state_normalized, state)
            #
            # # actions_normalized = 2.0 * (actions - self.act_q02) / (self.act_q98 - self.act_q02 + 1e-6) - 1.0
            # # actions_normalized = np.clip(actions_normalized, -1, 1)
            # # actions = np.where(mask, actions_normalized, actions)

            state = pad_vector(state, self.STATE_DIM)
            state_indicator = pad_vector(np.ones_like(state_std), self.STATE_DIM)
            state_std = pad_vector(state_std, self.STATE_DIM)
            state_mean = pad_vector(state_mean, self.STATE_DIM)
            state_norm = pad_vector(state_norm, self.STATE_DIM)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = pad_vector(actions, self.STATE_DIM)

            # # insert the completeness prediction into the each predicted action
            # target_completeness = completeness[step_id:step_id + self.CHUNK_SIZE]
            # if target_completeness.shape[0] < self.CHUNK_SIZE:
            #     # Pad the actions using the last action
            #     target_completeness = np.concatenate([
            #         target_completeness,
            #         np.tile(target_completeness[-1:], (self.CHUNK_SIZE - target_completeness.shape[0],))
            #     ], axis=0)
            # actions[:, target_qpos.shape[1]] = target_completeness

            # Parse the images
            def parse_img(key):
                imgs = []
                # for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                for i in (indices + [step_id]):
                    img = f['observations']['images'][key][i]
                    try:
                        img_bgr = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                    except:
                        print(f"cannot read image from {file_path}, please check your dataset.")
                    imgs.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # print(f"Warning: the number of images is less than {self.IMG_HISORY_SIZE}.")
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs

            # `cam_high` is the external camera image
            cam_high = parse_img('cam_high')

            # cam_high = np.zeros_like(cam_high)
            # Solution1: extract the cam_high
            # pred_labels = f['pred_labels'][:]  # 是 numpy 字符串数组
            # pred_labels_list = [json.loads(s.decode('utf-8')) for s in pred_labels]
            # pred_label = pred_labels_list[step_id]
            #
            # color_map = {
            #     'left bottle': (255, 0, 0),  # 红
            #     'left container': (0, 255, 0),  # 绿
            #     'right container': (0, 0, 255),  # 蓝
            #     'right bottle': (255, 255, 0),  # 黄
            # }
            # num, h, w, c = cam_high.shape
            # point_radius = 10
            #
            # img = Image.new('RGB', (w, h), (0, 0, 0))  # 注意PIL顺序是(w, h)
            # draw = ImageDraw.Draw(img)
            #
            # out_img = np.zeros_like(cam_high)  # [num, h, w, 3]，初始全黑
            # crop_size = 100
            # if "image0" in pred_label:
            #     gt_pairs = pred_label['image0']
            #     for idx_point, (cls, point) in enumerate(gt_pairs):
            #         # 防止坐标超界
            #         x = int(np.clip(point[0] * w, point_radius, w - point_radius - 1))
            #         y = int(np.clip(point[1] * h, point_radius, h - point_radius - 1))
            #         color = color_map.get(cls, (255, 255, 255))
            #         draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=color)
            #
            #         # === 以下是对cam_high的crop ===
            #         # crop边界clip
            #         half = crop_size // 2
            #         left = max(0, x - half)
            #         right = min(w, x + half + 1)
            #         top = max(0, y - half)
            #         bottom = min(h, y + half + 1)
            #
            #         # 只保留正方形区域，其他为黑色
            #         out_img[:, top:bottom, left:right, :] = cam_high[:, top:bottom, left:right, :]  # 合并多点区域
            #
            # # cam_high = np.array(img)  # (h, w, 3), uint8
            # # cam_high = np.repeat(cam_high[None, ...], num, axis=0)  # shape (num, h, w, 3)
            # cam_high = out_img

            # SolutionB: extract the mask
            def parse_det(key):
                pred_labels = f['pred_labels'][:]  # 是 numpy 字符串数组
                pred_labels_list = [json.loads(s.decode('utf-8')) for s in pred_labels]
                pred_label = pred_labels_list[step_id]

                num, h, w, c = cam_high.shape
                mask_dict = generate_masks(key, pred_label, w, h, patch_size=14)
                det_mask = np.repeat(mask_dict[key][None, ...], num, axis=0) # (num, h, w)
                return det_mask

            cam_high_det_mask = parse_det("image0") if "pred_labels" in f else np.zeros(cam_high.shape[:3]) # (num, h, w)
            cam_left_wrist_det_mask = parse_det("image1") if "pred_labels" in f else np.zeros(cam_high.shape[:3])
            cam_right_wrist_det_mask = parse_det("image2") if "pred_labels" in f else np.zeros(cam_high.shape[:3])

            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = parse_img('cam_left_wrist')
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img('cam_right_wrist')
            cam_right_wrist_mask = cam_high_mask.copy()

            def parse_depth(key, norm=False):
                imgs = []
                for i in (indices + [step_id]):
                    img = f['observations']['depth_images'][key][i]
                    try:
                        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_ANYDEPTH)

                        if key == "cam_high":
                            scale = 1000.
                        else:
                            scale = 10000.
                        img = img.astype(np.float32) / scale

                        img = np.where(img < 0, 0, img)  # 小于0的设为0
                        img = np.nan_to_num(img, nan=0, posinf=0, neginf=0)  # nan、+inf、-inf都设为0
                        img[img > 5] = 0.0

                        if norm:
                            # 只统计有效像素
                            valid_pixels = img[img > 0]
                            if valid_pixels.size > 0:
                                q02 = np.quantile(valid_pixels, 0.02)
                                q98 = np.quantile(valid_pixels, 0.98)
                                img = np.clip(img, q02, q98)
                                img = (img - q02) / (q98 - q02)
                                img = np.clip(img, 0, 1)
                            else:
                                img[:] = 0.0
                    except Exception as e:
                        print(f"cannot read image from {file_path}, please check your dataset. Error: {e}")
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1)),
                        imgs
                    ], axis=0)
                return imgs

            #(IMG_HISORY_SIZE,h,w)
            depth_high = parse_depth("cam_high", norm=False) if "depth_images" in f["observations"] else np.zeros(cam_high.shape[:3])
            depth_left_wrist = parse_depth('cam_left_wrist', norm=False) if "depth_images" in f["observations"] else np.zeros(cam_high.shape[:3])
            depth_right_wrist = parse_depth('cam_right_wrist', norm=False) if "depth_images" in f["observations"] else np.zeros(cam_high.shape[:3])
            depth_high_mask = cam_high_mask.copy() if "depth_images" in f["observations"] else np.zeros_like(cam_high_mask)
            depth_left_wrist_mask = depth_high_mask.copy()
            depth_right_wrist_mask = depth_high_mask.copy()

            # from utils.vis import plot_all_images, plot_all_joints, plot_all_images_with_depth, plot_all_class_points
            # plot_all_images(cam_high, cam_left_wrist, cam_right_wrist)
            # plot_all_images_with_depth(cam_high, cam_left_wrist, cam_right_wrist, depth_high, depth_left_wrist, depth_right_wrist)
            # plot_all_images_with_depth(batch_overlay_gaussian_mask(cam_high, cam_high_det_mask),
            #                            batch_overlay_gaussian_mask(cam_left_wrist, cam_left_wrist_det_mask),
            #                            batch_overlay_gaussian_mask(cam_right_wrist, cam_right_wrist_det_mask),
            #                            cam_high_det_mask,
            #                            cam_left_wrist_det_mask,
            #                            cam_right_wrist_det_mask)
            # plot_all_class_points([cam_high, cam_left_wrist, cam_right_wrist], pred_label, pred_label, "None")
            # plot_all_joints(actions, state)

            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot

            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high, #(IMG_HISORY_SIZE,h,w,3)
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,

                "depth_high": depth_high, #(IMG_HISORY_SIZE,h,w)
                "depth_high_mask": depth_high_mask,
                "depth_left_wrist": depth_left_wrist,
                "depth_left_wrist_mask": depth_left_wrist_mask,
                "depth_right_wrist": depth_right_wrist,
                "depth_right_wrist_mask": depth_right_wrist_mask,
                "cam_high_det_mask": cam_high_det_mask, #(IMG_HISORY_SIZE,h,w)
                "cam_left_wrist_det_mask": cam_left_wrist_det_mask,
                "cam_right_wrist_det_mask": cam_right_wrist_det_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            # qpos = f['observations']['qpos'][:]
            # target_qpos = f['action'][:]
            try:
                if self.action_type == "joint":
                    qpos = f['observations']['qpos'][:]
                    target_qpos = f['action'][:]
                    # qpos = f['observations']['qpos'][:][:, self.repeat_mask]
                    # target_qpos = f['action_joint'][:][:, self.repeat_mask]
                elif self.action_type == "relative_chunk_ee":
                    qpos = f['observations']['qpos'][:]
                    target_qpos = f['action'][:]
                elif self.action_type == "relative_chunk_ee_RT":
                    qpos = f['observations/eepos'][:].astype(np.float32)
                    target_qpos = qpos.copy()
                    # qpos = add_rotation_noise(qpos, noise_sigma=0.055, same_noise_for_both_arms=True)
                    qpos = convert_PosQuat2PosRotationMatrix_batch(qpos)
                    target_qpos[:, [7, 15]] = f['action'][:, [7, 15]].astype(np.float32)  # set gripper action value
            except Exception as e:
                print(file_path)
                print(f.keys())
                return False, None

            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 50: # drop < 5s
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
                print(f"Found {file_path} no qpos that exceeds the threshold.")
                return False, None

            # Parse the state and action
            state = qpos[first_idx - 1:]
            action = target_qpos[first_idx - 1:]

            # # todo: parse_state_only is a bit different with parse_full_state above - so can not be used for training
            # # herb: here means use on-line normalization, note that here use global mean/std
            # if self.use_self_norm:
            #     # normalize (rows: samples, cols: features)
            #     state = (state - self.qpos_mean) / self.qpos_std
            #     action = (action - self.act_mean) / self.act_std

            # state = pad_vector(state, self.STATE_DIM)
            # action = pad_vector(action, self.STATE_DIM)

            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }


if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)