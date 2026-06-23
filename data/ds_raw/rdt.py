import os
import fnmatch
import json

import h5py
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils.vis import get_history_indices

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

        dataset_json_path = Path(self.cfg.dev_dir) / 'config' / "dataset_meta" / "hdf5_rdt_list.json"
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
                dataset_name = np.random.choice(self.DATASET_NAMES, p=self.dataset_weights)
                # file_path = np.random.choice(self.file_paths_dict[dataset_name])
                file_path = np.random.choice(self.file_paths_dict[dataset_name], p=self.file_paths_dict[f"{dataset_name}_episode_weights"])

                # dataset_name = np.random.choice(self.DATASET_NAMES, p=self.sample_weights)
                # file_path = np.random.choice(self.file_paths_dict[dataset_name])
                # file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path, dataset_name) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample

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
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            target_qpos = f['action'][:]

            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
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

            # Load the instruction
            dir_path = os.path.dirname(file_path)
            with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
                instruction_dict = json.load(f_instr)
            # We have 1/3 prob to use original instruction,
            # 1/3 to use simplified instruction,
            # and 1/3 to use expanded instruction.
            # instruction_type = np.random.choice([
            #     'instruction', 'simplified_instruction', 'expanded_instruction'])
            instruction_type = np.random.choice([
                'instruction', 'simplified_instruction'])
            instruction = instruction_dict[instruction_type]
            if isinstance(instruction, list):
                instruction = np.random.choice(instruction)

            # Assemble the meta
            meta = {
                "dataset_name": dataset_name,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            # from utils.vis import plot_all_joints
            # target_qpos_full = f['action'][:] / np.array(
            #     [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]]
            # )
            # plot_all_joints(target_qpos_full, qpos)

            # Parse the state and action
            state = qpos[step_id:step_id + 1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))
            # actions = target_qpos[step_id:step_id + self.CHUNK_SIZE]
            indices = get_history_indices(step_id, self.IMG_HISORY_SIZE, 10)
            # left_id = max(step_id - self.IMG_HISORY_SIZE + 1, 0)
            # actions = target_qpos[left_id:step_id + self.CHUNK_SIZE]
            actions_history = target_qpos[indices]  # left pad first action until len=HISTORY_SIZE -1
            actions_future = target_qpos[step_id:step_id + self.CHUNK_SIZE]
            actions = np.concatenate([actions_history, actions_future], axis=0)
            if actions.shape[0] < self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1:  # future action is not enough
                # Pad the actions using the last action, right pad
                actions = np.concatenate([
                    actions,
                    # np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1))
                    np.tile(actions[-1:], (self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1 - actions.shape[0], 1))
                ], axis=0)

            # Rescale gripper to [0, 1]
            state = state / np.array(
                [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]]
            )
            actions = actions / np.array(
                [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]]
            )

            # e.g., CHUNK_SIZE = 3, HISTORY_SIZE = 3
            # len(obs)=3, len(actions)=5
            # case1: step_id = 1
            # | - | - | - | - |
            #     | - | - | - | - | - | - | - | - | - | - |
            #     0   1   2   3   4   5   6   7   8   9   10
            # return:
            #       obs:     [0,1]
            #       actions: [0,1,2,3]
            # pad:
            #       obs:     [0,0,1]
            #       actions: [0,0,1,2,3]
            # summary:
            # Pad the obs using the first obs, left pad (same)
            # Pad the actions using the first action, left pad (different)

            # case2: step_id = 9
            #                                 | - | - | - | - |
            #     | - | - | - | - | - | - | - | - | - | - |
            #     0   1   2   3   4   5   6   7   8   9   10
            # return:
            #       obs:     [7,8,9]
            #       actions: [7,8,9,10]
            # pad:
            #       obs:     [7,8,9]
            #       actions: [7,8,9,10,10]
            # summary:
            # Pad the obs using the first obs, left pad (same)
            # Pad the actions using the last action, right pad (different)

            state = pad_vector(state, self.STATE_DIM)
            state_indicator = pad_vector(np.ones_like(state_std), self.STATE_DIM)
            state_std = pad_vector(state_std, self.STATE_DIM)
            state_mean = pad_vector(state_mean, self.STATE_DIM)
            state_norm = pad_vector(state_norm, self.STATE_DIM)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = pad_vector(actions, self.STATE_DIM)

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
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs

            # `cam_high` is the external camera image
            cam_high = parse_img('cam_high')
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = parse_img('cam_left_wrist')
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img('cam_right_wrist')
            cam_right_wrist_mask = cam_high_mask.copy()

            # from utils.vis import plot_all_images, plot_all_joints
            # plot_all_images(cam_high, cam_left_wrist, cam_right_wrist)
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

                "depth_high": np.zeros(cam_high.shape[:3]), #(IMG_HISORY_SIZE,h,w)
                "depth_high_mask": np.zeros_like(cam_high_mask),
                "depth_left_wrist": np.zeros(cam_high.shape[:3]),
                "depth_left_wrist_mask": np.zeros_like(cam_high_mask),
                "depth_right_wrist": np.zeros(cam_high.shape[:3]),
                "depth_right_wrist_mask": np.zeros_like(cam_high_mask),
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
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
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

            # Rescale gripper to [0, 1]
            qpos = qpos / np.array(
                [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]]
            )
            target_qpos = f['action'][:] / np.array(
                [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]]
            )

            # Parse the state and action
            state = qpos[first_idx - 1:]
            action = target_qpos[first_idx - 1:]

            state = pad_vector(state, self.STATE_DIM)
            action = pad_vector(action, self.STATE_DIM)

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