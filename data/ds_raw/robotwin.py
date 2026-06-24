import os
import fnmatch
import json
import pickle
import h5py
import cv2
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from utils.vis import get_history_indices


def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = np.zeros(shape)
    new_vector[..., :current_dim] = vector
    return new_vector


class HDF5VLADataset:
    """Sample Robotwin episodes stored in HDF5."""

    def __init__(self, cfg, sample_weights=dict, ratio=1.0) -> None:
        self.cfg = cfg
        self.ratio = ratio
        self.ACTION_TYPE = cfg.dataset.action_type

        hdf5_dir = cfg.dataset.hdf5_dir

        self.DATASET_NAMES = list(sample_weights.keys())

        self.CHUNK_SIZE = cfg.dataset.action_chunk_size
        self.IMG_HISORY_SIZE = cfg.dataset.img_history_size
        self.STATE_DIM = cfg.dataset.state_dim
        self.global_downsample_rate = 3

        self.use_self_norm = False
        if hasattr(cfg.dataset, 'mean_std_path'):
            self.use_self_norm = True
            mean_std_path = cfg.dataset.mean_std_path

            if os.path.exists(mean_std_path):
                state_key = f"{self.ACTION_TYPE}_mean"
                state_std_key = f"{self.ACTION_TYPE}_std"
                with open(mean_std_path, 'rb') as f:
                    norm_info = pickle.load(f)
                    self.qpos_mean = np.array(norm_info[state_key], dtype=np.float32)
                    self.qpos_std = np.array(norm_info[state_std_key], dtype=np.float32)
                    self.act_mean = np.array(norm_info['action_mean'], dtype=np.float32)
                    self.act_std = np.array(norm_info['action_std'], dtype=np.float32)
                    print(f"===Load mean_std from {mean_std_path}")
                    print(self.qpos_mean)
                    print(self.qpos_std)
            else:
                raise ValueError(f"File does not exist: {mean_std_path}")

        self.file_paths_dict = {dataset_name: [] for dataset_name in self.DATASET_NAMES}

        dataset_json_path = Path(self.cfg.dev_dir) / 'config' / "dataset_meta" / "hdf5_robotwin_list.json"
        dataset_json_path.parent.mkdir(parents=True, exist_ok=True)
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
                for root, _, files in os.walk(f"{hdf5_dir}/{dataset_name}", followlinks=True):
                    for filename in fnmatch.filter(files, '*.hdf5'):
                        file_path = os.path.join(root, filename)
                        self.file_paths_dict[dataset_name].append(file_path)
                print(f"find hdf5 {dataset_name} Done!")

            self.num_total_episode = 0
            self.num_valid_timestep = 0
            self.num_valid_episode = 0
            for dataset_name in self.DATASET_NAMES:
                episode_lens = []
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

        if self.ratio < 1.0:
            print(f"[Dataset] Applying ratio={self.ratio:.2f} — using {self.ratio*100:.0f}% of episodes per task")
            for dataset_name in self.DATASET_NAMES:
                paths = self.file_paths_dict[dataset_name]
                n_keep = max(1, int(len(paths) * self.ratio))
                indices = np.linspace(0, len(paths) - 1, n_keep, dtype=int)
                kept_paths = [paths[i] for i in indices]
                self.file_paths_dict[dataset_name] = kept_paths

                ep_lens_all = self.file_paths_dict.get(f"{dataset_name}_episode_timesteps", [])
                if ep_lens_all:
                    kept_lens = [ep_lens_all[i] for i in indices]
                    total = sum(kept_lens) or 1
                    self.file_paths_dict[f"{dataset_name}_episode_weights"] = [l / total for l in kept_lens]
                    self.file_paths_dict[f"{dataset_name}_episode_timesteps"] = kept_lens

                print(f"  {dataset_name}: {len(paths)} -> {n_keep} episodes")

        dataset_weights = []
        for dataset_name in self.DATASET_NAMES:
            dataset_weights.append(sample_weights[dataset_name])

        dataset_weights = np.array(dataset_weights)
        self.dataset_weights = dataset_weights / np.sum(dataset_weights)
        for i, dataset_name in enumerate(self.DATASET_NAMES):
            print(f"+ Dataset {dataset_name} weight: {self.dataset_weights[i]}")

        self.file_paths = [
            (dataset_name, file_path)
            for dataset_name in self.DATASET_NAMES
            for file_path in self.file_paths_dict[dataset_name]
        ]

    def __len__(self):
        return len(self.file_paths)

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
            else:
                dataset_name, file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path, dataset_name) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            if index is not None:
                index = (index + 1) % len(self.file_paths)

    def __getitem__(self, index):
        return self.get_item(index)

    def load_state_action(self, file_path, action_type):
        with h5py.File(file_path, 'r') as f:
            if action_type == "qpos":
                qpos = f['observations']['state_qpos'][:]
                target_qpos = f['actions_qpos'][:]
                return qpos, target_qpos
            elif action_type == "eep":
                eep = f['observations']['state_eep'][:]
                target_eep = f['actions_eep'][:]
                return eep, target_eep
            else:
                raise NotImplementedError(f"Action type {action_type} not implemented.")

    def parse_hdf5_file(self, file_path, dataset_name):
        episode_id = file_path.split("/")[-2]
        task_id = file_path.split("/")[-3]
        dataset_root_path = Path(file_path).parents[2]
        video_root_path = dataset_root_path / task_id / episode_id

        qpos, target_qpos = self.load_state_action(file_path, self.ACTION_TYPE)
        num_steps = qpos.shape[0]
        if num_steps < 50:
            return False, None

        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.") 

        step_id = np.random.randint(first_idx - 1, num_steps)

        dir_path = os.path.dirname(file_path)
        instructions_path = os.path.join(dir_path, "instructions.json")
        with open(instructions_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions_names = instruction_dict['seen']
        instr_idx = random.randrange(len(instructions_names))
        instruction = instructions_names[instr_idx]

        meta = {
            "dataset_name": dataset_name,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction
        }

        state = qpos[step_id:step_id + 1]
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))
        indices = get_history_indices(step_id, self.IMG_HISORY_SIZE, 10)
        actions_history = target_qpos[indices]
        actions_future = target_qpos[step_id:step_id + self.CHUNK_SIZE * self.global_downsample_rate:self.global_downsample_rate]
        actions = np.concatenate([actions_history, actions_future], axis=0)
        if actions.shape[0] < self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1:
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1 - actions.shape[0], 1))
            ], axis=0)
            
        if self.use_self_norm:
            state = (state - self.qpos_mean) / self.qpos_std
            actions = (actions - self.act_mean) / self.act_std
                
        state = pad_vector(state, self.STATE_DIM)
        state_indicator = pad_vector(np.ones_like(state_std), self.STATE_DIM)
        state_std = pad_vector(state_std, self.STATE_DIM)
        state_mean = pad_vector(state_mean, self.STATE_DIM)
        state_norm = pad_vector(state_norm, self.STATE_DIM)
        actions = pad_vector(actions, self.STATE_DIM)

        cam_high = self.parse_video(video_root_path, 'cam_high', step_id, num_steps, indices)
        cam_left_wrist = self.parse_video(video_root_path, 'cam_left_wrist', step_id, num_steps, indices)
        cam_right_wrist = self.parse_video(video_root_path, 'cam_right_wrist', step_id, num_steps, indices)
    
        valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
        cam_high_mask = np.array(
            [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
        )
        cam_left_wrist_mask = cam_high_mask.copy()
        cam_right_wrist_mask = cam_high_mask.copy()

        result = {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_high_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_left_wrist_mask,
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_right_wrist_mask,

            "depth_high": np.zeros(cam_high.shape[:3]),
            "depth_high_mask": np.zeros_like(cam_high_mask),
            "depth_left_wrist": np.zeros(cam_high.shape[:3]),
            "depth_left_wrist_mask": np.zeros_like(cam_high_mask),
            "depth_right_wrist": np.zeros(cam_high.shape[:3]),
            "depth_right_wrist_mask": np.zeros_like(cam_high_mask),

            "cam_high_det_mask": np.zeros(cam_high.shape[:3]),
            "cam_left_wrist_det_mask": np.zeros(cam_high.shape[:3]),
            "cam_right_wrist_det_mask": np.zeros(cam_high.shape[:3]),
        }

        return True, result

    def parse_hdf5_file_state_only(self, file_path):
        qpos, target_qpos = self.load_state_action(file_path, self.ACTION_TYPE)
        num_steps = qpos.shape[0]
        if num_steps < 50:
            return False, None

        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            print(f"Found {file_path} no qpos that exceeds the threshold.")
            return False, None

        state = qpos[first_idx - 1:]
        action = target_qpos[first_idx - 1:]

        state = pad_vector(state, self.STATE_DIM)
        action = pad_vector(action, self.STATE_DIM)

        return True, {
            "state": state,
            "action": action
        }
            

    def parse_video(self, root_path, key, frame_id, num_steps, indices):
        imgs = []
        video_path = os.path.join(root_path, f"{key}.mp4")

        if not os.path.exists(video_path):
            return []

        verbose = getattr(self.cfg, 'debug', False)
        frame_indices = indices + [frame_id]

        for i in frame_indices:
            frame = self.read_images_from_video_opencv(video_path, i, num_steps, verbose=verbose)
            if len(frame) == 0:
                return []

            imgs.append(frame)

        imgs = np.stack(imgs)
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)

        return imgs

    def read_images_from_video_opencv(self, video_path, frame_id, episode_length, verbose=False):
        cap = None
        frame = []
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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                if verbose: print(f"Error: Could not read frame {frame_id} in {video_path}.")
                return []

            cap.release()

        except Exception as e:
            print(f"Exception occurred while processing {video_path}: {e}")

        finally:
            if cap is not None and cap.isOpened():
                cap.release()

        return frame