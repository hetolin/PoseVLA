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

        dataset_json_path = Path(self.cfg.dev_dir) / 'config' / "dataset_meta" / "hdf5_droid_list.json"
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

                self.file_paths_dict[dataset_name] = self.find_hdf5_files(f"{HDF5_DIR}/{dataset_name}")
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

    def find_hdf5_files(self, dataset_root):
        dataset_root = Path(dataset_root)
        lab_paths = [f for f in dataset_root.iterdir() if f.is_dir()]

        hdf5_files = []
        for lab_path in tqdm(lab_paths):
            states = ["success"]
            # states = ["success", "failure"]
            for state in states:
                day_paths = [f for f in (lab_path / state).iterdir() if f.is_dir()]
                for day_path in day_paths:
                    time_paths = [f for f in day_path.iterdir() if f.is_dir()]
                    for time_path in time_paths:
                        json_files = list(time_path.glob('*.json'))
                        if not json_files:
                            continue
                        if (time_path / "trajectory.h5").exists() and (time_path / "trajectory.h5").is_file():
                            hdf5_files.append((time_path / "trajectory.h5").as_posix())
            # break
        print(f"Total tar file: {len(hdf5_files)}")

        return hdf5_files


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
        try:
            json_file = list(Path(file_path).parent.glob('*.json'))[0]
            with open(json_file, 'r') as fin:
                info = json.load(fin)
        except FileNotFoundError:
            print(f"Annotation file not found: {json_file}")
            return False, None

        uuid = info['uuid']

        anno_lang_file = f"{self.cfg.dataset.hdf5_dir}/{dataset_name}/aggregated-annotations-030724.json"
        with open(anno_lang_file, 'r') as fin:
            anno_lang = json.load(fin)
        try:
            langs = anno_lang[uuid]
        except KeyError:
            langs = {"language_instruction1": "",
                     "language_instruction2": "",
                     "language_instruction3": ""}

        with h5py.File(file_path, 'r') as f:
            qpos_arm = f['observation']['robot_state']['joint_positions'][:]  # (N,7)
            qpos_gripper = f['observation']['robot_state']['gripper_position'][:].reshape(-1, 1)  # (N,1)
            qpos = np.concatenate([qpos_arm, qpos_gripper], axis=-1)  # (N,8)

            target_arm_qpos = f['action']["joint_position"][:]  # (N,7)
            target_gripper_qpos = f['action']["gripper_position"][:].reshape(-1, 1)  # (N,1)
            target_qpos = np.concatenate([target_arm_qpos, target_gripper_qpos], axis=-1)

            # from utils.vis import plot_all_joints
            # plot_all_joints(target_qpos, qpos)

            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            # if num_steps < 128:
            if num_steps < 75:  # drop < 5s
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

            instruction = np.random.choice(list(langs.values()))
            # Assemble the meta
            meta = {
                "dataset_name": dataset_name,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            # Parse the state and action
            state = qpos[step_id:step_id + 1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))
            # actions = target_qpos[step_id:step_id + self.CHUNK_SIZE]
            indices = get_history_indices(step_id, self.IMG_HISORY_SIZE, 30, False)

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

            state = pad_vector(state, self.STATE_DIM)
            state_indicator = pad_vector(np.ones_like(state_std), self.STATE_DIM)
            state_std = pad_vector(state_std, self.STATE_DIM)
            state_mean = pad_vector(state_mean, self.STATE_DIM)
            state_norm = pad_vector(state_norm, self.STATE_DIM)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = pad_vector(actions, self.STATE_DIM)

        # Parse the images
        video_root_path = Path(file_path).parent / "recordings/MP4"

        # TODO: support IMG_HISORY_SIZE
        # CAM_LIST = ["wrist_cam_serial", "ext1_cam_serial", "ext2_cam_serial"]
        cam_high = self.parse_video(video_root_path, info["ext1_cam_serial"], step_id, num_steps, indices)  # [] or (IMG_HISORY_SIZE, H, W, 3)
        cam_left_wrist = self.parse_video(video_root_path, info["ext2_cam_serial"], step_id, num_steps, indices)
        cam_right_wrist = self.parse_video(video_root_path, info["wrist_cam_serial"], step_id, num_steps, indices)

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
            # print(video_root_path, np.max(cam_high), np.min(cam_high), np.max(cam_left_wrist), np.min(cam_left_wrist), np.max(cam_right_wrist), np.min(cam_right_wrist))

        if len(cam_high) and len(cam_left_wrist) and len(cam_right_wrist):
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist_mask = cam_high_mask.copy()
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

        else:
            return False, None

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
            # print(f["observation"].keys())
            # print(f["observation"]["robot_state"].keys())
            # print(f["action"].keys())
            # print(f["action"]["robot_state"].keys())

            # print(f["action"]["joint_position"][:])
            # print(f["action"]["robot_state"]["joint_positions"][:])
            # print(f["observation"]["robot_state"]["joint_positions"][:])
            qpos_arm = f['observation']['robot_state']['joint_positions'][:] # (N,7)
            qpos_gripper = f['observation']['robot_state']['gripper_position'][:].reshape(-1, 1) # (N,1)
            qpos = np.concatenate([qpos_arm, qpos_gripper], axis=-1) # (N,8)

            target_arm_qpos = f['action']["joint_position"][:]# (N,7)
            target_gripper_qpos = f['action']["gripper_position"][:].reshape(-1, 1) # (N,1)
            target_qpos = np.concatenate([target_arm_qpos, target_gripper_qpos], axis=-1)

            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            # if num_steps < 128:
            if num_steps < 75:  # drop < 5s
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

            state = pad_vector(state, self.STATE_DIM)
            action = pad_vector(action, self.STATE_DIM)

            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }

    def parse_video(self, root_path, key, frame_id, num_steps, indices):
        # key: head, hand_left, hand_right
        # / root / hetolin / datasets / agibot / observations / 327 / 648642 / videos

        imgs = []
        video_path = os.path.join(root_path, f"{key}.mp4")

        # 有些任务的没有三个视角的视频，只存在一个视频
        if not os.path.exists(video_path):
            return []
        # for i in range(max(frame_id - self.IMG_HISORY_SIZE + 1, 0), frame_id + 1):
        for i in (indices + [frame_id]):
            frame = self.read_images_from_video_opencv(video_path, i, num_steps, verbose=self.cfg.debug)
            # not read valid frame
            if len(frame) == 0:
                return []

            # crop 1280*720 to 640*480
            crop_h, crop_w = 480, 640
            h, w, _ = frame.shape

            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            frame = frame[start_h:start_h + crop_h, start_w:start_w + crop_w]


            imgs.append(frame)

        imgs = np.stack(imgs)
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)

        return imgs

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


if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)