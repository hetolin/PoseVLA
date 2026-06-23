import math
import random
import traceback
from importlib import import_module
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from pathlib import Path
from torch.utils.data import ConcatDataset
from hydra.utils import instantiate
import cv2

from utils.image_corrupt import image_corrupt


class VLAConsumerDataset(Dataset):
    """Vision-language-action dataset for supervised HDF5 training."""

    def __init__(self, config: dict):
        super(VLAConsumerDataset, self).__init__()
        self.config = config

        self.num_cameras = config.dataset.num_cameras
        self.img_history_size = config.dataset.img_history_size
        self.cond_mask_prob = config.dataset.cond_mask_prob
        self.cam_ext_mask_prob = config.dataset.cam_ext_mask_prob
        self.use_hdf5 = config.dataset.use_hdf5

        self.hdf5_dataset = None

        self.DATASET_NAMES = []
        if self.use_hdf5:
            self.base_dataset_list = []

            print(f"Loading dataset {config.dataset.dataset_list}")
            for baseset_dict in config.dataset.dataset_list:
                print(f"Loading dataset {baseset_dict}")
                baseset = instantiate(baseset_dict, cfg=config, _recursive_=False)
                self.base_dataset_list.append(baseset)

                self.DATASET_NAMES.extend(list(baseset_dict.sample_weights.keys()))

            self.hdf5_dataset = ConcatDataset(self.base_dataset_list)
            self.dataset_name2id = {name: i for i, name in enumerate(self.DATASET_NAMES)}
            self.dataset_id2name = {i: name for i, name in enumerate(self.DATASET_NAMES)}
            print(self.dataset_name2id)
            print(f"Training Model on `{list(self.DATASET_NAMES)}` from `{config.dataset.hdf5_dir}`")

        else:
            raise NotImplementedError

        self.image_size = config.dataset.image_size
        self.auto_adjust_image_brightness = config.dataset.auto_adjust_image_brightness
        self.image_aug = config.dataset.image_aug
        self.head_camera_randomcrop_aug = config.dataset.head_camera_randomcrop_aug

        self.last_content = None
        self.last_meta = None

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    def __len__(self) -> int:
        if self.use_hdf5:
            return len(self.hdf5_dataset)

    def __getitem__(self, index):
        while True:
            data_dict = None
            try:
                if self.use_hdf5:
                    res = self.hdf5_dataset[index]
                    content = res['meta']
                    states = res['state']
                    actions = res['actions']
                    state_elem_mask = res['state_indicator']
                    image_metas = [
                        res['cam_high'], res['cam_high_mask'],
                        res['cam_left_wrist'], res['cam_left_wrist_mask'],
                        res['cam_right_wrist'], res['cam_right_wrist_mask'],
                    ]

                    depth_metas = [
                        res['depth_high'], res['depth_high_mask'],
                        res['depth_left_wrist'], res['depth_left_wrist_mask'],
                        res['depth_right_wrist'], res['depth_right_wrist_mask'],
                    ]

                    det_metas = [
                        res['cam_high_det_mask'],
                        res['cam_left_wrist_det_mask'],
                        res['cam_right_wrist_det_mask'],
                    ]

                data_dict = {}
                data_dict['dataset_name'] = content['dataset_name']
                data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]
                data_dict["states"] = states
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = state_elem_mask \
                    if random.random() > self.cond_mask_prob else np.zeros_like(state_elem_mask)

                image_metas = list(self.pairwise(image_metas))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob
                rearranged_images = []
                rearranged_dets = []
                for i in range(self.num_cameras):
                    for j in range(self.img_history_size):
                        images, image_mask = image_metas[i]
                        image, valid = images[j], image_mask[j]
                        dets = det_metas[i]
                        det = dets[j]
                        if valid and (math.prod(image.shape) > 0) and (random.random() > mask_probs[j]):
                            rearranged_images.append((image, True))
                            rearranged_dets.append((det))
                        else:
                            background_image = np.zeros_like(image)
                            rearranged_images.append((background_image.copy(), False))
                            background_det = np.zeros_like(det)
                            rearranged_dets.append((background_det.copy()))

                preprocessed_images = []
                preprocessed_dets = []
                for i, ((image, valid), det) in enumerate(zip(rearranged_images, rearranged_dets)):
                    image = Image.fromarray(image)
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image)
                        det = Image.fromarray(det)
                        det = transforms.Resize(self.image_size)(det)
                        det = np.array(det)

                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice([
                            "corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)

                        if self.head_camera_randomcrop_aug:
                            det = det[..., None]
                            cat_img = np.concatenate([image, det], axis=2)
                            cat_img = torch.from_numpy(cat_img).permute(2, 0, 1).float()

                            width, height = image.size
                            transform = transforms.Compose([
                                transforms.RandomCrop((int(height * 0.85), int(width * 0.85))),
                                transforms.Resize((height, width)),
                                transforms.RandomRotation(degrees=(-5, 5)),
                            ])

                            cat_img = transform(cat_img)
                            image = cat_img[:3].permute(1, 2, 0).numpy()
                            det = cat_img[3].numpy()

                    preprocessed_images.append(np.array(image))
                    preprocessed_dets.append(np.array(det)[..., None])

                depth_metas = list(self.pairwise(depth_metas))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob
                rearranged_depths = []
                for i in range(self.num_cameras):
                    for j in range(self.img_history_size):
                        depths, depth_mask = depth_metas[i]
                        depth, valid = depths[j], depth_mask[j]
                        if valid and (math.prod(depth.shape) > 0) and (random.random() > mask_probs[j]):
                            rearranged_depths.append((depth, True))
                        else:
                            rearranged_depths.append((np.zeros_like(depth).copy(), False))

                preprocessed_depths = []
                for i, (depth, valid) in enumerate(rearranged_depths):
                    if self.image_size is not None:
                        h, w = depth.shape[:2]
                        scale = self.image_size / min(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                    preprocessed_depths.append(np.array(depth)[..., None])

                data_dict["images"] = [torch.from_numpy(img).permute(2, 0, 1) for img in preprocessed_images]
                data_dict["images"] = [resize_with_pad(img.unsqueeze(0),
                                                       self.image_size,
                                                       self.image_size,
                                                       pad_value=0)[0] for img in data_dict["images"]]
                data_dict["instructions"] = content["instruction"] if random.random() > self.cond_mask_prob else ""
                data_dict["depths"] = [torch.from_numpy(depth).permute(2, 0, 1) for depth in preprocessed_depths]
                data_dict["depths"] = [resize_with_pad(depth.unsqueeze(0),
                                                       self.image_size,
                                                       self.image_size,
                                                       pad_value=0, mode="nearest")[0] for depth in data_dict["depths"]]

                data_dict["dets"] = [torch.from_numpy(det).permute(2, 0, 1) for det in preprocessed_dets]
                data_dict["dets"] = [resize_with_pad(det.unsqueeze(0),
                                                       self.image_size,
                                                       self.image_size,
                                                       pad_value=0, mode="nearest")[0] for det in data_dict["dets"]]

                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v).float()

                return data_dict
            except BaseException as e:
                if data_dict is not None:
                    print(f"Error catched when processing sample from {data_dict.get('dataset_name')}:", e)
                else:
                    print(f"Error catched when processing sample:", e)
                traceback.print_exc()
                index = (index + 1) % len(self)

def resize_with_pad(img, width, height, pad_value=-1, mode="bilinear"):
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    interpolate_params = {
        'size': (resized_height, resized_width),
        'mode': mode,
    }

    if mode != "nearest":
        interpolate_params['align_corners'] = False

    resized_img = F.interpolate(img, **interpolate_params)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    pw = pad_width // 2
    ph = pad_height // 2
    padded_img = F.pad(resized_img, (pw, pad_width - pw, ph, pad_height - ph), value=pad_value)
    return padded_img


class DataCollatorForPI0ConsumerDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, config) -> None:
        self.config = config
        self.IMG_HISORY_SIZE = config.dataset.img_history_size

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        base_keys = ["states", "actions", "images", "instructions", "depths", "dets"]
        batch = {k: [] for k in base_keys}

        for instance in instances:
            for key in ['states', 'actions']:
                batch[key].append(torch.as_tensor(instance[key]))

            batch["images"].append(torch.stack(instance["images"], dim=0))
            batch["depths"].append(torch.stack(instance["depths"], dim=0))
            batch["dets"].append(torch.stack(instance["dets"], dim=0))
            batch["instructions"].append(instance["instructions"])

        stacked = {}
        for key, value in batch.items():
            if isinstance(value[0], torch.Tensor):
                stacked[key] = torch.stack(value, dim=0)
            else:
                stacked[key] = value

        h = self.IMG_HISORY_SIZE
        all_imgs = stacked["images"].float()
        all_depths = stacked["depths"]
        all_dets = stacked["dets"]

        return {
            "observation.images.top_head": all_imgs[:, 0:h] / 255.0,
            "observation.images.hand_left": all_imgs[:, h:2 * h] / 255.0,
            "observation.images.hand_right": all_imgs[:, 2 * h:3 * h] / 255.0,

            "observation.depths.top_head": all_depths[:, 0:h],
            "observation.depths.hand_left": all_depths[:, h:2 * h],
            "observation.depths.hand_right": all_depths[:, 2 * h:3 * h],

            "observation.images.top_head.det": all_dets[:, 0:h],
            "observation.images.hand_left.det": all_dets[:, h:2 * h],
            "observation.images.hand_right.det": all_dets[:, 2 * h:3 * h],

            "observation.state": stacked["states"][:, 0, :],
            "action": stacked["actions"],
            "task": stacked["instructions"],
        }

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}


def _load_lerobot_dataset_api():
    import lerobot
    from lerobot.configs.default import DatasetConfig

    version = getattr(lerobot, "__version__", "未知")
    if version == "0.1.0":
        dataset_module = import_module("lerobot.common.datasets.lerobot_dataset")
        factory_module = import_module("lerobot.common.datasets.factory")
        transforms_module = import_module("lerobot.common.datasets.transforms")
    elif version == "0.4.2":
        dataset_module = import_module("lerobot.datasets.lerobot_dataset")
        factory_module = import_module("lerobot.datasets.factory")
        transforms_module = import_module("lerobot.datasets.transforms")
    else:
        raise NotImplementedError(f"Unsupported lerobot version: {version}")

    return (
        DatasetConfig,
        transforms_module.ImageTransforms,
        transforms_module.ImageTransformsConfig,
        dataset_module.LeRobotDataset,
        dataset_module.LeRobotDatasetMetadata,
        dataset_module.MultiLeRobotDataset,
        factory_module.resolve_delta_timestamps,
    )


def make_dataset(cfg_pi0, cfg):
    """Create a LeRobot dataset from config."""
    (
        DatasetConfig,
        ImageTransforms,
        ImageTransformsConfig,
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
        resolve_delta_timestamps,
    ) = _load_lerobot_dataset_api()

    transform_config = ImageTransformsConfig(enable=cfg.dataset.image_transforms, max_num_transforms=3)
    dataset_config = DatasetConfig(repo_id=cfg.dataset.repo_id,
                                   episodes=cfg.dataset.episodes,
                                   image_transforms=transform_config,
                                   local_files_only=cfg.dataset.local_files_only,
                                   use_imagenet_stats=cfg.dataset.use_imagenet_stats,
                                   video_backend=cfg.dataset.video_backend)

    image_transforms = (
        ImageTransforms(dataset_config.image_transforms) if dataset_config.image_transforms.enable else None
    )

    lerobot_dir = Path(cfg.dataset.lerobot_dir)
    print(f"Training Model on {cfg.dataset.repo_id} from {lerobot_dir}")
    if isinstance(dataset_config.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(dataset_config.repo_id,
                                         root=lerobot_dir / dataset_config.repo_id,
                                         local_files_only=dataset_config.local_files_only)
        delta_timestamps = resolve_delta_timestamps(cfg_pi0, ds_meta)
        dataset = LeRobotDataset(
            dataset_config.repo_id,
            root=lerobot_dir / dataset_config.repo_id,
            episodes=dataset_config.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=dataset_config.video_backend,
            local_files_only=dataset_config.local_files_only,
        )
    else:
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id[0],
                                         root=lerobot_dir / cfg.dataset.repo_id[0],
                                         local_files_only=cfg.dataset.local_files_only)
        delta_timestamps = resolve_delta_timestamps(cfg_pi0, ds_meta)
        dataset = MultiLeRobotDataset(
            dataset_config.repo_id,
            root=lerobot_dir,
            episodes=dataset_config.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=dataset_config.video_backend,
            local_files_only=dataset_config.local_files_only,
        )
        import logging
        from pprint import pformat

        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def _eep_dim_labels(action_type, dim):
    if action_type != "eep":
        return [f"dim_{i}" for i in range(dim)]

    arm_labels = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
    labels = [f"left_{name}" for name in arm_labels] + [f"right_{name}" for name in arm_labels]
    if dim > len(labels):
        labels.extend(f"pad_{i}" for i in range(len(labels), dim))
    return labels[:dim]


def _plot_action_state(action, state, labels, save_path):
    import matplotlib.pyplot as plt

    dim = action.shape[1]
    ncols = min(4, dim)
    nrows = math.ceil(dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False, sharex=True)

    for i in range(dim):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        ax.plot(action[:, i], label="action", linewidth=2, marker="o", markersize=2)
        ax.axhline(state[i], label="state", linestyle="--", linewidth=1.5)
        ax.set_title(f"{labels[i]} (std={action[:, i].std():.3f})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend()

    for i in range(dim, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Saved action/state trajectories to {save_path}")


def debug_visualize(cfg):
    from utils.vis import plot_all_images

    dataset = VLAConsumerDataset(cfg)
    data_collator = DataCollatorForPI0ConsumerDataset(cfg)
    sample = dataset[0]
    batch = data_collator([sample])

    print(batch.keys())
    print(batch["observation.images.top_head"].shape)
    print(batch["observation.images.hand_left"].shape)
    print(batch["observation.images.hand_right"].shape)
    print(batch["observation.state"].shape)
    print(batch["action"].shape)
    print(batch["task"])

    plot_all_images(
        batch["observation.images.top_head"][0],
        batch["observation.images.hand_left"][0],
        batch["observation.images.hand_right"][0],
        save_path="dataset_wan_images.png",
    )

    state_mask = torch.as_tensor(sample["state_elem_mask"]).bool()
    valid_dims = torch.where(state_mask)[0]
    print(f"valid state/action dims: {len(valid_dims)} / {state_mask.numel()}")

    action = batch["action"][0][:, valid_dims].detach().cpu().numpy()
    state = batch["observation.state"][0][valid_dims].detach().cpu().numpy()
    labels = _eep_dim_labels(cfg.dataset.action_type, action.shape[1])
    _plot_action_state(
        action,
        state,
        labels,
        save_path="dataset_wan_joints.png",
    )


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="./config", config_name="base_postrain")
    def main(cfg):
        debug_visualize(cfg)

    main()

