"""HDF5-format VLA dataset and lerobot-format action dataset factory.

Originally lived in ``dataset.py`` at the project root. The collator that used
to live alongside the dataset has been moved to
``collators.CollatorForActionConsumerDataset``.
"""

# ===== Standard library =====
import math
import os
import random
import traceback
from pathlib import Path

# ===== Third-party libraries =====
import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

# ===== Local modules =====
from embodied_pi0_action.utils.image_corrupt import image_corrupt


def process_video_images(current_image, future_images, target_height, target_width):
    """
    current_image: np.array (H, W, 3)
    future_images: np.array (T, H, W, 3)
    """
    # 1. Concatenate then convert to (B, C, H, W) Tensor in [0, 1].
    combined = np.concatenate([current_image[None, ...], future_images], axis=0)
    combined_tensor = torch.from_numpy(combined).permute(0, 3, 1, 2).float() / 255.0

    # 2. Aspect-preserving pad-resize.
    combined_tensor = resize_with_pad(combined_tensor, target_width, target_height, pad_value=0)

    # 3. Apply identical augmentation parameters across all frames.
    transform = transforms.Compose([
        # transforms.RandomRotation(degrees=(-5, 5)),
        # transforms.RandomCrop((int(target_height * 0.9), int(target_width * 0.9))),
        transforms.Resize((target_height, target_width), antialias=True),
    ])
    augmented_tensor = transform(combined_tensor)

    current_out = augmented_tensor[0]      # (3, H, W)
    future_out = augmented_tensor[1:]      # (T, 3, H, W)
    return current_out, future_out


class VLAConsumerDataset(Dataset):
    """A vision-language-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.num_cameras = config.dataset.num_cameras
        self.img_history_size = config.dataset.img_history_size
        self.cond_mask_prob = config.dataset.cond_mask_prob
        self.cam_ext_mask_prob = config.dataset.cam_ext_mask_prob
        self.use_hdf5 = config.dataset.use_hdf5
        self.video_height = config.dataset.video_height
        self.video_width = config.dataset.video_width

        self.hdf5_dataset = None

        self.DATASET_NAMES = []
        if self.use_hdf5:
            self.base_dataset_list = []

            print(f"Loading dataset {config.dataset.dataset_list}")
            for baseset_dict in config.dataset.dataset_list:
                print(f"Loading dataset {baseset_dict}")
                # NOTE: the parameter name in baseset_dict cannot be ``config``;
                # otherwise instantiate() raises:
                # TypeError: instantiate() got multiple values for argument 'config'
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
        # For robustness, we will try to load the data until we succeed.
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

                data_dict = {}
                data_dict['dataset_name'] = content['dataset_name']
                data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]
                data_dict["states"] = states
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = (
                    state_elem_mask
                    if random.random() > self.cond_mask_prob
                    else np.zeros_like(state_elem_mask)
                )

                # ---- process rgb & depth images ----
                image_metas = list(self.pairwise(image_metas))
                depth_metas = list(self.pairwise(depth_metas))
                mask_probs = [self.cond_mask_prob] * self.img_history_size
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob

                preprocessed_images = []
                preprocessed_depths = []

                for cam_idx in range(self.num_cameras):
                    images_cam, image_mask_cam = image_metas[cam_idx]
                    depths_cam, depth_mask_cam = depth_metas[cam_idx]

                    # Collect per-frame validity flags for this camera.
                    cam_frames = []  # [(image, depth, rgb_valid, depth_valid), ...]
                    for j in range(self.img_history_size):
                        img, img_v = images_cam[j], image_mask_cam[j]
                        dep, dep_v = depths_cam[j], depth_mask_cam[j]
                        keep = random.random() > mask_probs[j]

                        rgb_valid = bool(img_v) and (math.prod(img.shape) > 0) and keep
                        depth_valid = bool(dep_v) and (math.prod(dep.shape) > 0) and keep

                        cam_frames.append((
                            img if rgb_valid else np.zeros_like(img),
                            dep if depth_valid else np.zeros_like(dep),
                            rgb_valid,
                            depth_valid,
                        ))

                    has_valid = any(f[2] for f in cam_frames)

                    # Decide an augmentation strategy shared by all frames of this camera.
                    do_aug = has_valid and self.image_aug and (random.random() > 0.5)
                    aug_type = random.choice(["corrput_only", "color_only", "both"]) if do_aug else None

                    # ColorJitter parameters (sampled once per camera).
                    cj_params = None
                    if do_aug and aug_type != "corrput_only":
                        cj = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)
                        cj_params = cj.get_params(cj.brightness, cj.contrast, cj.saturation, cj.hue)

                    # RandomCrop + Rotation parameters (sampled once per camera).
                    crop_params = None
                    rotation_angle = None
                    if do_aug and self.head_camera_randomcrop_aug:
                        sample_img = cam_frames[0][0]
                        if self.image_size is not None:
                            pil_tmp = transforms.Resize(self.image_size)(Image.fromarray(sample_img))
                            _w, _h = pil_tmp.size
                        else:
                            _h, _w = sample_img.shape[:2]
                        crop_h, crop_w = int(_h * 0.85), int(_w * 0.85)
                        crop_params = (random.randint(0, _h - crop_h),
                                       random.randint(0, _w - crop_w),
                                       crop_h, crop_w, _h, _w)
                        rotation_angle = random.uniform(-5, 5)

                    # Per-frame processing for RGB + depth.
                    for j in range(self.img_history_size):
                        img, dep, rgb_valid, depth_valid = cam_frames[j]

                        # ---- RGB ----
                        image = Image.fromarray(img)
                        if self.image_size is not None:
                            image = transforms.Resize(self.image_size)(image)

                        if rgb_valid and self.auto_adjust_image_brightness:
                            pixel_values = list(image.getdata())
                            avg_bright = sum(sum(p) for p in pixel_values) / (len(pixel_values) * 255.0 * 3)
                            if avg_bright <= 0.15:
                                image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

                        if rgb_valid and do_aug:
                            # ColorJitter (same parameters across this camera's frames).
                            if cj_params is not None:
                                fn_idx, bf, cf, sf, hf = cj_params
                                for fid in fn_idx:
                                    if fid == 0 and bf is not None:
                                        image = transforms.functional.adjust_brightness(image, bf)
                                    elif fid == 1 and cf is not None:
                                        image = transforms.functional.adjust_contrast(image, cf)
                                    elif fid == 2 and sf is not None:
                                        image = transforms.functional.adjust_saturation(image, sf)
                                    elif fid == 3 and hf is not None:
                                        image = transforms.functional.adjust_hue(image, hf)
                            # image_corrupt is independent per frame.
                            if aug_type != "color_only":
                                image = image_corrupt(image)

                        # ---- Depth ----
                        if self.image_size is not None:
                            h, w = dep.shape[:2]
                            scale = self.image_size / min(h, w)
                            dep = cv2.resize(dep, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

                        # ---- Shared geometric augmentation (crop + rotation) ----
                        if crop_params is not None:
                            ct, cl, ch, cw, oh, ow = crop_params
                            # RGB
                            if rgb_valid:
                                img_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
                                img_t = transforms.functional.crop(img_t, ct, cl, ch, cw)
                                img_t = transforms.functional.resize(img_t, [oh, ow])
                                img_t = transforms.functional.rotate(img_t, rotation_angle)
                                image = img_t[:3].permute(1, 2, 0).numpy()
                            # Depth
                            if depth_valid:
                                dep_t = torch.from_numpy(dep).unsqueeze(0).float()  # (1, H, W)
                                dep_t = transforms.functional.crop(dep_t, ct, cl, ch, cw)
                                dep_t = transforms.functional.resize(
                                    dep_t, [oh, ow], interpolation=transforms.InterpolationMode.NEAREST
                                )
                                dep_t = transforms.functional.rotate(dep_t, rotation_angle)
                                dep = dep_t.squeeze(0).numpy()  # (H, W)

                        preprocessed_images.append(np.array(image))
                        preprocessed_depths.append(np.array(dep)[..., None])

                # ---- process current/future images (optional) ----
                if "current_image" in res and "future_images" in res:
                    current_image = res["current_image"]
                    future_images = res["future_images"]

                    current_image, future_images = process_video_images(
                        current_image,
                        future_images,
                        self.video_height,
                        self.video_width,
                    )

                    res["current_image"] = current_image
                    res["future_images"] = future_images

                    data_dict["current_image"] = current_image
                    data_dict["future_images"] = future_images

                data_dict["images"] = [torch.from_numpy(img).permute(2, 0, 1) for img in preprocessed_images]
                data_dict["images"] = [
                    resize_with_pad(img.unsqueeze(0), self.image_size, self.image_size, pad_value=0)[0]
                    for img in data_dict["images"]
                ]
                data_dict["instructions"] = (
                    content["instruction"] if random.random() > self.cond_mask_prob else ""
                )
                data_dict["depths"] = [torch.from_numpy(depth).permute(2, 0, 1) for depth in preprocessed_depths]
                data_dict["depths"] = [
                    resize_with_pad(
                        depth.unsqueeze(0), self.image_size, self.image_size, pad_value=0, mode="nearest"
                    )[0]
                    for depth in data_dict["depths"]
                ]

                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v).float()  # numpy float64 -> tensor float32

                return data_dict
            except BaseException as e:
                if data_dict is not None:
                    print(f"Error catched when processing sample from {data_dict.get('dataset_name')}:", e)
                else:
                    print("Error catched when processing sample:", e)
                traceback.print_exc()
                index = (index + 1) % len(self)


def resize_with_pad(img, width, height, pad_value=-1, mode="bilinear"):
    """Aspect-preserving resize, then center-pad to (height, width)."""
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

    # Center pad.
    pw = pad_width // 2
    ph = pad_height // 2
    padded_img = F.pad(resized_img, (pw, pad_width - pw, ph, pad_height - ph), value=pad_value)
    return padded_img


# =============================================================================
# Lerobot-format action dataset factory (used by data_factory's lerobot branch)
# =============================================================================

import logging
from pprint import pformat

from pi0._lerobot_compat import (
    ImageTransforms,
    ImageTransformsConfig,
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
    resolve_delta_timestamps,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from omegaconf import DictConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def make_dataset(cfg_pi0: PreTrainedConfig, cfg: DictConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """

    transform_config = ImageTransformsConfig(enable=cfg.dataset.image_transforms.enable, max_num_transforms=3)
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
        # Read meta / data from lerobot_dir / dataset_config.repo_id.
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
        # Read meta / data from lerobot_dir / dataset_config.repo_id.
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id[0],
                                         root=lerobot_dir / cfg.dataset.repo_id[0],
                                         local_files_only=cfg.dataset.local_files_only)
        delta_timestamps = resolve_delta_timestamps(cfg_pi0, ds_meta)
        # MultiLeRobotDataset internally sets root = lerobot_dir / dataset_config.repo_id[0..n].
        dataset = MultiLeRobotDataset(
            dataset_config.repo_id,
            root=lerobot_dir,
            episodes=dataset_config.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=dataset_config.video_backend,
            local_files_only=dataset_config.local_files_only,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index , indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


# =============================================================================
# Test entry
# =============================================================================

import hydra
from embodied_pi0_action.utils.vis import (
    plot_all_images,
    plot_all_joints,
    plot_all_images_with_depth,
)


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="base",
)
def test(cfg):
    from collators import CollatorForActionConsumerDataset

    cfg.dataloader.batch_size = 1
    cfg.dataloader.num_workers = 1
    cfg.dataloader.shuffle = True
    cfg.dataloader.pin_memory = False
    cfg.dataloader.persistent_workers = False

    dataset = VLAConsumerDataset(cfg)
    print(len(dataset))
    print(dataset.__getitem__(0).keys())
    print(dataset.__getitem__(0)['states'].shape)
    print(dataset.__getitem__(0)['actions'].shape)
    print(dataset.__getitem__(0)['states'])
    print(dataset.__getitem__(0)['dataset_name'])
    print(dataset.__getitem__(0)['data_idx'])
    print(dataset.__getitem__(0)['instructions'])
    data_collator = CollatorForActionConsumerDataset(cfg)
    train_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset, collate_fn=data_collator)
    for batch in train_dataloader:
        print(batch.keys())
        print(batch['observation.images.top_head'].shape)
        print(batch['observation.images.hand_left'].shape)
        print(batch['observation.images.hand_right'].shape)
        print(batch['observation.state'].shape)
        print(batch['action'].shape)
        print(batch['task'])
        print(batch['observation.depths.top_head'].shape)
        print(batch['observation.depths.hand_left'].shape)
        print(batch['observation.depths.hand_right'].shape)

        plot_all_images_with_depth(
            batch['observation.images.top_head'][0],
            batch['observation.images.hand_left'][0],
            batch['observation.images.hand_right'][0],
            batch['observation.depths.top_head'][0],
            batch['observation.depths.hand_left'][0],
            batch['observation.depths.hand_right'][0],  # (HISTORY_SIZE, 1, H, W)
        )
        plot_all_joints(batch['action'][0], batch['observation.state'][0].unsqueeze(0))  # (N, D)

        plot_all_images(
            batch["future_images"][0],
            batch["future_images"][0],
            batch["future_images"][0],
        )
        break


if __name__ == "__main__":
    test()