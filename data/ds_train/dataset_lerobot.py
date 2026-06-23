import logging
from pprint import pformat
from pathlib import Path
import torch
from data.ds_raw.interndata_a1 import load_interndata_a1
from data.ds_raw.interndata_a1 import REWARD, ACTION, OBS_PREFIX, FEATURE_MAPPING, IMAGE_MAPPING
from mapping_token import decode_text_to_scene_with_tokenizer, BinTokenizer, decode_text_to_scene
from data.ds_train.dataset_agibot import DataCollatorForAgibotConsumerDataset
from mapping_token import text_to_class_attr_dict_tokenizer, BinTokenizer, text_to_class_attr_dict
from utils.vis import visualize_2d_3d_all, visualize_traj, visualize_views
from torch.utils.data import ConcatDataset
import time

from pi0._lerobot_compat import (
    ImageTransforms,
    ImageTransformsConfig,
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
    StreamingLeRobotDataset,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from omegaconf import DictConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
        cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        elif key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        elif key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        # important!
        elif key in FEATURE_MAPPING[ds_meta.robot_type][ACTION] and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        # if key in IMAGE_MAPPING[ds_meta.robot_type].keys() and hasattr(cfg, "image_delta_indices") and cfg.image_delta_indices is not None:
        #     delta_timestamps[key] = [i / ds_meta.fps for i in cfg.image_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps



def make_dataset(cfg_pi0: PreTrainedConfig, cfg: DictConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    transform_config = ImageTransformsConfig(enable=cfg.image_transforms.enable, max_num_transforms=3)

    # filter augmentation
    transform_keys = ["brightness", "contrast", "saturation", "hue", "sharpness", "affine"]
    for key in transform_keys:
        if hasattr(cfg.image_transforms, key):
            is_enabled = getattr(cfg.image_transforms, key)
            if not is_enabled:
                transform_config.tfs.pop(key, None)

    dataset_config = DatasetConfig(
        repo_id=cfg.repo_id,
        episodes=cfg.episodes,
        image_transforms=transform_config,
        video_backend=cfg.video_backend,
    )

    image_transforms = (
        ImageTransforms(dataset_config.image_transforms) if dataset_config.image_transforms.enable else None
    )

    lerobot_dir = Path(cfg.lerobot_dir)
    # print(f"Training Model on {cfg.repo_id} from {lerobot_dir}")

    if isinstance(dataset_config.repo_id, str):
        # 直接从lerobot_dir / dataset_config.repo_id读取meta, data等
        ds_meta = LeRobotDatasetMetadata(dataset_config.repo_id,
                                         root=lerobot_dir / dataset_config.repo_id,
                                         local_files_only=dataset_config.local_files_only)
        delta_timestamps = resolve_delta_timestamps(cfg_pi0, ds_meta)
        dataset = LeRobotDataset(
            dataset_config.repo_id,
            root=lerobot_dir / dataset_config.repo_id,
            episodes=dataset_config.episodes,
            delta_timestamps=delta_timestamps,
            tolerance_s=0.01,
            image_transforms=image_transforms,
            video_backend=dataset_config.video_backend,
            # local_files_only=dataset_config.local_files_only,
        )
    else:
        # 直接从lerobot_dir / dataset_config.repo_id读取meta, data等
        ds_meta = LeRobotDatasetMetadata(cfg.repo_id[0],
                                         root=lerobot_dir / cfg.repo_id[0],
                                         # local_files_only=cfg.local_files_only
                                         )
        delta_timestamps = resolve_delta_timestamps(cfg_pi0, ds_meta)
        # 会自动在MultiLeRobotDataset内部设置root=lerobot_dir/dataset_config.repo_id[0,1,...,n]
        dataset = MultiLeRobotDataset(
            dataset_config.repo_id,
            root=lerobot_dir,
            episodes=dataset_config.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=dataset_config.video_backend,
            # local_files_only=dataset_config.local_files_only,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index , indent=2)}"
        )

    if cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

import hydra
from embodied_pi0_action.utils.vis import plot_all_images, plot_all_joints, plot_all_images_with_depth, batch_overlay_gaussian_mask
from pi0.modeling_pi0 import PI0Config
@hydra.main(
        version_base=None,
    config_path="../../config",
        config_name="base",
    )
def test(cfg):
    cfg.dataloader.batch_size = 1
    cfg.dataloader.num_workers = 1
    cfg.dataloader.shuffle = True
    cfg.dataloader.pin_memory = False
    cfg.dataloader.persistent_workers = False
    bin_tokenizer = BinTokenizer(cfg.statistics_path_6d_dataset)
    pi0_config = PI0Config(
        tokenizer_model_path=(cfg.model.tokenizer_model_path),
        n_action_steps=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,
        chunk_size=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,  # not used
        optimizer_lr=cfg.training.optimizer_lr,
        optimizer_betas=tuple(cfg.training.optimizer_betas),
        optimizer_eps=cfg.training.optimizer_eps,
        optimizer_weight_decay=cfg.training.optimizer_weight_decay,
        scheduler_warmup_steps=cfg.training.scheduler_warmup_steps,
        scheduler_decay_steps=cfg.training.scheduler_decay_steps,
        scheduler_decay_lr=cfg.training.scheduler_decay_lr,
        is_knowledge_insulation=cfg.training.is_knowledge_insulation,
        resize_imgs_with_padding=(cfg.dataset.image_size, cfg.dataset.image_size),
        pi05=cfg.training.pi05,
        vis_attn=cfg.training.vis_attn,
        add_extra_token=cfg.training.add_extra_token,
        add_image_token=cfg.training.add_image_token,
        add_prior=cfg.training.add_prior)

    # yaml_name = "lerobot_group01"
    # group_cfg = getattr(cfg.dataset_lerobot, yaml_name)
    # dataset = make_dataset(pi0_config, group_cfg)
    #
    # dataset = LeRobotDatasetWrapper(cfg, dataset, bin_tokenizer)

    # yaml_names = ["lerobot_group01", "lerobot_group02", "lerobot_group03",
    #               "lerobot_group04", "lerobot_group05", "lerobot_group06"]
    yaml_names = ["lerobot_group06"]
    all_wrapped_datasets = load_interndata_a1(pi0_config, cfg, bin_tokenizer, make_dataset, yaml_names)
    dataset = ConcatDataset(all_wrapped_datasets)

    print(len(dataset))
    print(dataset.__getitem__(0).keys())
    print(dataset.__getitem__(0)['states'].shape)
    print(dataset.__getitem__(0)['actions'].shape)
    print(dataset.__getitem__(0)['states'])
    print(dataset.__getitem__(0)['dataset_name'])
    print(dataset.__getitem__(0)['data_idx'])
    print(dataset.__getitem__(0)['instructions'])
    data_collator = DataCollatorForAgibotConsumerDataset(cfg)
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

        plot_all_images(batch['observation.images.top_head'][0],
                        batch['observation.images.hand_left'][0],
                        batch['observation.images.hand_right'][0])
        plot_all_images_with_depth(batch['observation.images.top_head'][0],
                                   batch['observation.images.hand_left'][0],
                                   batch['observation.images.hand_right'][0],
                                   batch['observation.depths.top_head'][0],
                                   batch['observation.depths.hand_left'][0],
                                   batch['observation.depths.hand_right'][0]  # (HISTORY_SIZE, 1, H, W)
                                   )
        plot_all_images_with_depth(batch['observation.images.top_head'][0],
                                   batch['observation.images.hand_left'][0],
                                   batch['observation.images.hand_right'][0],
                                   batch['observation.depth_priors.top_head'][0][:, 1],
                                   batch['observation.depth_priors.hand_left'][0][:, 1],
                                   batch['observation.depth_priors.hand_right'][0][:, 1]  # (HISTORY_SIZE, 1, H, W)
                                   )
        plot_all_joints(batch['action'][0], batch['observation.state'][0].unsqueeze(0))  # (N, D)

        if cfg.uniform_mapping_6d_dataset:
            res = decode_text_to_scene(batch["text_label"][0])
        else:
            res = decode_text_to_scene_with_tokenizer(batch["text_label"][0], bin_tokenizer)
        print(res)

        images = {"image0": batch["observation.images.top_head"][0, 0],
                  "image1": batch["observation.images.hand_left"][0, 0],
                  "image2": batch["observation.images.hand_right"][0, 0]}
        intrinsics = {"image0": batch["observation.images.top_head.intrinsics"][0],
                      "image1": batch["observation.images.hand_left.intrinsics"][0],
                      "image2": batch["observation.images.hand_right.intrinsics"][0]}
        depths = {"image0": batch["observation.depth_priors.top_head"][0, 0],
                  "image1": batch["observation.depth_priors.hand_left"][0, 0],
                  "image2": batch["observation.depth_priors.hand_right"][0, 0]}
        rays = {"image0": batch["observation.rays.top_head"][0, 0],
                "image1": batch["observation.rays.hand_left"][0, 0],
                "image2": batch["observation.rays.hand_right"][0, 0]}

        # print(text_label)
        visualize_2d_3d_all(images, res, res, intrinsics, batch["task"][0], vis_2d=True, vis_3d=True)
        visualize_views(images, depths, rays)

        break


if __name__ == "__main__":
    test()