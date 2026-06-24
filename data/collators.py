"""Unified collators for PoseVLA datasets.

Detection-style branch
----------------------
``CollatorForDetectionDataset`` is shared by all four detection datasets
(omni6d / bop / clutter / omni3d). They all emit samples with an identical
schema (3 cameras: top_head / hand_left / hand_right), so the collator
infers ``num_cameras`` directly from each sample and needs no config.

Action-style branch
-------------------
``CollatorForActionConsumerDataset`` is used together with
``datasets.dataset_hdf5.VLAConsumerDataset`` to assemble HDF5 action
batches in the format expected by the PI0 model.
"""
from typing import Any, Dict, Sequence

import torch


class CollatorForDetectionDataset:
    """Assemble a detection batch in the format expected by the PI0 model.

    All four detection datasets (omni6d / bop / clutter / omni3d) emit
    samples with the same multi-camera schema (3 cameras in the order
    ``top_head, hand_left, hand_right``), so this collator is parameter-free
    and infers ``num_cameras`` from the first sample at call time.
    """

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # Number of cameras is inferred from the first sample.
        num_cameras = len(instances[0]["images"])

        # Multi-camera image tensor: [B, num_cameras, 3, 224, 224], scaled to 0-1
        batch_images = [torch.stack(sample["images"], dim=0) for sample in instances]
        images_multi = torch.stack(batch_images, dim=0).float() / 255.0

        # images_vggt: [B, 1, num_cameras, 3, 518, 518]
        images_vggt = torch.stack(
            [torch.stack(sample["images_vggt"], dim=0) for sample in instances],
            dim=0,
        ).unsqueeze(1)

        # intrinsics: [B, num_cameras, 4]
        intrinsics_multi = torch.stack(
            [torch.stack(sample["camera_intrinsics"], dim=0) for sample in instances],
            dim=0,
        )

        # K: [B, num_cameras, 3, 3]
        K_multi = torch.stack(
            [torch.stack([sample["camera_K"][i] for i in range(num_cameras)], dim=0)
             for sample in instances],
            dim=0,
        )

        # rays: [B, num_cameras, 3, H, W]
        rays_multi = torch.stack(
            [torch.stack([sample["camera_rays"][i] for i in range(num_cameras)], dim=0)
             for sample in instances],
            dim=0,
        )

        # know_depth: [B, num_cameras, 2, H, W]
        know_depth_multi = torch.stack(
            [torch.stack([sample["camera_know_depth"][i] for i in range(num_cameras)], dim=0)
             for sample in instances],
            dim=0,
        )

        # depth: [B, num_cameras, H, W]
        depth_multi = torch.stack(
            [torch.stack([sample["camera_depth"][i].squeeze(0) for i in range(num_cameras)], dim=0)
             for sample in instances],
            dim=0,
        )

        batch_format = {
            "observation.images.top_head":   images_multi[:, 0, :].unsqueeze(1),
            "observation.images.hand_left":  images_multi[:, 1, :].unsqueeze(1),
            "observation.images.hand_right": images_multi[:, 2, :].unsqueeze(1),
            "observation.depths.top_head":   depth_multi[:, 0, :].unsqueeze(1),
            "observation.depths.hand_left":  depth_multi[:, 1, :].unsqueeze(1),
            "observation.depths.hand_right": depth_multi[:, 2, :].unsqueeze(1),
            "observation.depth_priors.top_head":   know_depth_multi[:, 0, :].unsqueeze(1),
            "observation.depth_priors.hand_left":  know_depth_multi[:, 1, :].unsqueeze(1),
            "observation.depth_priors.hand_right": know_depth_multi[:, 2, :].unsqueeze(1),
            "observation.rays.top_head":   rays_multi[:, 0, :].unsqueeze(1),
            "observation.rays.hand_left":  rays_multi[:, 1, :].unsqueeze(1),
            "observation.rays.hand_right": rays_multi[:, 2, :].unsqueeze(1),
            "observation.images.top_head.intrinsics":   intrinsics_multi[:, 0, :],
            "observation.images.hand_left.intrinsics":  intrinsics_multi[:, 1, :],
            "observation.images.hand_right.intrinsics": intrinsics_multi[:, 2, :],
            "task":        [sample["task"]                   for sample in instances],
            "text_label":  [sample["text_label"]             for sample in instances],
            "pose":        [sample["camera_pose"]            for sample in instances],
            "size":        [sample["camera_bbox_side_len"]   for sample in instances],
            "class_label": [sample["camera_class_labels"]    for sample in instances],
            "bbox_2d":     [sample["bbox_2d"]                for sample in instances],
        }
        return batch_format


class CollatorForActionConsumerDataset:
    """Collate examples for HDF5 VLA action training (paired with
    ``datasets.dataset_hdf5.VLAConsumerDataset``).
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.IMG_HISORY_SIZE = config.dataset.img_history_size

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. Initialize base + optional fields.
        base_keys = ["states", "actions", "images", "instructions", "depths"]
        optional_keys = ["current_image", "future_images"]

        batch = {k: [] for k in base_keys}
        for ok in optional_keys:
            if ok in instances[0]:
                batch[ok] = []

        # 2. Collect data and convert to tensors.
        for instance in instances:
            for key in ['states', 'actions']:
                batch[key].append(torch.as_tensor(instance[key]))

            # Stack image / depth sequences: (cams * history, C, H, W).
            batch["images"].append(torch.stack(instance["images"], dim=0))
            batch["depths"].append(torch.stack(instance["depths"], dim=0))

            batch["instructions"].append(instance["instructions"])

            for ok in optional_keys:
                if ok in batch:
                    batch[ok].append(torch.as_tensor(instance[ok]))

        # 3. Stack lists into tensors.
        stacked = {}
        for key, value in batch.items():
            if isinstance(value[0], torch.Tensor):
                stacked[key] = torch.stack(value, dim=0)
            else:
                stacked[key] = value

        # 4. Map to the final batch format.
        h = self.IMG_HISORY_SIZE
        all_imgs = stacked["images"].float()
        all_depths = stacked["depths"]

        batch_format = {
            # Image branch (normalized to [0, 1]).
            "observation.images.top_head":   all_imgs[:, 0:h] / 255.0,
            "observation.images.hand_left":  all_imgs[:, h:2 * h] / 255.0,
            "observation.images.hand_right": all_imgs[:, 2 * h:3 * h] / 255.0,

            # Depth branch.
            "observation.depths.top_head":   all_depths[:, 0:h],
            "observation.depths.hand_left":  all_depths[:, h:2 * h],
            "observation.depths.hand_right": all_depths[:, 2 * h:3 * h],

            # State and action.
            "observation.state": stacked["states"][:, 0, :],  # current frame state, (B, D)
            "action": stacked["actions"],                     # (B, Chunk, D)
            "task": stacked["instructions"],                  # List[str]
        }

        # 5. Forward optional fields if present.
        for ok in optional_keys:
            if ok in stacked:
                batch_format[ok] = stacked[ok]

        return batch_format


__all__ = [
    "CollatorForDetectionDataset",
    "CollatorForActionConsumerDataset",
]