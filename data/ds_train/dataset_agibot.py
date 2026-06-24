import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from embodied_pi0_action.utils.image_corrupt import image_corrupt
from pathlib import Path
from torch.utils.data import ConcatDataset
from hydra.utils import instantiate
from utils.mapping_token import BinTokenizer
import cv2
from data.ds_train.dataset_omni6d import _build_sparse_depth, _generate_rays
from utils.vis import visualize_2d_3d_all, visualize_traj, visualize_views
from utils.mapping_token import decode_text_to_scene, encode_scene_to_text
from utils.mapping_token import decode_text_to_scene_with_tokenizer, encode_scene_to_text_with_tokenizer, BinTokenizer, broadcast_class_names, encode_scene_to_text_with_tokenizer_ablation
from data.ds_raw.interndata_a1 import resize_with_pad

class AgibotConsumerDataset(Dataset):
    """A vision-languange-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """

    def __init__(self, config: dict, tokenizer: BinTokenizer = None, action_type: str = 'ee', future_video: bool = False, global_downsample_rate: int=1):
        super(AgibotConsumerDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.num_cameras = config.dataset.num_cameras
        self.img_history_size = config.dataset.img_history_size
        self.cond_mask_prob = config.dataset.cond_mask_prob
        self.cam_ext_mask_prob = config.dataset.cam_ext_mask_prob
        self.use_hdf5 = config.dataset.use_hdf5
        self.hdf5_dataset = None


        self.DATASET_NAMES = []
        if self.use_hdf5:
            self.base_dataset_list = []

            # Instantiate each base dataset with common configuration
            print(f"Loading dataset {config.dataset.dataset_list}")
            for baseset_dict in config.dataset.dataset_list:
                print(f"Loading dataset {baseset_dict}")
                # baseset_dict的参数名不能为config! TypeError: instantiate() got multiple values for argument 'config'
                baseset = instantiate(baseset_dict, cfg=config, tokenizer=tokenizer, action_type=action_type, future_video=future_video, global_downsample_rate=global_downsample_rate, _recursive_=False)  # set _recursive_=False!
                self.base_dataset_list.append(baseset)

                self.DATASET_NAMES.extend(list(baseset_dict.sample_weights.keys()))

            self.hdf5_dataset = ConcatDataset(self.base_dataset_list)
            # Create the mapping between dataset name and id
            self.dataset_name2id = {name: i for i, name in enumerate(self.DATASET_NAMES)}
            self.dataset_id2name = {i: name for i, name in enumerate(self.DATASET_NAMES)}
            print(self.dataset_name2id)
            print(f"Training Model on `{list(self.DATASET_NAMES)}` from `{config.dataset.hdf5_dir}`")

        else:
            raise NotImplementedError
        # Load dataset stat
        # with open(config.dataset.dataset_stat_json, 'r') as f:
        #     dataset_stat = json.load(f)
        # self.dataset_stat = dataset_stat

        # self.tokenizer = tokenizer
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
        # For robustness, we will try to load the data until we succeed
        while True:
            data_dict = None
            try:
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

                intrinsics = res["intrinsics"]
                poses = res["poses"]
                # text_labels = res["text_labels"]
                multi_2ds = res["multi_2ds"] #[(N,2), (N,2), (N,2)]
                multi_poses = res["multi_poses"]
                class_names = res["class_names"]
                multi_images = res["multi_images"]


                data_dict = {}
                data_dict['dataset_name'] = content['dataset_name']
                data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]
                data_dict["states"] = states
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = state_elem_mask \
                    if random.random() > self.cond_mask_prob else np.zeros_like(state_elem_mask)

                '''process rgb image'''
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
                            # We replace the invalid images with the background image
                            # and also randomly mask images by the background image
                            background_image = np.zeros_like(image)
                            rearranged_images.append((background_image.copy(), False))
                            background_det = np.zeros_like(det)
                            rearranged_dets.append((background_det.copy()))

                preprocessed_images = []
                preprocessed_dets = []
                for i, ((image, valid), det) in enumerate(zip(rearranged_images, rearranged_dets)):
                    image = Image.fromarray(image)
                    # if self.image_size is not None:
                    #     image = transforms.Resize(self.image_size)(image) # will keep ratio (H, W, 3)
                    #     det = Image.fromarray(det)
                    #     det = transforms.Resize(self.image_size)(det) # will keep ratio (H, W)
                    #     det = np.array(det)

                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

                    # Only apply image augmentation to 50% of the images
                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice([
                            "corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)

                        # "top camera random resize"
                        # if False:
                        # # if self.head_camera_randomcrop_aug:
                        #     # if i==0: # head camera
                        #     # if True: # apply aug to all images
                        #         det = det[..., None]  # [h, w, 1]
                        #         cat_img = np.concatenate([image, det], axis=2)  # [h, w, 4]
                        #         cat_img = torch.from_numpy(cat_img).permute(2, 0, 1).float()  # [4, h, w]
                        #
                        #         width, height = image.size
                        #         transform = transforms.Compose([
                        #             # Please provide only two dimensions(h, w) for size.
                        #             transforms.RandomCrop((int(height * 0.85), int(width * 0.85))),  # 随机裁剪
                        #             transforms.Resize((height, width)),  # 调整大小
                        #             transforms.RandomRotation(degrees=(-5, 5)),  # 随机旋转
                        #         ])
                        #         # image = transform(image)
                        #
                        #         cat_img = transform(cat_img)
                        #         image = cat_img[:3].permute(1, 2, 0).numpy()
                        #         det = cat_img[3].numpy()  # [h, w]

                    preprocessed_images.append(np.array(image)) # [<np.array>, ..., <np.array>]
                    preprocessed_dets.append(np.array(det)[..., None]) # [[h,w,1],...[h,w,1]]

                '''process depth image'''
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
                    preprocessed_depths.append(np.array(depth)[..., None])  # [<np.array>, ..., <np.array>] (H,W,1)

                new_images = []
                new_pts = []
                padded_depths = []
                new_intrinsics = []
                valid_masks = []
                sparse_depths = []
                camera_rays = []

                # 同时遍历所有相关数据
                # zip(images, depths, pts, intrinsics)
                for img_np, depth_np, pts, K_np in zip(preprocessed_images, preprocessed_depths, multi_2ds, intrinsics):
                    # 1. 统一转换为 Tensor 并增加 Batch 维度 (1, C, H, W)
                    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)  # (1, 3, H, W)
                    depth_t = torch.from_numpy(depth_np).permute(2, 0, 1).float().unsqueeze(0)  # (1, 1, H, W)
                    K_t = torch.from_numpy(K_np).float()  # (3, 3)

                    # 2. 调用一次函数，处理所有模态
                    # 注意：random_crop_ratio_range 建议在这里统一传入
                    padded_img, padded_depth, valid_mask, new_K, pts_new = resize_with_pad(
                        img=img_t,
                        width=self.config.dataset.image_size,
                        height=self.config.dataset.image_size,
                        depth=depth_t,
                        random_crop_ratio_range=self.config.dataset.get("random_crop_ratio_range", (0.8, 1.0)),
                        pad_value=0,
                        intrinsic=K_t,
                        pts=pts
                    )

                    # 3. 移除 Batch 维度并存入列表
                    # RGB
                    new_images.append(padded_img[0])  # (3, H, W)

                    # Points
                    new_pts.append(pts_new.squeeze(0) if len(pts_new) > 0 else [])# 如果 pts 是 []，这里依然是 []

                    # Depth
                    p_depth = padded_depth.squeeze()  # (H, W)
                    padded_depths.append(p_depth)

                    # Intrinsic
                    p_K = new_K[0]  # (3, 3)
                    new_intrinsics.append(p_K)

                    # Mask
                    p_mask = valid_mask.squeeze()  # (H, W)
                    valid_masks.append(p_mask)

                    # 4. 构造稀疏深度 (基于对齐后的深度图)
                    depth_valid_mask = (p_depth > 0).float()
                    sparse = torch.stack([p_depth, depth_valid_mask], dim=0)  # (2, H, W)
                    sparse_depths.append(sparse)

                    # 5. 构造视线 (基于对齐后的内参和 Mask)
                    H, W = p_depth.shape[-2:]
                    rays = _generate_rays(H, W, p_K, p_mask)  # (3, H, W)
                    camera_rays.append(rays)

                if random.random() > self.config.depth_full_prob:
                    sparse_depths = [torch.zeros_like(sparse) for sparse in sparse_depths]

                if random.random() > self.config.ray_prob:
                    camera_rays = [torch.zeros_like(ray) for ray in camera_rays]

                data_dict["instructions"] = content["instruction"] if random.random() > self.cond_mask_prob else ""
                multi_2ds = new_pts  # 更新后的坐标
                data_dict["images"] = new_images
                data_dict["depths"] = padded_depths  # list of (H, W)
                data_dict["intrinsics"] = new_intrinsics  # list of (3, 3)
                data_dict["depth_priors"] = sparse_depths  # list of (2, H, W)
                data_dict["camera_rays"] = camera_rays  # list of (3, H, W)

                data_dict["dets"] = [torch.from_numpy(det).permute(2, 0, 1) for det in preprocessed_dets]
                data_dict["dets"] = [resize_with_pad(det.unsqueeze(0),
                                                       self.image_size,
                                                       self.image_size,
                                                       pad_value=0, mode="nearest")[0] for det in data_dict["dets"]]

                data_dict["camera_pose"] = poses
                text_labels = encode_scene_to_text_with_tokenizer(class_names=class_names,
                                                              multi_images=data_dict["images"],
                                                              multi_2ds=multi_2ds,
                                                              multi_poses=multi_poses,
                                                              multi_sizes=broadcast_class_names(multi_poses, None),
                                                              tokenizer=self.tokenizer,
                                                              near2far=False)

                #TODO:  ablation: output pose directly
                # text_labels = encode_scene_to_text_with_tokenizer_ablation(class_names=class_names,
                #                                                         multi_images=data_dict["images"],
                #                                                         multi_2ds=broadcast_class_names(multi_poses, None),
                #                                                         multi_poses=multi_poses,
                #                                                         multi_sizes=broadcast_class_names(multi_poses, None),
                #                                                         tokenizer=self.tokenizer,
                #                                                         near2far=False)

                data_dict["text_labels"] = text_labels

                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v).float() # numpy float64 -> tensor float32

                return data_dict
            except BaseException as e:
                # Print the error info
                if data_dict is not None:
                    print(f"Error catched when processing sample from {data_dict.get('dataset_name')}:", e)
                else:
                    print(f"Error catched when processing sample:", e)
                traceback.print_exc()
                # Try incresing the index
                index = (index + 1) % len(self)


class DataCollatorForAgibotConsumerDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, config) -> None:
        self.config = config
        self.IMG_HISORY_SIZE = config.dataset.img_history_size

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # batch = {
        #     "states": [],
        #     "actions": [],
        #     "state_elem_mask": [],
        #     "images": [],
        #     "instructions": [],
        #     "depths": [],
        #     "dets": []
        # }
        #
        # for instance in instances:
        #     # Convert all the numpy arrays to tensor
        #     keys_to_check = ['states', 'actions', 'state_elem_mask']
        #     for key in keys_to_check:
        #         if isinstance(instance[key], torch.Tensor):
        #             item = instance[key]
        #         else:
        #             item = torch.from_numpy(instance[key])
        #         batch[key].append(item)

            # instance["images"]: [(3,H,W), (3,H,W), ... (3,H,W)] history * cams
            # torch.stack(instance["images"], dim=0) (history * cams, 3, H, W)
        batch_images = [torch.stack(sample["images"], dim=0) for sample in instances]
        batch_depths = [torch.stack(sample["depths"], dim=0) for sample in instances]
        batch_depth_priors = [torch.stack(sample["depth_priors"], dim=0) for sample in instances]

        rays_multi = [torch.stack(sample["camera_rays"], dim=0) for sample in instances] #[(N,3,h,w)]
        rays_multi = torch.stack(rays_multi, dim=0) #(B,N,3,h,w)
        intrinsics_multi = [torch.stack(sample["intrinsics"], dim=0) for sample in instances] #[(N,3,3)]
        intrinsics_multi = torch.stack(intrinsics_multi, dim=0)  # (B,N,3,3)

        batch_poses = [sample["camera_pose"] for sample in instances]
        batch_states = torch.stack([sample["states"] for sample in instances], dim=0)
        batch_actions = torch.stack([sample["actions"] for sample in instances], dim=0)

        batch_tasks = [sample["instructions"] for sample in instances]
        batch_text_labels = [sample["text_labels"] for sample in instances]

        # # prob = 0.5 with ray map
        # B, C, H, W, D = rays_multi.shape  # C=3
        # prob_zero = torch.tensor([0.5, 0.5, 0.5])  # shape (3,)
        # rand_vals = torch.rand(B, 3)  # shape (B,3)
        # mask = rand_vals > prob_zero[None, :]  # shape (B,3)
        #
        # mask = mask.to(rays_multi.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1,1)
        # mask = mask.expand(-1, -1, H, W, D)
        # rays_multi = rays_multi * mask


        # batch["images"]: [(history * cams, 3, H, W), ..., (history * cams, 3, H, W)] B
        batch_format = {
            "observation.images.top_head": torch.stack([v[0:self.IMG_HISORY_SIZE] for v in batch_images], dim=0) / 255.,
            # (B,IMG_HISORY_SIZE,C,H,W) 0-1
            "observation.images.hand_left": torch.stack([v[self.IMG_HISORY_SIZE:self.IMG_HISORY_SIZE*2] for v in batch_images], dim=0) / 255.,
            # (B,IMG_HISORY_SIZE,C,H,W) 0-1
            "observation.images.hand_right": torch.stack([v[self.IMG_HISORY_SIZE*2:self.IMG_HISORY_SIZE*3] for v in batch_images], dim=0) / 255.,
            # (B,IMG_HISORY_SIZE,C,H,W) 0-1
            "observation.state": batch_states[:, 0, :], #batch["states"][:, 0, :],  # (B, D)
            "action": batch_actions, #batch["actions"][:, :, :],  # (B, Chunk, D)
            'task': batch_tasks,  # List: len(bs)

            # [B, history, 1, h, w])
            "observation.depths.top_head": torch.stack([v[0:self.IMG_HISORY_SIZE] for v in batch_depths], dim=0),
            "observation.depths.hand_left": torch.stack([v[self.IMG_HISORY_SIZE:self.IMG_HISORY_SIZE*2] for v in batch_depths], dim=0),
            "observation.depths.hand_right": torch.stack([v[self.IMG_HISORY_SIZE*2:self.IMG_HISORY_SIZE*3] for v in batch_depths], dim=0),

            # [B, history, 2, h, w])
            "observation.depth_priors.top_head": torch.stack([v[0:self.IMG_HISORY_SIZE] for v in batch_depth_priors], dim=0),
            "observation.depth_priors.hand_left": torch.stack([v[self.IMG_HISORY_SIZE:self.IMG_HISORY_SIZE * 2] for v in batch_depth_priors], dim=0),
            "observation.depth_priors.hand_right": torch.stack([v[self.IMG_HISORY_SIZE * 2:self.IMG_HISORY_SIZE * 3] for v in batch_depth_priors], dim=0),

            "observation.rays.top_head": rays_multi[:, 0, :].unsqueeze(1),  # (B,History,3,H,W)
            "observation.rays.hand_left": rays_multi[:, 1, :].unsqueeze(1),
            "observation.rays.hand_right": rays_multi[:, 2, :].unsqueeze(1),

            "text_label": batch_text_labels,

            "observation.images.top_head.intrinsics": intrinsics_multi[:, 0, :], #(B,3,3)
            "observation.images.hand_left.intrinsics": intrinsics_multi[:, 1, :],
            "observation.images.hand_right.intrinsics": intrinsics_multi[:, 2, :],

            "pose": batch_poses,

        }

        return batch_format


import hydra
from embodied_pi0_action.utils.vis import plot_all_images, plot_all_joints, plot_all_images_with_depth, batch_overlay_gaussian_mask

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
    cfg.dataset.dataset_list = [cfg.dataset.agibot]

    bin_tokenizer = BinTokenizer(cfg.statistics_path_6d_dataset)

    dataset = AgibotConsumerDataset(cfg, tokenizer=bin_tokenizer, action_type='ee', global_downsample_rate=3)
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
        print(batch['observation.depth_priors.top_head'].shape)
        print(batch['observation.depth_priors.hand_left'].shape)
        print(batch['observation.depth_priors.hand_right'].shape)

        plot_all_images(batch['observation.images.top_head'][0],
                        batch['observation.images.hand_left'][0],
                        batch['observation.images.hand_right'][0])
        plot_all_images_with_depth(batch['observation.images.top_head'][0],
                                   batch['observation.images.hand_left'][0],
                                   batch['observation.images.hand_right'][0],
                                   batch['observation.depths.top_head'][0],
                                   batch['observation.depths.hand_left'][0],
                                   batch['observation.depths.hand_right'][0] #(HISTORY_SIZE, 1, H, W)
                                   )
        plot_all_images_with_depth(batch['observation.images.top_head'][0],
                                   batch['observation.images.hand_left'][0],
                                   batch['observation.images.hand_right'][0],
                                   batch['observation.depth_priors.top_head'][0][:, 1],
                                   batch['observation.depth_priors.hand_left'][0][:, 1],
                                   batch['observation.depth_priors.hand_right'][0][:, 1]  # (HISTORY_SIZE, 1, H, W)
                                   )
        plot_all_joints(batch['action'][0], batch['observation.state'][0].unsqueeze(0)) #(N, D)

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
        visualize_2d_3d_all(images, res, res, intrinsics,  batch["task"][0], vis_2d=True, vis_3d=True)
        visualize_views(images, depths, rays)

        break

if __name__ == "__main__":
    test()