from posevla.modeling_posevla import PoseVLAPolicy, PoseVLAConfig, bin_tokenizer
from data.collators import CollatorForDetectionDataset
from data.ds_train.detection.dataset_omni3d import Omni3DConsumerDataset
from data.ds_train.detection.dataset_omni6d import _generate_rays
from utils.mapping_token import decode_text_to_scene_with_tokenizer
from utils.vis import visualize_2d_3d_all, visualize_views
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import h5py
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# resize_with_pad（支持 intrinsic 和 pts 同步变换）
# ============================================================
def resize_with_pad(
    img, width, height, pad_value=-1, mode="bilinear", intrinsic=None, pts=None
):
    """
    对输入图片 img 进行缩放和居中 padding，支持内参和点坐标同步变换。
    参数：
        img: (b, c, h, w)  torch.Tensor
        width, height:      输出目标图片尺寸
        pad_value:          填充像素值
        mode:               插值方式
        intrinsic:          (b,3,3) or (3,3) or None
        pts:                (N,2) or (b,N,2) or None
    返回：
        padded_img, valid_mask, new_intrinsic, pts_new
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    device = img.device
    b, c, cur_height, cur_width = img.shape

    # 计算缩放比例
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # 插值缩放
    interpolate_params = {
        'size': (resized_height, resized_width),
        'mode': mode,
    }
    if mode != "nearest":
        interpolate_params['align_corners'] = False
    resized_img = F.interpolate(img, **interpolate_params)

    # pad到中心
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    pw = pad_width // 2
    ph = pad_height // 2
    padded_img = F.pad(resized_img, (pw, pad_width - pw, ph, pad_height - ph), value=pad_value)

    # 有效区域mask
    valid_mask = torch.zeros((b, 1, height, width), device=device, dtype=padded_img.dtype)
    valid_mask[..., ph:ph + resized_height, pw:pw + resized_width] = 1

    # 缩放因子
    scale_x = resized_width / cur_width
    scale_y = resized_height / cur_height

    # intrinsic变换
    new_intrinsic = None
    if intrinsic is not None:
        K = intrinsic
        if K.dim() == 2:
            K = K[None].expand(b, -1, -1)
        elif K.dim() == 3 and K.shape[0] == 1:
            K = K.expand(b, -1, -1)
        assert K.shape == (b, 3, 3), f"intrinsic shape should be (b,3,3), got {K.shape}"

        new_intrinsic = K.clone()
        new_intrinsic[:, 0, 0] *= scale_x
        new_intrinsic[:, 1, 1] *= scale_y
        new_intrinsic[:, 0, 2] = new_intrinsic[:, 0, 2] * scale_x + pw
        new_intrinsic[:, 1, 2] = new_intrinsic[:, 1, 2] * scale_y + ph

    # pts变换
    pts_new = None
    if pts is not None:
        if isinstance(pts, torch.Tensor):
            pts_new = pts.clone()
            if pts_new.numel() > 0:
                pts_new[..., 0] = pts_new[..., 0] * scale_x + pw
                pts_new[..., 1] = pts_new[..., 1] * scale_y + ph
        elif isinstance(pts, np.ndarray):
            pts_new = pts.copy()
            if pts_new.size > 0:
                pts_new[..., 0] = pts_new[..., 0] * scale_x + pw
                pts_new[..., 1] = pts_new[..., 1] * scale_y + ph
        elif isinstance(pts, (list, tuple)) and len(pts) == 0:
            pts_new = pts

    return padded_img, valid_mask, new_intrinsic, pts_new


# ============================================================
# HDF5 数据解析
# ============================================================
camera_keys = {
    "high": "observations/images/cam_high",
    "left": "observations/images/cam_left_wrist",
    "right": "observations/images/cam_right_wrist"
}

depth_keys = {
    "high": "observations/depth_images/cam_high",
    "left": "observations/depth_images/cam_left_wrist",
    "right": "observations/depth_images/cam_right_wrist"
}

# ===== 默认 hdf5 文件路径（可按需修改） =====
hdf5_file = "/home/hetolin/robot_code/embodied_pi0_action/mydata/xtrainer2/chuangyuan/hdf5/D2184/pnp_773_50hz_depth/20250812-164358.hdf5" # 1100, 100


def parse_data(hdf5_file, key, idx):
    """从 hdf5 文件中解析指定帧的 RGB 图像和深度图。"""
    with h5py.File(hdf5_file, 'r') as f:
        N = f[camera_keys["high"]].shape[0]
        print(f"Total frames: {N}")

        img = f[key][idx]  # (H,W,3), uint8
        img_bgr = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth_key = key.replace("images", "depth_images")
        print(depth_key)
        if depth_key in f.keys():
            depth = f[depth_key][idx]
            depth = cv2.imdecode(np.frombuffer(depth, np.uint8), cv2.IMREAD_ANYDEPTH)
            if "high" in key:
                depth = depth.astype(np.float32) / 1000.0  # realsense 435
            else:
                depth = depth.astype(np.float32) / 10000.0  # realsense 405
            print(np.max(depth), np.min(depth))
            depth = np.where(depth < 0, 0, depth)
            depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
            depth[depth > 2] = 0.0
        else:
            print("no depth", f.keys())
            depth = None

        return img_rgb, depth


# ============================================================
# 构造模型输入 batch
# ============================================================
def batch_input(
    img,
    depth=None,
    intrinsic=None,
    resize_hw=224,
    task_text="detect the object"
):
    """
    将单张图片（+ 可选深度 + 可选内参）构造为模型推理所需的 batch 字典。

    参数:
        img: np.ndarray (H, W, 3) RGB 图像
        depth: np.ndarray (H, W) 深度图, 若为 None 则使用零张量
        intrinsic: (3,3) 相机内参 np.ndarray 或 torch.Tensor, None 则 rays 为零
        resize_hw: 重塑到的高和宽
        task_text: 任务描述文本
    返回:
        batch 字典
    """
    # 1. 读取图片 (单帧)
    img_np = np.array(img)  # (H, W, 3)
    img_th = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float() / 255.  # (1,1,3,H,W)
    B, history, C, H, W = img_th.shape
    img_th = img_th.view(B * history, C, H, W)

    # 2. 处理深度
    if depth is not None:
        depth_np = np.array(depth)
        if depth_np.dtype != np.float32:
            depth_np = depth_np.astype(np.float32)
        if depth_np.ndim == 2:
            depth_np = depth_np[None, ...]  # (1, H, W)
        depth_th = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()  # (1,1,1,H,W)
    else:
        depth_th = torch.zeros((1, 1, 1, H, W), dtype=torch.float32)
    depth_th = depth_th.view(B * history, 1, H, W)

    # 3. 处理 intrinsic
    if intrinsic is not None:
        if isinstance(intrinsic, np.ndarray):
            K = intrinsic.astype(np.float32)
        elif isinstance(intrinsic, torch.Tensor):
            K = intrinsic.cpu().numpy().astype(np.float32)
        else:
            raise ValueError("intrinsic类型必须为 np.ndarray 或 torch.Tensor")
        K_th = torch.from_numpy(K).unsqueeze(0)  # (1,3,3)
    else:
        K_th = None

    # 4. resize image, depth, intrinsic
    padded_img, valid_mask, new_intrinsic, _ = resize_with_pad(
        img_th, width=resize_hw, height=resize_hw, pad_value=0,
        mode="bilinear", intrinsic=K_th
    )

    padded_depth, valid_mask_depth, _, _ = resize_with_pad(
        depth_th, width=resize_hw, height=resize_hw, pad_value=0,
        mode="nearest", intrinsic=None
    )

    # 5. 组织 batch 格式
    padded_img = padded_img.view(B, history, C, resize_hw, resize_hw)
    padded_depth = padded_depth.view(B, history, 1, resize_hw, resize_hw)
    valid_mask = valid_mask.view(B, history, 1, resize_hw, resize_hw)

    # depth mask, 第二通道
    depth_mask = (padded_depth > 0).float()
    depths = torch.cat([padded_depth, depth_mask], dim=2)  # (B,history,2,224,224)

    # intrinsic shape变回 (B,3,3)
    if new_intrinsic is not None:
        new_intrinsic = new_intrinsic.view(B, 3, 3)
    else:
        new_intrinsic = torch.zeros((B, 3, 3), dtype=torch.float32)

    # --------- rays -----------
    if intrinsic is not None:
        ray = _generate_rays(
            height=resize_hw,
            width=resize_hw,
            K=new_intrinsic.squeeze(),  # (3,3)
            valid_mask=valid_mask.squeeze()
        )  # (3,224,224)
        rays_th = ray.unsqueeze(0).unsqueeze(0)  # (B=1,history=1,3,H,W)
    else:
        rays_th = torch.zeros((B, history, 3, resize_hw, resize_hw), dtype=torch.float32)

    batch = {
        "observation.images.top_head": padded_img,
        "observation.images.hand_left": padded_img,
        "observation.images.hand_right": padded_img,

        "task": [task_text],

        "observation.depth_priors.top_head": depths,
        "observation.depth_priors.hand_left": depths,
        "observation.depth_priors.hand_right": depths,

        "observation.rays.top_head": rays_th,
        "observation.rays.hand_left": rays_th,
        "observation.rays.hand_right": rays_th,

        "observation.images.top_head.intrinsics": new_intrinsic,
        "observation.images.hand_left.intrinsics": new_intrinsic,
        "observation.images.hand_right.intrinsics": new_intrinsic,

        "text_label": [task_text]
    }
    return batch


# ============================================================
# 主函数
# ============================================================
@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    # ===== 配置 =====
    # 请修改为你的 checkpoint 路径
    ckpt_path = ("ckpt/pi0_224_prior_weighted_sample_intra_neg_nooverlap1_bs7_gpu16_lr5e-5_decay1e-10/99999/model")
    cfg.training.batch_size = 1

    is_real_world = True

    weight_dtype = torch.bfloat16
    posevla_config = PoseVLAConfig(
        tokenizer_model_path=(cfg.model.tokenizer_model_path),
        n_action_steps=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,
        chunk_size=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,
        optimizer_lr=cfg.training.optimizer_lr,
        optimizer_betas=tuple(cfg.training.optimizer_betas),
        optimizer_eps=cfg.training.optimizer_eps,
        optimizer_weight_decay=cfg.training.optimizer_weight_decay,
        scheduler_warmup_steps=cfg.training.scheduler_warmup_steps,
        scheduler_decay_steps=cfg.training.scheduler_decay_steps,
        scheduler_decay_lr=cfg.training.scheduler_decay_lr,
        is_knowledge_insulation=cfg.training.is_knowledge_insulation,
        resize_imgs_with_padding=(cfg.dataset.image_size, cfg.dataset.image_size),
        pi05=False,
        vis_attn=True,
        add_extra_token=True,
        add_image_token=True,
        add_prior=True,
        skip_init_weights=True)

    # ===== 加载模型 =====
    policy = PoseVLAPolicy.from_pretrained(ckpt_path, local_files_only=True, config=posevla_config)
    policy = policy.eval().to(weight_dtype).cuda()

    if is_real_world:
        # ===== Real-world 推理模式 =====
        # 方式1: 从图片文件读取
        img_path = "assets/robot_rgb.png"
        depth_path = "assets/robot_depth.png"
        img = Image.open(img_path).convert('RGB')
        depth = cv2.imread(depth_path, -1) / 1000.

        # 方式2: 从 hdf5 文件读取
        # hdf5_path = hdf5_file  # 请在文件顶部设置 hdf5_file 路径
        # img, depth = parse_data(hdf5_path, "observations/images/cam_right_wrist", 1100)

        # 相机内参（根据实际相机修改）
        # Realsense D435
        # cam_K = np.array([[604.449, 0, 315.557],
        #                   [0, 603.732, 251.64],
        #                   [0, 0, 1]])

        # Realsense D405
        cam_K = np.array([[436.6096496582031, 0.0, 311.78460693359375],
                          [0.0, 435.50750732421875, 240.27752685546875],
                          [0.0, 0.0, 1.0]])

        task_text = "detect the bottle"
        batch = batch_input(img, depth=depth, intrinsic=cam_K, task_text=task_text)

        # 保存 RGB 图像和深度图为 png
        img_np = np.array(img)
        cv2.imwrite("infer_rgb.png", img_np[..., ::-1])  # RGB -> BGR 保存
        if depth is not None:
            # 同时保存原始深度值（16bit，单位mm）
            depth_mm = (depth * 1000).astype(np.uint16)
            cv2.imwrite("infer_depth_raw.png", depth_mm)
            # 保存 plasma 伪彩色深度图（用于可视化）
            depth_vis = depth.copy()
            depth_vis[depth_vis <= 0] = np.nan
            d_min, d_max = np.nanmin(depth_vis), np.nanmax(depth_vis)
            depth_norm = np.where(np.isnan(depth_vis), 0, (depth_vis - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
            depth_color[depth <= 0] = 0  # 无效区域置黑
            cv2.imwrite("infer_depth_vis.png", depth_color)
            print(f"已保存: infer_rgb.png, infer_depth_raw.png, infer_depth_vis.png")

    else:
        # ===== 数据集推理模式 =====
        omni3d_dataset = Omni3DConsumerDataset(config=cfg, tokenizer=bin_tokenizer, yaml_name="omni3d_test")
        data_vlm_collator = CollatorForDetectionDataset()
        train_vlm_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=omni3d_dataset, collate_fn=data_vlm_collator)
        batch = next(iter(train_vlm_dataloader))

    # ===== 推理 =====
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(weight_dtype).cuda()

    output_res = policy.forward_evaluate_ntp(batch)
    pred, gt = output_res["pred"], output_res["gt"]

    pred_text = pred[0]
    gt_text = gt[0]
    pred_res = decode_text_to_scene_with_tokenizer(pred_text, bin_tokenizer)
    gt_res = decode_text_to_scene_with_tokenizer(gt_text, bin_tokenizer)

    print("=" * 60)
    print("预测文本:", pred_text)
    print("=" * 60)
    print("预测结果:", pred_res)
    print("GT 结果:", gt_res)
    print("=" * 60)

    # ===== 可视化 =====
    images = {
        "image0": batch["observation.images.top_head"][0, 0],  # (3, h, w)
        "image1": batch["observation.images.hand_left"][0, 0],
        "image2": batch["observation.images.hand_right"][0, 0]
    }
    intrinsics = {
        "image0": batch["observation.images.top_head.intrinsics"][0],  # (3, 3)
        "image1": batch["observation.images.hand_left.intrinsics"][0],
        "image2": batch["observation.images.hand_right.intrinsics"][0]
    }
    depths = {
        "image0": batch["observation.depth_priors.top_head"][0, 0],  # (2, h, w)
        "image1": batch["observation.depth_priors.hand_left"][0, 0],
        "image2": batch["observation.depth_priors.hand_right"][0, 0]
    }
    rays = {
        "image0": batch["observation.rays.top_head"][0, 0],  # (3, H, W)
        "image1": batch["observation.rays.hand_left"][0, 0],
        "image2": batch["observation.rays.hand_right"][0, 0]
    }

    if is_real_world:
        visualize_2d_3d_all(images, gt_res, pred_res, intrinsics, batch["task"][0])
    else:
        visualize_2d_3d_all(images, gt_res, pred_res, intrinsics, batch["task"][0])

    visualize_views(images, depths, rays)


if __name__ == '__main__':
    main()