"""
Omni3D 数据集加载器,输出与 PI0 模型兼容的格式

基于 DetAny3D 的 pickle 格式,参考 dataset_omni6d.py 的输出结构
支持多数据集、深度图、3D 检测等功能
"""

import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from shapely.geometry import MultiPoint
from shapely.geometry import box
from utils.vis import visualize_2d_3d_all, visualize_views
from utils.mapping_token import text_to_class_attr_dict, map_3d_label_to_string
from utils.mapping_token import text_to_class_attr_dict_tokenizer, map_3d_label_to_string_tokenizer, BinTokenizer
from typing import List
import json
# Import crop helpers from the sibling Omni6D dataset module.
# NOTE: use the absolute package path (`data.ds_train.detection.dataset_omni6d`)
# rather than the bare name `dataset_omni6d`, because Hydra changes the
# working directory at runtime and bare-name imports would fail. Any
# ImportError here is fatal (the dataset cannot operate without these
# helpers), so we deliberately do NOT swallow the exception.
from data.ds_train.detection.dataset_omni6d import (
    _crop_resize_with_pad,
    _resize_with_pad,
    _generate_rays,
    CropParamManager,
    _intrinsics_to_matrix,
    _generate_coord_grid,
    _depth_to_pcl,
)

#################### Pow3r utils ####################
# from pow3r.datasets.utils.modalities import _gen_rays
def _gen_rays(true_shape, K):
    assert isinstance(K, np.ndarray), "K must be a numpy array"
    H, W = true_shape
    pix = np.mgrid[:W, :H].T.astype(np.float32)
    rays = compute_rays(pix, K) # H,W,3
    return rays.transpose(2,0,1) # 3,H,W

# from pow3r.utils.geometry import compute_rays
def compute_rays(pix, K):
    return geotrf(inv(K), pix, ncol=3)

# from dust3r.utils.geometry import geotrf,inv
def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

            if pts.shape[-1] + 1 == Trf.shape[-1]:
                Trf = Trf.swapaxes(-1, -2)  # transpose Trf
                pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
            elif pts.shape[-1] == Trf.shape[-1]:
                Trf = Trf.swapaxes(-1, -2)  # transpose Trf
                pts = pts @ Trf
            else:
                pts = Trf @ pts.T
                if pts.ndim >= 2:
                    pts = pts.swapaxes(-1, -2)

            if norm:
                pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
                if norm != 1:
                    pts *= norm

            res = pts[..., :ncol].reshape(*output_reshape, ncol)
            return res
def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


############################################################
# from pow3r.pow3r.datasets.utils.modalities import gen_sparse_depth_hjs
def gen_sparse_depth_hjs(
    true_shape,
    valid_mask: torch.Tensor,
    depthmap: torch.Tensor,
    n_pts_min,
    n_pts_max,
    rng=None,
    norm=True
):
    depthmap=depthmap.squeeze()
    H, W = true_shape
    # n_pts_max = n_pts_max or H * W
    n_pts_max = n_pts_max


    # 保证 rng 存在 (PyTorch generator)
    if rng is None:
        rng = torch.Generator(device=valid_mask.device)

    # 有效像素位置
    valid_pixels = valid_mask[0].reshape(-1)  # [H*W]
    valid_indices = torch.nonzero(valid_pixels, as_tuple=False).squeeze(1)  # [N]

    num_valid_pixels = valid_indices.numel()

    # 取 min/max
    n_pts_min = min(n_pts_min, num_valid_pixels)
    n_pts_max = min(n_pts_max, num_valid_pixels)

    # 从 [n_pts_min, n_pts_max] 中采样点数（对数分布的话，你原来用 randint_logspace，可以保留）
    n_pts = torch.randint(n_pts_min, n_pts_max + 1, (1,), generator=rng).item()

    # 随机选择像素索引
    perm = torch.randperm(num_valid_pixels, generator=rng, device=valid_mask.device)
    unmasked_indices = valid_indices[perm[:n_pts]]

    # 构造 mask
    mask = torch.zeros(H * W, dtype=torch.float32, device=valid_mask.device)
    if n_pts_max>0:
        mask[unmasked_indices] = 1.0
    mask = mask.view(H, W)


    # 稀疏深度
    sparse_depth = depthmap * mask

    # 归一化
    # if norm and n_pts > 0:
    #     mean_val = sparse_depth[mask > 0].mean()
    #     sparse_depth = sparse_depth / (mean_val + 1e-8)

        # sparse_depth = sparse_depth / (sparse_depth.max() + 1e-8)

    return torch.stack((sparse_depth, mask))  # (2, H, W)

def compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rotation_matrix=None):
    '''
    output_corners_order:
        (3) +---------+. (2)
            | ` .     |  ` .
            | (7) +---+-----+ (6)
            |     |   |     |
        (4) +-----+---+. (1)|
            ` .   |     ` . |
            (8) ` +---------+ (5)

    '''
    corners = np.array([
        [w / 2, h / 2, l / 2],
        [w / 2, -h / 2, l / 2],
        [-w / 2, -h / 2, l / 2],
        [-w / 2, h / 2, l / 2],
        [w / 2, h / 2, -l / 2],
        [w / 2, -h / 2, -l / 2],
        [-w / 2, -h / 2, -l / 2],
        [-w / 2, h / 2, -l / 2]
    ])

    fore_plane_center = np.array([
        [0, 0, l / 2]
    ])
    if rotation_matrix is not None:
        R = rotation_matrix
    else:
        R = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
    corners_rotated = corners @ R.T

    corners_translated = corners_rotated + np.array([x, y, z])

    fore_plane_center = fore_plane_center @ R.T + np.array([x, y, z])

    return corners_translated, fore_plane_center


def get_2d_coord_np(width, height, fmt="CHW"):
    """生成2D坐标网格"""
    x = np.linspace(0, width - 1, width, dtype=np.float32)
    y = np.linspace(0, height - 1, height, dtype=np.float32)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy


def depth_to_pcl_tensor(depth, K, xymap, valid):
    """将深度图转换为3D点云 (H, W, 3)"""
    with torch.no_grad():
        if isinstance(xymap, np.ndarray):
            xymap = torch.from_numpy(xymap).to(device=depth.device, dtype=depth.dtype)
        elif not isinstance(xymap, torch.Tensor):
            xymap = torch.tensor(xymap, device=depth.device, dtype=depth.dtype)

        xymap = xymap.to(device=depth.device, dtype=depth.dtype)
        K = K.reshape(-1)
        cx, cy, fx, fy = K[2], K[5], K[0], K[4]
        valid = valid.bool()

        x_map = xymap[0]
        y_map = xymap[1]

        real_x = (x_map - cx) * depth / fx
        real_y = (y_map - cy) * depth / fy
        real_z = depth

        pcl = torch.stack([real_x, real_y, real_z], dim=-1)
        pcl[~valid] = 0.0

    return pcl


def project_to_image(points_3d, K):
    """
    将3D点投影到图像平面

    Args:
        points_3d: 3D点坐标 (N, 3) numpy array
        K: 相机内参矩阵 (3, 3) numpy array

    Returns:
        points_2d: 2D点坐标 (N, 2) numpy array
    """
    points_3d_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    K_extended = np.hstack((K, np.zeros((3, 1))))
    points_2d = K_extended @ points_3d_homo.T

    new_points_2d = points_2d[:2, :] / points_2d[2:3, :]
    new_points_2d[0, :] = np.sign(points_2d[0, :]) * np.abs(points_2d[0, :]) / np.abs(points_2d[2, :])
    new_points_2d[1, :] = np.sign(points_2d[1, :]) * np.abs(points_2d[1, :]) / np.abs(points_2d[2, :])
    return new_points_2d.T


@dataclass
class _SampleEntry:
    """Omni3D 样本的索引信息（每个物体类别一个entry）"""
    dataset_idx: int      # 数据集索引
    sample_idx: int       # 样本在该数据集中的索引
    category_id: int      # 物体类别ID


class Omni3DConsumerDataset(Dataset):
    """
    Omni3D 数据集,输出与 PI0 模型兼容的格式

    从 pickle 文件加载数据,支持多数据集训练
    参考 DetAny3D 的数据加载方式,输出 dataset_omni6d.py 的格式
    """

    def __init__(self, config: Any, tokenizer: BinTokenizer, yaml_name="omni3d_train") -> None:
        super().__init__()
        self.config = config

        # 检查配置
        if not hasattr(config.dataset_omni3d, yaml_name):
            raise ValueError("config.dataset_omni3d 未配置")

        # omni3d_cfg = config.dataset_omni3d
        omni3d_cfg = getattr(config.dataset_omni3d, yaml_name)

        self.omni3d_cfg = omni3d_cfg
        self.enable_random_crop=bool(omni3d_cfg.get("enable_random_crop",True))

        # 图像参数
        self.image_size = int(omni3d_cfg.get("image_size", 224))
        self.vggt_image_size = int(omni3d_cfg.get("vggt_image_size", 518))
        self.img_history_size = int(omni3d_cfg.get("img_history_size", config.dataset.img_history_size))

        # 多相机配置（参考 dataset_omni6d.py）
        default_camera_configs = [
            {"name": "top_head", "crop_scale": (0.90, 0.90), "aspect_ratio": (1.33, 1.33)},
            {"name": "hand_left", "crop_scale": (0.25, 0.25), "aspect_ratio": (1.33, 1.33)},
            {"name": "hand_right", "crop_scale": (0.55, 0.55), "aspect_ratio": (1.33, 1.33)},
        ]
        self.camera_configs = omni3d_cfg.get("camera_configs", default_camera_configs)
        self.num_cameras = len(self.camera_configs)

        # 数据集模式
        self.mode = omni3d_cfg.get("mode", "train")

        self.sample_retries = int(omni3d_cfg.get("max_retries", 10))
        # 获取数据集配置
        # 优先从 dataset_omni3d 配置中读取，如果没有则尝试从全局 dataset 读取
        if hasattr(omni3d_cfg, 'dataset'):
            # 从 dataset_omni3d.dataset 读取
            if self.mode == 'train':
                if hasattr(omni3d_cfg.dataset, 'train'):
                    self.dataset_dict = omni3d_cfg.dataset.train
                else:
                    raise ValueError(f"config.dataset_omni3d.dataset.train 未配置")
            elif self.mode == 'val':
                if hasattr(omni3d_cfg.dataset, 'val'):
                    self.dataset_dict = omni3d_cfg.dataset.val
                else:
                    raise ValueError(f"config.dataset_omni3d.dataset.val 未配置")
            elif self.mode == 'test':
                if hasattr(omni3d_cfg.dataset, 'test'):
                    self.dataset_dict = omni3d_cfg.dataset.test
                else:
                    raise ValueError(f"config.dataset_omni3d.dataset.test 未配置")
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
        else:
            raise ValueError(
                "数据集配置未找到。请在 config/dataset_omni3d/omni3d.yaml 中添加:\n"
                "dataset:\n"
                "  train:\n"
                "    your_dataset:\n"
                "      pkl_path: '/path/to/train.pkl'\n"
                "  val:\n"
                "    your_val_dataset:\n"
                "      pkl_path: '/path/to/val.pkl'"
            )

        # 加载数据集
        self.dataset_name_list = []
        self.pkl_path_list = []
        self.len_idx = []
        self.pkl_list = []

        # 调试模式
        self.debug_mode = bool(omni3d_cfg.get("debug", False))

        # 路径替换配置（用于跨机器迁移 pkl 数据）
        self.old_data_root = omni3d_cfg.get("old_data_root", None)
        self.data_root = omni3d_cfg.get("data_root", None)
        if self.old_data_root and self.data_root:
            print(f"[Omni3DConsumerDataset] 路径替换: {self.old_data_root} → {self.data_root}")

        # 加载数据集
        for dataset_name in self.dataset_dict.keys():
            dataset_info = self.dataset_dict[dataset_name]
            self._load_single_dataset(dataset_name, dataset_info)

        # 构建累积索引
        self.idx_cum = np.cumsum(self.len_idx)

        # 构建扩展的索引（每个物体类别一个entry）
        print(f"[Omni3DConsumerDataset] 构建扩展索引...")
        self.entries = []
        total_samples = 0
        total_entries = 0

        for dataset_idx, pkl_data in enumerate(self.pkl_list):
            dataset_name = self.dataset_name_list[dataset_idx]
            dataset_samples = len(pkl_data)
            dataset_entries = 0

            for sample_idx, instance in enumerate(pkl_data):
                category_groups = instance.get('category_groups', {})

                # 为每个类别创建一个 entry
                for category_id in category_groups.keys():
                    self.entries.append(_SampleEntry(
                        dataset_idx=dataset_idx,
                        sample_idx=sample_idx,
                        category_id=category_id
                    ))
                    dataset_entries += 1

            total_samples += dataset_samples
            total_entries += dataset_entries
            print(f"  - {dataset_name}: {dataset_samples} 个样本 → {dataset_entries} 个 entries")

        print(f"  - 总计: {total_samples} 个样本 → {total_entries} 个 entries")

        # 坐标缓存
        self.coord_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        # Crop参数管理器（与 dataset_omni6d.py 一致）
        if CropParamManager is not None:
            self.crop_param_manager = CropParamManager(enable_random=self.enable_random_crop)
        else:
            self.crop_param_manager = None
            print("Warning: CropParamManager 未导入，将使用简化的crop处理")

        print(f"[Omni3DConsumerDataset] 初始化完成")
        print(f"  - 模式: {self.mode}")
        print(f"  - 数据集数量: {len(self.dataset_name_list)}")
        print(f"  - 总样本数: {len(self)}")
        print(f"  - 图像尺寸: {self.image_size}")
        print(f"  - VGGT 图像尺寸: {self.vggt_image_size}")

        self.tokenizer = tokenizer

        self.id2name_dict = omni3d_cfg.id2name  # 这是个 dict
        category_file = 'cache/omni3d_category_list.json'
        if not os.path.exists(category_file):
            category_all = set()
            for v in self.id2name_dict.values():
                # 预处理每个value
                category_all.add(v.replace("_", " ").lower())
            self.category_all_list = list(category_all)
            self.category_all_set = category_all

            with open(category_file, 'w', encoding='utf-8') as fp:
                json.dump(self.category_all_list, fp, ensure_ascii=False, indent=4)
            print(f'Category list saved to omni3d_category_list.json, Total {len(self.category_all_list)} categories')
        else:
            with open(category_file, 'r', encoding='utf-8') as fp:
                self.category_all_list = json.load(fp)
            self.category_all_set = set(self.category_all_list)  # 如果后面需要set形式也可以再加上这一句

        category_files = [
            'cache/omni3d_category_list.json',
            'cache/omni6d_category_list.json',
            'cache/clutter_category_list.json',
            'cache/bop_category_list.json'
        ]

        category_all = set()
        for filename in category_files:
            with open(filename, 'r', encoding='utf-8') as fp:
                categories = json.load(fp)
                # 预处理每个类别名：下划线转空格、小写
                for c in categories:
                    category_all.add(c.replace('_', ' ').lower())

        self.category_full_set = category_all

        print(f' Total {len(self.category_all_set)} categories in this dataset')
        print(f' Total {len(self.category_full_set)} categories')

    def _load_single_dataset(self, dataset_name, dataset_info):
        """加载单个数据集"""
        self.dataset_name_list.append(dataset_name)
        self.pkl_path_list.append(dataset_info.pkl_path)

        # 检查 pkl 文件是否存在
        if not os.path.exists(dataset_info.pkl_path):
            raise FileNotFoundError(f"数据集 pkl 文件不存在: {dataset_info.pkl_path}")

        with open(dataset_info.pkl_path, 'rb') as f:
            tmp_pkl = pickle.load(f)[:]
            self.pkl_list.append(tmp_pkl)

        num_samples = len(tmp_pkl)
        self.len_idx.append(num_samples)

        # 验证数据集质量（检查前几个样本）
        if self.debug_mode and num_samples > 0:
            self._validate_dataset_samples(tmp_pkl, dataset_name, max_check=min(5, num_samples))

        print(f"  - {dataset_name}: {num_samples} 个样本")

    def _validate_dataset_samples(self, pkl_data, dataset_name, max_check=5):
        """验证数据集样本的质量（调试模式）"""
        print(f"\n    [调试] 验证数据集 {dataset_name} 的前 {max_check} 个样本...")
        issues = []

        for i in range(max_check):
            instance = pkl_data[i]

            # 检查图像路径
            if 'img_path' not in instance:
                issues.append(f"样本 {i}: 缺少 img_path 字段")
            elif not os.path.exists(instance['img_path']):
                issues.append(f"样本 {i}: 图像文件不存在: {instance['img_path']}")

            # 检查相机内参
            if 'K' not in instance:
                issues.append(f"样本 {i}: 缺少相机内参 K")

            # 检查标注数据
            if 'category_groups' not in instance:
                issues.append(f"样本 {i}: 缺少 category_groups 字段")
            elif len(instance['category_groups']) == 0:
                issues.append(f"样本 {i}: category_groups 为空（无标注）")

        if issues:
            print(f"    ⚠️ 发现 {len(issues)} 个问题:")
            for issue in issues[:10]:  # 最多显示10个问题
                print(f"      - {issue}")
            if len(issues) > 10:
                print(f"      ... (还有 {len(issues) - 10} 个问题未显示)")
        else:
            print(f"    ✓ 验证通过")

    def _get_relative_index(self, index):
        """从 entry 获取数据集索引、样本索引和类别ID"""
        entry = self.entries[index]
        return entry.dataset_idx, entry.sample_idx, entry.category_id

    def __len__(self) -> int:
        """返回扩展索引的总数（每个物体类别一个entry）"""
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取单个样本,带重试机制"""
        original_index = index
        last_error = None

        for retry in range(self.sample_retries):
            try:
                dataset_idx, sample_idx, category_id = self._get_relative_index(index)
                pkl_now = self.pkl_list[dataset_idx]
                dataset_name = self.dataset_name_list[dataset_idx]
                instance = pkl_now[sample_idx]

                sample = self._load_sample(instance, dataset_name, category_id)
                if sample is not None and sum(len(bbox) for bbox in sample['bbox_2d'])>0:
                    return sample
                else:
                    last_error = f"_load_sample 返回 None (可能原因: 文件不存在或标注为空)"
            except Exception as e:
                last_error = str(e)
                if self.debug_mode:
                    import traceback
                    print(f"⚠️ 样本加载失败 (尝试 {retry+1}/{self.sample_retries})")
                    print(f"   索引: {index} (dataset={dataset_idx}, sample={sample_idx}, category={category_id})")
                    print(f"   错误: {e}")
                    if retry == 0:  # 只在第一次失败时打印完整堆栈
                        traceback.print_exc()

            # 重试时随机选择新的索引
            index = random.randint(0, len(self) - 1)

        # 所有重试都失败,抛出详细的错误信息
        error_msg = (
            f"⚠️ 无法加载任何有效样本！\n"
            f"  原始索引: {original_index}\n"
            f"  数据集长度: {len(self)}\n"
            f"  最后一次错误: {last_error}\n"
            f"  已尝试 {self.sample_retries} 次随机采样\n"
            f"  建议检查:\n"
            f"    1. 数据集路径是否正确\n"
            f"    2. 图像文件是否存在\n"
            f"    3. pkl 文件中的 img_path 字段是否正确\n"
            f"    4. category_groups 是否为空\n"
            f"  提示: 设置 debug=True 查看详细错误信息"
        )
        raise RuntimeError(error_msg)

    def _load_sample(self, instance: Dict, dataset_name: str, category_id: int) -> Optional[Dict[str, Any]]:
        """加载单个样本的具体实现（支持多视角相机）

        Args:
            instance: 样本数据字典
            dataset_name: 数据集名称
            category_id: 预先确定的物体类别ID（从 entry 中获取）
        """
        # 加载图像
        img_path = instance['img_path']
        # 路径替换（跨机器迁移）
        if self.old_data_root and self.data_root:
            img_path = img_path.replace(self.old_data_root, self.data_root)
        if not os.path.exists(img_path):
            return None

        rgb = cv2.imread(img_path)
        if rgb is None:
            return None

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 加载深度图
        depth_path = instance.get('depth_path', None)
        # 深度图路径也需要替换
        if depth_path and self.old_data_root and self.data_root:
            depth_path = depth_path.replace(self.old_data_root, self.data_root)
        depth = self._load_depth(depth_path, dataset_name, rgb, img_path)

        # 加载相机内参（原始内参，numpy 格式）
        K_orig = instance['K'].astype(np.float32)
        if K_orig.ndim == 3:
            K_orig = K_orig.squeeze(0)
        if K_orig.shape != (3, 3):
            if self.debug_mode:
                print(f"⚠️ 警告: K 矩阵形状异常 {K_orig.shape}，预期 [3, 3]，尝试 reshape")
            K_orig = K_orig.reshape(3, 3)

        # 提取原始内参
        fx_orig = K_orig[0, 0]
        fy_orig = K_orig[1, 1]
        cx_orig = K_orig[0, 2]
        cy_orig = K_orig[1, 2]

        # ============================================================
        # 数据处理流程 (多摄像头不同crop):
        # 1. 为N个摄像头生成不同crop的图像（根据配置）
        # 2. 原始RGB/depth → crop + resize + pad → 518x518 (vggt)
        # 3. vggt_518 → resize + pad → 224x224
        # 4. 基于224分辨率生成所有标注: bbox, rays, K等
        # ============================================================

        # 使用从配置读取的相机设置
        camera_configs = self.camera_configs

        # 为每个摄像头生成不同crop的图像
        camera_images_224 = []
        camera_images_518 = []
        camera_depths_224 = []
        camera_valid_masks_224 = []
        camera_intrinsics = []
        camera_transform_params = []  # 保存每个相机的变换参数，用于转换 2D bbox

        depth_mask = (depth > 0).astype(np.uint8)

        # 获取样本索引（用于 CropParamManager）
        # 由于 Omni3D 数据是从 pkl 加载的，我们使用 img_path 的哈希作为 sample_idx
        sample_idx = hash(img_path) % 100000

        for cam_idx, cam_config in enumerate(camera_configs):
            if self.crop_param_manager is not None and _crop_resize_with_pad is not None:
                # 使用crop参数管理器获取crop参数（确保一致性）
                crop_params_518 = self.crop_param_manager.get_or_create_params(
                    sample_idx=sample_idx,
                    cam_idx=cam_idx,
                    size_key="518",
                    orig_width=rgb.shape[1],
                    orig_height=rgb.shape[0],
                    crop_scale=cam_config["crop_scale"],
                    aspect_ratio=cam_config["aspect_ratio"]
                )

                # 提取crop参数用于预设
                preset_crop_params_518 = (
                    crop_params_518.crop_width,
                    crop_params_518.crop_height,
                    crop_params_518.crop_left,
                    crop_params_518.crop_top
                )

                # 步骤1: 原始数据 → 518x518 (crop + resize + pad)
                rgb_518_cam, valid_mask_518_cam, resize_ratio_518_cam, pad_l_518_cam, pad_t_518_cam, crop_l_518_cam, crop_t_518_cam = _crop_resize_with_pad(
                    rgb, self.vggt_image_size, self.vggt_image_size,
                    pad_value=0, interpolation=cv2.INTER_LINEAR,
                    crop_scale=cam_config["crop_scale"],
                    aspect_ratio=cam_config["aspect_ratio"],
                    debug=True,  # 使用中心crop
                    preset_crop_params=preset_crop_params_518
                )

                # 深度图也要用相同的crop参数
                depth_518_cam, _, _, _, _, _, _ = _crop_resize_with_pad(
                    depth, self.vggt_image_size, self.vggt_image_size,
                    pad_value=0.0, interpolation=cv2.INTER_NEAREST,
                    crop_scale=cam_config["crop_scale"],
                    aspect_ratio=cam_config["aspect_ratio"],
                    debug=True,
                    preset_crop_params=preset_crop_params_518
                )
                depth_mask_518_cam, _, _, _, _, _, _ = _crop_resize_with_pad(
                    depth_mask, self.vggt_image_size, self.vggt_image_size,
                    pad_value=0, interpolation=cv2.INTER_NEAREST,
                    crop_scale=cam_config["crop_scale"],
                    aspect_ratio=cam_config["aspect_ratio"],
                    debug=True,
                    preset_crop_params=preset_crop_params_518
                )

                # 步骤2: 518 → 224 (resize + pad)
                rgb_224_cam, _, resize_ratio_224_cam, pad_l_224_cam, pad_t_224_cam = _resize_with_pad(
                    rgb_518_cam, self.image_size, self.image_size, pad_value=0, interpolation=cv2.INTER_LINEAR
                )
                valid_mask_224_cam, _, _, _, _ = _resize_with_pad(
                    valid_mask_518_cam, self.image_size, self.image_size, pad_value=0, interpolation=cv2.INTER_NEAREST
                )
                depth_224_cam, _, _, _, _ = _resize_with_pad(
                    depth_518_cam, self.image_size, self.image_size, pad_value=0.0, interpolation=cv2.INTER_NEAREST
                )
                depth_mask_224_cam, _, _, _, _ = _resize_with_pad(
                    depth_mask_518_cam, self.image_size, self.image_size, pad_value=0, interpolation=cv2.INTER_NEAREST
                )

                # 保存变换参数（用于转换 2D bbox）
                transform_params = {
                    'crop_left': crop_l_518_cam,
                    'crop_top': crop_t_518_cam,
                    'resize_ratio_518': resize_ratio_518_cam,
                    'pad_left_518': pad_l_518_cam,
                    'pad_top_518': pad_t_518_cam,
                    'resize_ratio_224': resize_ratio_224_cam,
                    'pad_left_224': pad_l_224_cam,
                    'pad_top_224': pad_t_224_cam,
                }
                camera_transform_params.append(transform_params)

                # 调整相机内参 (原始 → crop → 518 → 224)
                # Step 1: crop偏移
                fx_after_crop = fx_orig
                fy_after_crop = fy_orig
                cx_after_crop = cx_orig - crop_l_518_cam
                cy_after_crop = cy_orig - crop_t_518_cam

                # Step 2: resize + pad to 518
                fx_518_cam = fx_after_crop / resize_ratio_518_cam
                fy_518_cam = fy_after_crop / resize_ratio_518_cam
                cx_518_cam = cx_after_crop / resize_ratio_518_cam + pad_l_518_cam
                cy_518_cam = cy_after_crop / resize_ratio_518_cam + pad_t_518_cam

                # Step 3: resize + pad to 224
                fx_cam = fx_518_cam / resize_ratio_224_cam
                fy_cam = fy_518_cam / resize_ratio_224_cam
                cx_cam = cx_518_cam / resize_ratio_224_cam + pad_l_224_cam
                cy_cam = cy_518_cam / resize_ratio_224_cam + pad_t_224_cam

                camera_images_224.append(rgb_224_cam)
                camera_images_518.append(rgb_518_cam)
                camera_depths_224.append(depth_224_cam)
                camera_valid_masks_224.append(valid_mask_224_cam)
                camera_intrinsics.append([fx_cam, fy_cam, cx_cam, cy_cam])
            else:
                # 如果没有导入crop函数，使用简化的处理方式（所有相机使用相同数据）
                # 这部分作为fallback，保持原有的简单处理
                print(f"Warning: 使用简化的crop处理，所有相机将共享相同的crop")
                # ... 这里可以添加fallback代码 ...
                return None  # 暂时返回 None

        # 将每个摄像头的518图像转换为tensor列表
        images_vggt_list = [
            torch.from_numpy(rgb_518).permute(2, 0, 1).contiguous().float() / 255.0
            for rgb_518 in camera_images_518
        ]

        # 使用top_head的数据作为主要数据
        rgb_224 = camera_images_224[0]
        depth_224 = camera_depths_224[0]
        depth_mask_224 = (camera_depths_224[0] > 0).astype(np.uint8)
        valid_mask_224 = camera_valid_masks_224[0]

        # 使用top_head的内参
        fx, fy, cx, cy = camera_intrinsics[0]

        # ============================================================
        # 处理 3D 检测标注
        # ============================================================

        # 提取目标信息
        category_groups = instance.get('category_groups', {})
        if len(category_groups) == 0:
            return None

        # 使用预先确定的类别ID（从索引中获取）
        if category_id not in category_groups:
            return None
        objects = category_groups[category_id]

        # 获取类别名称和标签
        class_name = objects[0].get('label_name', 'unknown') if len(objects) > 0 else 'unknown'
        class_label = objects[0].get('label', -1) if len(objects) > 0 else -1

        # 步骤3: 为每个摄像头独立处理2D bbox和3D标注
        # 为每个摄像头生成对应crop下的bbox、bbox_side_len、pose
        camera_bboxes = []  # 每个摄像头的bbox列表
        camera_bbox_side_len = []  # 每个摄像头的3D尺寸列表
        camera_pose = []  # 每个摄像头的位姿列表
        camera_class_labels = []  # 每个摄像头的类别标签列表

        for cam_idx, cam_config in enumerate(camera_configs):
            bbox_tensor_list = []
            bbox_side_len_list = []
            pose_list_cam = []
            class_label_list = []

            # 获取当前相机的内参（4元素格式）
            cam_fx, cam_fy, cam_cx, cam_cy = camera_intrinsics[cam_idx]

            # 构建3x3 K矩阵（numpy格式，用于 project_to_image）
            cam_K_np = np.eye(3, dtype=np.float32)
            cam_K_np[0, 0], cam_K_np[1, 1] = cam_fx, cam_fy
            cam_K_np[0, 2], cam_K_np[1, 2] = cam_cx, cam_cy

            cam_depth_224 = camera_depths_224[cam_idx]
            cam_valid_mask_224 = camera_valid_masks_224[cam_idx]
            cam_images_224 = camera_images_224[cam_idx]

            # 获取当前相机的变换参数
            transform_params = camera_transform_params[cam_idx]

            img_size = tuple(cam_images_224.shape[:2])  # (height, width) 顺序

            for obj in objects:
                # 提取3D bbox参数
                x, y, z, w, h, l, _yaw = obj['3d_bbox']
                pose = obj['rotation_pose']
                vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, _yaw, pose)

                # 使用正确的K矩阵投影
                vertices_2d = project_to_image(vertices_3d, cam_K_np)

                # ============================================================
                # 1. 计算原始的2D bbox（不受画布约束，直接从投影点计算AABB）
                # ============================================================
                bbox_2d_unconstrained = np.array([
                    np.min(vertices_2d[:, 0]),  # min_x
                    np.min(vertices_2d[:, 1]),  # min_y
                    np.max(vertices_2d[:, 0]),  # max_x
                    np.max(vertices_2d[:, 1])  # max_y
                ])

                # ============================================================
                # 2. 计算受画布约束的2D bbox（与图像边界求交集）
                # ============================================================
                # 使用 Shapely 库计算投影点的凸包（convex hull）
                # MultiPoint: 创建多点几何对象
                # convex_hull: 计算包含所有点的最小凸多边形
                polygon_from_2d_box = MultiPoint(vertices_2d).convex_hull

                # 创建图像画布的边界框（用于裁剪超出图像的部分）
                img_canvas = box(0, 0, img_size[1], img_size[0])  # (x_min, y_min, x_max, y_max)

                # 检查投影的多边形是否与图像画布相交
                if polygon_from_2d_box.intersects(img_canvas):
                    # 计算多边形与图像画布的交集
                    img_intersection = polygon_from_2d_box.intersection(img_canvas)

                    # 获取交集多边形的顶点坐标
                    intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

                    # 计算交集多边形的外接矩形（axis-aligned bounding box）
                    min_x = min(intersection_coords[:, 0])  # 最小 x 坐标
                    min_y = min(intersection_coords[:, 1])  # 最小 y 坐标
                    max_x = max(intersection_coords[:, 0])  # 最大 x 坐标
                    max_y = max(intersection_coords[:, 1])  # 最大 y 坐标
                else:
                    # 如果投影完全在图像外，跳过这个目标
                    continue
                    # min_x = 0  # 最小 x 坐标
                    # min_y = 0  # 最小 y 坐标
                    # max_x = 0  # 最大 x 坐标
                    # max_y = 0  # 最大 y 坐标
                # 构建2D边界框 [x1, y1, x2, y2]
                bbox_2d_polygon = [min_x, min_y, max_x, max_y]

                # 转换为张量（使用 float32 保持精度）
                bbox_2d_constrained_tensor = torch.tensor(bbox_2d_polygon, dtype=torch.float32)
                bbox_2d_unconstrained_tensor = torch.from_numpy(bbox_2d_unconstrained).float()

                # 调用 filter_objects 进行过滤
                # 使用原始bbox（不受画布约束）进行过滤判断，更准确地评估对象质量
                if not self.filter_objects(
                        obj=obj,
                        bbox_2d_tensor=bbox_2d_unconstrained_tensor,
                        before_pad_size=img_size,  # (height, width) = (224, 224)
                        K=cam_K_np,  # 使用正确的 3x3 numpy 矩阵
                        dataset_name=dataset_name,
                        min_size=10,
                        min_height_ratio=0.05,
                        min_depth=0.01,
                        max_depth=10,
                        image_mask=cam_valid_mask_224  # 使用当前相机的valid mask
                ):
                    # 如果过滤失败，跳过此对象
                    continue

                # 使用受画布约束的bbox作为最终的2D标注（用于训练）
                # 这样可以确保bbox坐标在图像范围内
                bbox_tensor_list.append(bbox_2d_constrained_tensor)

                # 转换旋转矩阵为四元数
                if 'rotation_pose' in obj:
                    rotation_matrix = obj['rotation_pose']
                    rot = R.from_matrix(rotation_matrix)
                    quat_xyzw = rot.as_quat()
                    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                else:
                    # 如果没有旋转矩阵,使用单位四元数
                    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

                # 组合姿态
                pose = np.concatenate([quat_wxyz, np.array([x, y, z])])
                pose_list_cam.append(pose)

                # 3D尺寸
                bbox_side_len = np.array([w, h, l])
                bbox_side_len_list.append(bbox_side_len)
                class_label_list.append(class_label)

            # 保存当前摄像头的bbox和3D标注
            camera_bboxes.append(bbox_tensor_list)
            if bbox_side_len_list:
                camera_bbox_side_len.append(torch.tensor(np.array(bbox_side_len_list), dtype=torch.float32))
                camera_pose.append(torch.tensor(np.array(pose_list_cam), dtype=torch.float32))
                # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow.
                # camera_bbox_side_len.append(torch.tensor(bbox_side_len_list, dtype=torch.float32))
                # camera_pose.append(torch.tensor(pose_list_cam, dtype=torch.float32))
            else:
                camera_bbox_side_len.append(torch.empty(0, 3, dtype=torch.float32))
                camera_pose.append(torch.empty(0, 7, dtype=torch.float32))
            camera_class_labels.append(class_label_list)

        # 使用top_head (第0个摄像头) 的标注作为主要标注
        bbox_side_len_tensor = camera_bbox_side_len[0] if camera_bbox_side_len[0].numel() > 0 else torch.zeros((0, 3), dtype=torch.float32)
        pose_tensor = camera_pose[0] if camera_pose[0].numel() > 0 else torch.zeros((0, 7), dtype=torch.float32)

        # ============================================================
        # 为每个摄像头生成3D感知数据（rays, pcl, know_depth等）
        # ============================================================

        camera_rays_list = []
        camera_pcl_list = []
        camera_know_depth_list = []
        camera_depth_tensors = []
        camera_depth_mask_tensors = []
        camera_K_list = []

        # 用top_head的深度来计算avg_scale（统一尺度）
        top_depth_tensor = torch.from_numpy(camera_depths_224[0]).unsqueeze(0).float()
        top_depth_mask_tensor = torch.from_numpy((camera_depths_224[0] > 0).astype(np.uint8)).unsqueeze(0)
        top_fx, top_fy, top_cx, top_cy = camera_intrinsics[0]
        top_K = torch.from_numpy(_intrinsics_to_matrix(top_fx, top_fy, top_cx, top_cy)) if _intrinsics_to_matrix is not None else torch.eye(3, dtype=torch.float32)

        coord_map = self.coord_cache.get((self.image_size, self.image_size))
        if coord_map is None:
            if _generate_coord_grid is not None:
                coord_map = _generate_coord_grid(self.image_size, self.image_size)
                self.coord_cache[(self.image_size, self.image_size)] = coord_map
            else:
                coord_map = torch.from_numpy(get_2d_coord_np(self.image_size, self.image_size))
        coord_map = coord_map.to(torch.device("cpu"))

        # 计算top_head的点云以获取avg_scale
        if _depth_to_pcl is not None:
            top_pcl = _depth_to_pcl(top_depth_tensor.squeeze(0), top_K, coord_map, top_depth_mask_tensor.squeeze(0))
        else:
            top_pcl = depth_to_pcl_tensor(top_depth_tensor.squeeze(0), top_K, coord_map, top_depth_mask_tensor.squeeze(0))

        if top_depth_mask_tensor.sum() > 0:
            dist = torch.linalg.norm(top_pcl, dim=-1)
            # avg_scale = dist[top_depth_mask_tensor.squeeze(0) > 0].mean().clamp(min=1e-3, max=1e3)
            if self.config.max_norm_6d_dataset:
                avg_scale = dist[top_depth_mask_tensor.squeeze(0) > 0].max().clamp(min=1e-3, max=1e3)
            else:
                avg_scale = torch.tensor(1.0)
        else:
            avg_scale = torch.tensor(1.0)

        # 为每个摄像头生成对应的3D数据
        for cam_idx in range(len(camera_images_224)):
            cam_depth_224 = camera_depths_224[cam_idx]
            cam_valid_mask_224 = camera_valid_masks_224[cam_idx]
            cam_fx, cam_fy, cam_cx, cam_cy = camera_intrinsics[cam_idx]

            # 构建K矩阵
            if _intrinsics_to_matrix is not None:
                cam_K = torch.from_numpy(_intrinsics_to_matrix(cam_fx, cam_fy, cam_cx, cam_cy))
            else:
                cam_K = torch.eye(3, dtype=torch.float32)
                cam_K[0, 0], cam_K[1, 1] = cam_fx, cam_fy
                cam_K[0, 2], cam_K[1, 2] = cam_cx, cam_cy
            camera_K_list.append(cam_K)

            # 深度张量
            cam_depth_tensor = torch.from_numpy(cam_depth_224).unsqueeze(0).float()
            cam_depth_mask_tensor = torch.from_numpy((cam_depth_224 > 0).astype(np.uint8)).unsqueeze(0)
            cam_img_mask_tensor = torch.from_numpy(cam_valid_mask_224).unsqueeze(0)

            # 生成点云
            if _depth_to_pcl is not None:
                cam_pcl = _depth_to_pcl(cam_depth_tensor.squeeze(0), cam_K, coord_map, cam_depth_mask_tensor.squeeze(0))
            else:
                cam_pcl = depth_to_pcl_tensor(cam_depth_tensor.squeeze(0), cam_K, coord_map, cam_depth_mask_tensor.squeeze(0))

            # 生成rays
            if _generate_rays is not None:
                cam_rays = _generate_rays(self.image_size, self.image_size, cam_K, cam_img_mask_tensor.squeeze(0).bool())
            else:
                cam_rays = torch.as_tensor(np.array(_gen_rays((self.image_size, self.image_size), np.array(cam_K))),
                                          dtype=torch.float32).contiguous()
                cam_rays = cam_rays * cam_img_mask_tensor.squeeze(0)

            # 生成稀疏深度
            cam_know_depth = torch.cat([cam_depth_tensor, cam_depth_mask_tensor], dim=0)  # (2,h,w)

            # 归一化（使用统一的avg_scale）
            cam_depth_tensor = cam_depth_tensor / avg_scale
            cam_know_depth = cam_know_depth.clone()
            cam_know_depth[0] = cam_know_depth[0] / avg_scale
            cam_pcl = cam_pcl / avg_scale

            camera_depth_tensors.append(cam_depth_tensor)
            camera_depth_mask_tensors.append(cam_depth_mask_tensor)
            camera_rays_list.append(cam_rays)
            camera_pcl_list.append(cam_pcl)
            camera_know_depth_list.append(cam_know_depth)

        if random.random() > self.omni3d_cfg.depth_full_prob:
            camera_know_depth_list = [torch.zeros_like(kd) for kd in camera_know_depth_list]

        if random.random() > self.omni3d_cfg.ray_prob:
            camera_rays_list = [torch.zeros_like(r) for r in camera_rays_list]

        # 归一化每个相机的3D标注（使用统一的avg_scale）
        for cam_idx in range(len(camera_configs)):
            if camera_bbox_side_len[cam_idx].numel() > 0:
                camera_bbox_side_len[cam_idx] = camera_bbox_side_len[cam_idx] / avg_scale
                camera_pose[cam_idx][:, 4:7] = camera_pose[cam_idx][:, 4:7] / avg_scale

        # 向后兼容：使用top_head（第0个相机）的数据
        K = camera_K_list[0]
        rays = camera_rays_list[0]
        depth_tensor = camera_depth_tensors[0]
        depth_mask_tensor = camera_depth_mask_tensors[0]
        img_mask_tensor = torch.from_numpy(camera_valid_masks_224[0]).unsqueeze(0)
        pcl = camera_pcl_list[0]
        know_depth = camera_know_depth_list[0]

        # 为每个摄像头生成对应的图像张量（同一时间的多视角）
        images: List[torch.Tensor] = [
            torch.from_numpy(camera_images_224[cam_idx]).permute(2, 0, 1).contiguous()
            for cam_idx in range(self.num_cameras)
        ]
        depths: List[torch.Tensor] = [
            torch.from_numpy(camera_depths_224[cam_idx]).unsqueeze(0).float()
            for cam_idx in range(self.num_cameras)
        ]

        # 创建掩码（category 和 instance level）
        # Omni3D 没有 pixel-level mask，使用top_head的bbox创建简单mask
        if camera_bboxes[0]:
            # 使用第一个bbox创建mask（简化版本）
            mask_category = torch.zeros((1, self.image_size, self.image_size), dtype=torch.uint8)
            mask_instance_level = torch.zeros((len(camera_bboxes[0]), 1, self.image_size, self.image_size), dtype=torch.uint8)
        else:
            mask_category = torch.zeros((1, self.image_size, self.image_size), dtype=torch.uint8)
            mask_instance_level = torch.zeros((0, 1, self.image_size, self.image_size), dtype=torch.uint8)

        class_name_clean = class_name.replace("_", " ").lower()
        if class_name_clean not in self.category_all_set:
            print(class_name_clean, dataset_name, '==================')
            return None

        # 10%负样本
        if random.random() < self.omni3d_cfg.neg_prob and len(self.category_full_set) > 1:
            # 把当前类别名分词，形成集合
            positive_words = set(class_name_clean.split())

            negative_candidates = []
            # 遍历所有类别（已是clean格式）
            for name in self.category_all_set:
                # 跳过与当前类别相同的项
                if name == class_name_clean:
                    continue
                # 分词成集合，与正样本的集合比对
                candidate_words = set(name.split())
                # 确保与正样本无单词重叠
                if positive_words.isdisjoint(candidate_words):
                    negative_candidates.append(name)

            # 如果找到了合格的负类别，随机选择一个作为当前类别
            if negative_candidates:
                class_name_clean = random.choice(negative_candidates)
                text_label = ''
                for view_i in range(len(images)):
                    text_label += f"<image{view_i}>"
                    text_label += "{}{}".format(class_name_clean, self.tokenizer.EXTRA_NO_OBJ_TOKENS)
                    text_label += f"</image{view_i}>"
        else:
            if self.config.uniform_mapping_6d_dataset:
                text_label = map_3d_label_to_string(class_name_clean, images, camera_bboxes, camera_pose, camera_bbox_side_len)
            else:
                text_label = map_3d_label_to_string_tokenizer(class_name_clean, images, camera_bboxes, camera_pose, camera_bbox_side_len, self.tokenizer)

        # 组装样本（与其它 detection dataset 完全一致的精简格式）
        sample = {
            # ===== 输入图像 / 几何（多相机列表） =====
            "images":        images,             # List[Tensor(3,224,224)]
            "images_vggt":   images_vggt_list,   # List[Tensor(3,518,518)]
            "camera_K":            camera_K_list,             # List[Tensor(3,3)]
            "camera_intrinsics":   [torch.tensor(intr, dtype=torch.float32) for intr in camera_intrinsics],  # List[Tensor(4,)]
            "camera_rays":         camera_rays_list,          # List[Tensor(3,H,W)]
            "camera_depth":        camera_depth_tensors,      # List[Tensor(1,H,W)]
            "camera_know_depth":   camera_know_depth_list,    # List[Tensor(2,H,W)]

            # ===== 文本 / 任务 =====
            "task":        f"detect the {class_name_clean}",
            "text_label":  text_label,

            # ===== 多相机 2D / 3D 标注 =====
            "bbox_2d":              camera_bboxes,             # List[List[Tensor(4,)]]
            "camera_pose":          camera_pose,               # List[Tensor(N,7)]
            "camera_bbox_side_len": camera_bbox_side_len,      # List[Tensor(N,3)]
            "camera_class_labels":  camera_class_labels,       # List[List[int]]
        }

        self.crop_param_manager.clear_cache()
        return sample

    def filter_objects(self, obj, bbox_2d_tensor, before_pad_size, K, dataset_name,
                       min_size=10, min_height_ratio=0.05, min_depth=0.01, max_depth=10, image_mask=None):
        """
        根据可见性、截断、2D bbox中心、高度和深度过滤目标

        Args:
            obj (dict): 包含目标信息的字典
            bbox_2d_tensor (torch.Tensor): 2D bbox [x1, y1, x2, y2]
            before_pad_size (tuple): padding之前的图像尺寸 (height, width)
            K (torch.Tensor): 相机内参矩阵 (3, 3) 或 (1, 3, 3)
            dataset_name (str): 数据集名称
            min_size (int): bbox的最小尺寸（宽度或高度）
            min_height_ratio (float): bbox高度相对于图像高度的最小比例
            min_depth (float): 保留目标的最小深度值
            max_depth (float): 保留目标的最大深度值
            image_mask (np.ndarray): 可选的图像掩码

        Returns:
            bool: 如果目标通过过滤返回True，否则返回False
        """
        # 检查bbox尺寸
        # if (bbox_2d_tensor[2] - bbox_2d_tensor[0]) < min_size or (bbox_2d_tensor[3] - bbox_2d_tensor[1]) < min_size:
        #     return False

        # 如果提供了image_mask，检查可见区域
        if image_mask is not None:
            # 提取坐标（浮点）
            x1, y1 = float(bbox_2d_tensor[0]), float(bbox_2d_tensor[1])
            x2, y2 = float(bbox_2d_tensor[2]), float(bbox_2d_tensor[3])

            # 获取图像尺寸
            h, w = image_mask.shape

            # 转换为整数并截断到图像范围内（避免负索引）
            x1_int = max(0, min(int(x1), w))
            y1_int = max(0, min(int(y1), h))
            x2_int = max(0, min(int(x2), w))
            y2_int = max(0, min(int(y2), h))

            # 安全地提取mask区域（确保不会有负索引）
            if y2_int > y1_int and x2_int > x1_int:
                mask_2dbbox = image_mask[y1_int:y2_int, x1_int:x2_int]
                size_vis = int(mask_2dbbox.sum())
            else:
                size_vis = 0

            # 计算完整bbox面积（使用浮点坐标，包括图像外的部分）
            # 目标：计算"图像内可见部分 / 完整物体bbox（包括图像外）"的占比
            size_full = (y2 - y1) * (x2 - x1)

            # 计算可见占比
            if size_full > 0:
                vis_ratio = size_vis / size_full
            else:
                vis_ratio = 0

            # 调试输出
            if self.debug_mode:
                print(f"  bbox_float=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                print(f"  bbox_int=[{x1_int}, {y1_int}, {x2_int}, {y2_int}]")
                print(f"  size_vis={size_vis}, size_full={size_full:.1f}, vis_ratio={vis_ratio:.4f}")

            # 过滤条件：可见部分占完整bbox的比例 < 30% 或 可见像素数不足
            # 动态阈值：至少为完整bbox面积的5%，或150像素（取较大值）
            min_vis_pixels = max(150.0, size_full * 0.05)
            if vis_ratio < 0.3 or size_vis < min_vis_pixels:
                if self.debug_mode:
                    print(f"  ❌ 过滤: vis_ratio={vis_ratio:.3f} < 0.3 或 size_vis={size_vis} < {min_vis_pixels:.1f}")
                return False

        # # 检查bbox高度比例
        # if bbox_2d_tensor[3] - bbox_2d_tensor[1] < min_height_ratio * before_pad_size[0]:
        #     return False

        # 获取3D bbox参数
        x, y, z, w, h, l, yaw = obj['3d_bbox']

        # 检查可见性和截断
        visibility = obj.get('visibility', 1)  # 默认为1（完全可见）
        truncation = obj.get('truncation', 0)  # 默认为0（无截断）

        if visibility == -1:
            visibility = 1
        if truncation == -1:
            truncation = 0

        # 基于可见性和截断的过滤
        if visibility < 0.333 or truncation > 0.33:
            # print(visibility)
            # print(truncation)
            # print("1056")
            return False

        # 将3D中心投影到2D
        K_np = K.squeeze(0).numpy() if isinstance(K, torch.Tensor) else K
        if K_np.ndim == 3:
            K_np = K_np.squeeze(0)

        center_2d = project_to_image(np.array([[x, y, z]]), K_np).squeeze(0)

        # 检查2D中心是否在图像边界内
        if center_2d[0] < 0 or center_2d[0] > before_pad_size[1] or \
           center_2d[1] < 0 or center_2d[1] > before_pad_size[0]:
            return False

        # 检查深度是否在有效范围内
        depth = z
        # if depth < min_depth or depth > max_depth:
        #     return False
        if depth < min_depth:
            print("1077")
            return False
        # 对于某些特定数据集，无需额外过滤
        if not 'kitti' in dataset_name and not 'sunrgbd' in dataset_name and \
                not 'arkitscenes' in dataset_name and not 'objectron' in dataset_name and \
                not 'hypersim' in dataset_name and not 'nuscenes' in dataset_name:
            return True

        # zero-shot类别过滤（如果配置中启用）
        if hasattr(self, 'omni3d_cfg') and hasattr(self.omni3d_cfg, 'zero_shot') and self.omni3d_cfg.zero_shot:
            # cubercnn categories
            thing_dataset_id_to_contiguous_id = {
                "0": 0, "1": 1, "3": 2, "4": 3, "5": 4, "8": 5, "9": 6, "10": 7, "11": 8, "12": 9,
                "13": 10, "14": 11, "15": 12, "16": 13, "17": 14, "18": 15, "19": 16, "20": 17,
                "21": 18, "22": 19, "23": 20, "24": 21, "25": 22, "26": 23, "27": 24, "28": 25,
                "29": 26, "30": 27, "31": 28, "32": 29, "33": 30, "34": 31, "35": 32, "36": 33,
                "37": 34, "38": 35, "39": 36, "40": 37, "42": 38, "43": 39, "44": 40, "45": 41,
                "46": 42, "47": 43, "48": 44, "49": 45, "52": 46, "53": 47, "57": 48, "61": 49
            }
            if f"{obj['label']}" not in thing_dataset_id_to_contiguous_id.keys():
                if self.mode == 'train':
                    return False
                elif hasattr(self.omni3d_cfg, 'inference_basic') and self.omni3d_cfg.inference_basic:
                    return False

        # 所有检查通过
        return True

    def _load_depth(self, depth_path, dataset_name, img, img_path=None):
        """加载深度图（参考 DetAny3D 实现）

        Args:
            depth_path: 深度图路径
            dataset_name: 数据集名称
            img: 图像数组（用于获取尺寸）
            img_path: 图像路径（用于HDF5深度加载）

        Returns:
            depth: 深度图数组 [H, W]
        """
        height, width = img.shape[:2]

        # 情况1: depth_path 为 None，尝试从HDF5文件夹加载
        if depth_path is None:
            depth = self._load_depth_from_hdf5_folder(dataset_name, img_path, img)
            if depth is not None:
                return depth
            else:
                # 无法从HDF5加载，创建零深度图
                if self.debug_mode:
                    print(f"⚠️ Warning: No depth path and no HDF5 depth, creating zero depth map")
                return np.zeros((height, width), dtype=np.float32)

        # 情况2: 文件不存在
        if not os.path.exists(depth_path):
            if self.debug_mode:
                print(f"⚠️ Warning: Depth file not found: {depth_path}, creating zero depth map")
            return np.zeros((height, width), dtype=np.float32)

        # 情况3: PNG 深度图
        if depth_path.endswith('.png'):
            try:
                depth = np.array(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
                if hasattr(self.dataset_dict[dataset_name], 'metric_scale'):
                    depth = depth / self.dataset_dict[dataset_name].metric_scale
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error loading PNG depth: {e}, creating zero depth map")
                return np.zeros((height, width), dtype=np.float32)

        # 情况4: NPY 文件
        elif depth_path.endswith('.npy'):
            try:
                depth = np.load(depth_path).astype(np.float32)
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error loading NPY depth: {e}, creating zero depth map")
                return np.zeros((height, width), dtype=np.float32)

        # 情况5: Objectron 专用 HDF5 格式
        elif depth_path.endswith('objectron_depth.hdf5'):
            try:
                # 需要传入图片路径来获取对应的深度数据
                from omni3d_hdf5_dataset import read_objectron_depth_from_hdf5
                depth = read_objectron_depth_from_hdf5(depth_path, img_path, visualize=False)

                if depth is not None:
                    depth = depth.astype(np.float32)

                    # 确保深度数据是2维的
                    if depth.ndim == 1:
                        if self.debug_mode:
                            print(f"⚠️ Warning: 1D depth data detected, creating zero depth map")
                        return np.zeros((height, width), dtype=np.float32)
                    elif depth.ndim == 3:
                        if depth.shape[2] == 1:
                            depth = depth.squeeze(2)
                        else:
                            depth = np.mean(depth, axis=2)

                    # 检查深度数据的有效性
                    if depth.size == 0:
                        if self.debug_mode:
                            print(f"⚠️ Warning: Empty depth data, creating zero depth map")
                        return np.zeros((height, width), dtype=np.float32)

                    # 根据数据集配置进行缩放
                    if hasattr(self.dataset_dict[dataset_name], 'metric_scale'):
                        depth = depth / self.dataset_dict[dataset_name].metric_scale

                    if self.debug_mode:
                        print(f"✓ Loaded Objectron HDF5 depth, shape: {depth.shape}, range: {depth.min():.3f} - {depth.max():.3f}")
                else:
                    if self.debug_mode:
                        print(f"⚠️ Warning: No Objectron depth data found, creating zero depth map")
                    return np.zeros((height, width), dtype=np.float32)
            except ImportError:
                if self.debug_mode:
                    print(f"⚠️ Warning: Cannot import read_objectron_depth_from_hdf5, creating zero depth map")
                return np.zeros((height, width), dtype=np.float32)
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error loading Objectron HDF5 depth: {e}, creating zero depth map")
                return np.zeros((height, width), dtype=np.float32)

        # 情况6: Hypersim HDF5 格式
        elif depth_path.endswith('.hdf5'):
            try:
                import h5py
                # Hypersim 特定参数
                intWidth = 1024
                intHeight = 768
                fltFocal = 886.81

                hf = h5py.File(depth_path, 'r')
                n1 = hf.get('dataset')[:]
                npyDistance = np.array(n1)

                # 计算图像平面坐标
                npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
                npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
                npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
                npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

                # 转换距离到深度
                depth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
                depth = depth.astype(np.float32)

                if self.debug_mode:
                    print(f"✓ Loaded Hypersim HDF5 depth, shape: {depth.shape}, range: {depth.min():.3f} - {depth.max():.3f}")
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error loading Hypersim HDF5 depth: {e}, creating zero depth map")
                return np.zeros((height, width), dtype=np.float32)

        # 情况7: 不支持的格式
        else:
            if self.debug_mode:
                print(f"⚠️ Warning: Unsupported depth format: {depth_path}, creating zero depth map")
            return np.zeros((height, width), dtype=np.float32)

        # 过滤异常值
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0

        return depth

    def _load_depth_from_hdf5_folder(self, dataset_name, img_path, img):
        """从指定文件夹的HDF5文件中加载深度数据

        Args:
            dataset_name: 数据集名称
            img_path: 图片路径
            img: 图片数组，用于获取尺寸

        Returns:
            depth: 深度图数组，如果无法加载则返回None
        """
        try:
            # 定义HDF5文件路径映射
            hdf5_folder = "/home/tione/notebook/workspace/shanejhuang/3D-MOOD/data"
            hdf5_file_mapping = {
                'kitti': 'KITTI_object_depth.hdf5',
                'nuscenes': 'nuscenes_depth.hdf5',
                'arkitscenes': 'ARKitScenes_depth.hdf5',
                'objectron': 'objectron_depth.hdf5',
                'hypersim': 'hypersim_depth.hdf5',
                'sunrgbd': 'SUNRGBD_depth.hdf5'
            }

            # 根据数据集名称获取对应的HDF5文件名
            if dataset_name not in hdf5_file_mapping:
                if self.debug_mode:
                    print(f"⚠️ No HDF5 file mapping found for dataset: {dataset_name}")
                return None

            hdf5_filename = hdf5_file_mapping[dataset_name]
            hdf5_path = os.path.join(hdf5_folder, hdf5_filename)

            # 检查HDF5文件是否存在
            if not os.path.exists(hdf5_path):
                if self.debug_mode:
                    print(f"⚠️ HDF5 file not found: {hdf5_path}")
                return None

            # 尝试使用 HDF5DepthReader 读取深度数据
            try:
                import sys
                if '/home/tione/notebook/workspace/shanejhuang/DetAny3D' not in sys.path:
                    sys.path.append('/home/tione/notebook/workspace/shanejhuang/DetAny3D')
                from omni3d_hdf5_dataset import HDF5DepthReader

                reader = HDF5DepthReader(hdf5_path)
                depth = self._get_depth_from_hdf5_reader(reader, dataset_name, img_path)

                if depth is not None:
                    # 确保深度数据是2维的
                    if depth.ndim == 3:
                        depth = depth.squeeze(2)

                    # 根据数据集配置进行缩放
                    if hasattr(self.dataset_dict[dataset_name], 'metric_scale'):
                        depth = depth / self.dataset_dict[dataset_name].metric_scale

                    if self.debug_mode:
                        print(f"✓ Loaded depth from HDF5: {hdf5_path}, shape: {depth.shape}")
                    return depth
                else:
                    if self.debug_mode:
                        print(f"⚠️ No depth found in HDF5 for image: {img_path}")
                    return None
            except ImportError as e:
                if self.debug_mode:
                    print(f"⚠️ Cannot import HDF5DepthReader: {e}")
                return None
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Error loading depth from HDF5 folder: {e}")
            return None

    def _get_depth_from_hdf5_reader(self, reader, dataset_name, img_path):
        """
        使用HDF5DepthReader智能获取深度数据

        Args:
            reader: HDF5DepthReader实例
            dataset_name: 数据集名称
            img_path: 图片路径

        Returns:
            depth: 深度图数组，如果无法加载则返回None
        """
        try:
            # 从图片路径中提取必要信息
            img_basename = os.path.basename(img_path)
            img_name_without_ext = os.path.splitext(img_basename)[0]

            # 根据数据集类型智能解析参数
            if 'kitti' in dataset_name.lower():
                dataset_name='kitti'
                # KITTI: KITTI_object/training/image_2/000000.png -> training/image_2/000000.png
                # split = "training"
                # scene_id = "training"
                # 从完整路径中提取图像名称
                path_parts = img_path.split('/')
                if 'image_2' in path_parts:
                    image_2_idx = path_parts.index('image_2')
                    split = path_parts[image_2_idx-1]
                    scene_id = path_parts[image_2_idx-1]
                    if image_2_idx + 1 < len(path_parts):
                        image_name = path_parts[image_2_idx + 1]  # 直接使用原始文件名
                    else:
                        image_name = img_name_without_ext + '.png'
                else:
                    image_name = img_name_without_ext + '.png'

            elif 'nuscenes' in dataset_name.lower():
                dataset_name='nuscenes'
                # nuScenes: samples/CAM_FRONT/xxx.png
                split = "samples"
                scene_id = "samples"
                # 从完整路径中提取图像名称
                path_parts = img_path.split('/')
                if 'CAM_FRONT' in path_parts:
                    cam_idx = path_parts.index('CAM_FRONT')
                    if cam_idx + 1 < len(path_parts):
                        # 确保图像名称是PNG格式
                        original_name = path_parts[cam_idx + 1]
                        if original_name.endswith('.jpg'):
                            image_name = original_name.replace('.jpg', '.png')
                        elif original_name.endswith('.png'):
                            image_name = original_name
                        else:
                            image_name = original_name + '.png'
                    else:
                        image_name = img_name_without_ext + '.png'
                else:
                    image_name = img_name_without_ext + '.png'

            elif 'arkitscenes' in dataset_name.lower():
                dataset_name='arkitscenes'
                # ARKitScenes: /path/to/ARKitScenes/Training/41048229/7085.745_00002216.jpg -> Training/41048229/7085.745_00002216.png
                path_parts = img_path.split('/')

                # 查找ARKitScenes在路径中的位置
                arkit_idx = -1
                for i, part in enumerate(path_parts):
                    if part == 'ARKitScenes':
                        arkit_idx = i
                        break

                if arkit_idx != -1 and arkit_idx + 3 < len(path_parts):
                    # 确保路径格式正确：ARKitScenes/Training/scene_id/image_name
                    if path_parts[arkit_idx + 1] == 'Training' or path_parts[arkit_idx + 1] == 'Validation':
                        split = path_parts[arkit_idx + 1]
                        scene_id = path_parts[arkit_idx + 2]  # 41048229
                        image_name = path_parts[arkit_idx + 3].replace('.jpg', '.png')  # 7085.745_00002216.png
                    else:
                        print(f"Warning: Could not parse ARKitScenes path: {img_path}")
                        return None
                else:
                    print(f"Warning: Could not find ARKitScenes in path: {img_path}")
                    return None

            elif 'objectron' in dataset_name.lower():
                dataset_name='objectron'
                # Objectron: /path/to/objectron/train/bike_batch_10_3_0000000.jpg -> train/bike_batch_10_3_0000000_depth.png
                # split = "train"
                # scene_id = "train"
                path_parts = img_path.split('/')

                # 查找objectron在路径中的位置
                objectron_idx = -1
                for i, part in enumerate(path_parts):
                    if part == 'objectron':
                        objectron_idx = i
                        break

                if objectron_idx != -1 and objectron_idx + 2 < len(path_parts):
                    # 确保路径格式正确：objectron/train/image_name
                    if path_parts[objectron_idx + 1] == 'train':
                        split = "train"
                        scene_id = "train"
                        base_name = path_parts[objectron_idx + 2].replace('.jpg', '')  # 移除.jpg
                        image_name = base_name + '_depth.png'  # 添加_depth.png
                    elif path_parts[objectron_idx + 1] == 'test':
                        split = "test"
                        scene_id = "test"
                        base_name = path_parts[objectron_idx + 2].replace('.jpg', '')  # 移除.jpg
                        image_name = base_name + '_depth.png'  # 添加_depth.png
                    else:
                        image_name = img_name_without_ext + '_depth.png'
                else:
                    image_name = img_name_without_ext + '_depth.png'

            elif 'hypersim' in dataset_name.lower():
                dataset_name='hypersim'
                # Hypersim: /path/to/hypersim/ai_001_001/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg -> ai_001_001/images/scene_cam_00_final_preview/frame.0000.tonemap.png
                # split = "Training"
                path_parts = img_path.split('/')

                # 查找hypersim在路径中的位置
                hypersim_idx = -1
                for i, part in enumerate(path_parts):
                    if part == 'hypersim':
                        hypersim_idx = i
                        break

                if hypersim_idx != -1 and hypersim_idx + 4 < len(path_parts):
                    # 确保路径格式正确：hypersim/ai_001_001/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg
                    if path_parts[hypersim_idx + 2] == 'images':
                        scene_id = path_parts[hypersim_idx + 1]  # ai_001_001
                        # 构建完整的图像路径：images/scene_cam_00_final_preview/frame.0000.tonemap.png
                        sub_path = '/'.join(path_parts[hypersim_idx + 3:-1])  # scene_cam_00_final_preview
                        image_name = f"{sub_path}/{path_parts[-1].replace('.tonemap.jpg', '.tonemap.png')}"
                    else:
                        print(f"Warning: Could not parse Hypersim path: {img_path}")
                        return None
                else:
                    print(f"Warning: Could not find hypersim in path: {img_path}")
                    return None

            else:
                print(f"Warning: Unknown dataset type: {dataset_name}")
                return None

            # 尝试获取深度数据
            depth = reader.get_depth_map(
                dataset=dataset_name,
                scene_id=scene_id,
                image_name=image_name,
                split=split,
                depth_scale=1000.0,  # 默认深度缩放因子
                max_depth=80.0       # 默认最大深度
            )

            return depth

        except Exception as e:
            # 只在调试模式下打印详细信息
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"Error in _get_depth_from_hdf5_reader: {e}")
            return None

    def _crop_hw(self, img, depth, K):
        """裁剪图像和深度图"""
        if img.dim() == 4:
            img = img.squeeze(0)

        h, w = img.shape[1:3]

        # 确定性裁剪 (中心裁剪,确保可以被14整除)
        crop_width, crop_height, left, top = self._deterministic_crop_coords(orig_width=w, orig_height=h)

        h = crop_height
        w = crop_width
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        center_h, center_w = h // 2, w // 2
        start_h = center_h - new_h // 2
        start_w = center_w - new_w // 2

        img_cropped = img[:, start_h + top:start_h + top + new_h, start_w + left:start_w + left + new_w]
        depth_cropped = depth[start_h + top:start_h + top + new_h, start_w + left:start_w + left + new_w]

        K_cropped = None
        if K is not None:
            K_cropped = K.clone()
            K_cropped[0, 2] -= (start_w + left)
            K_cropped[1, 2] -= (start_h + top)

        return img_cropped.unsqueeze(0), depth_cropped, K_cropped, left, top

    def _deterministic_crop_coords(self, crop_scale=(0.75, 0.75), aspect_ratio=(1.33, 1.33),
                                   orig_width=224, orig_height=224):
        """计算确定性裁剪坐标"""
        # scale = (crop_scale[0] + crop_scale[1]) / 2.0
        scale = random.uniform(*crop_scale)
        log_ratio_min = np.log(aspect_ratio[0])
        log_ratio_max = np.log(aspect_ratio[1])
        target_ratio = np.exp((log_ratio_min + log_ratio_max) / 2.0)

        crop_area = orig_width * orig_height * scale
        crop_width = int(round(np.sqrt(crop_area * target_ratio)))
        crop_height = int(round(np.sqrt(crop_area / target_ratio)))

        crop_width = min(crop_width, orig_width)
        crop_height = min(crop_height, orig_height)

        left = (orig_width - crop_width) // 2
        top = (orig_height - crop_height) // 2

        return crop_width, crop_height, left, top

    def _process_depth(self, img_for_sam, depth, before_pad_size, dataset_name, K):
        """处理深度图,进行居中填充"""
        h_pad, w_pad = img_for_sam.shape[0], img_for_sam.shape[1]
        h, w = before_pad_size

        depth_padded = torch.zeros((h_pad, w_pad), dtype=depth.dtype, device=depth.device)
        depth_mask_padded = torch.zeros_like(depth_padded)

        pad_top = (h_pad - h) // 2
        pad_left = (w_pad - w) // 2

        start_h = pad_top
        end_h = start_h + h
        start_w = pad_left
        end_w = start_w + w

        depth_padded[start_h:end_h, start_w:end_w] = depth
        depth_mask_padded[start_h:end_h, start_w:end_w] = 1

        if K.dim() == 3:
            K_adjusted = K.squeeze(0).clone()
            need_unsqueeze = True
        else:
            K_adjusted = K.clone()
            need_unsqueeze = False

        # 应用深度过滤
        if hasattr(self.dataset_dict[dataset_name], 'max_distance'):
            if self.dataset_dict[dataset_name].max_distance:
                depth_mask_padded[depth_padded >= self.dataset_dict[dataset_name].max_distance] = 0

        if hasattr(self.dataset_dict[dataset_name], 'min_distance'):
            if self.dataset_dict[dataset_name].min_distance:
                depth_mask_padded[depth_padded <= self.dataset_dict[dataset_name].min_distance] = 0

        K_adjusted[0, 2] += pad_left
        K_adjusted[1, 2] += pad_top

        if need_unsqueeze:
            K_adjusted = K_adjusted.unsqueeze(0)

        return depth_padded, depth_mask_padded, K_adjusted, pad_left, pad_top

    def _resize_with_pad_np(self, img: np.ndarray, target_w: int, target_h: int,
                            pad_value: float = 0, interpolation: int = cv2.INTER_LINEAR) -> Tuple[
        np.ndarray, float, int, int]:
        """调整图像大小并填充到目标尺寸 (numpy版本)

        Args:
            img: 输入图像 (H, W) 或 (H, W, C)
            target_w: 目标宽度
            target_h: 目标高度
            pad_value: 填充值
            interpolation: 插值方法

        Returns:
            padded_img: 填充后的图像
            resize_ratio: 缩放比例
            pad_left: 左边填充像素数
            pad_top: 上边填充像素数
        """
        # 获取原始尺寸
        h, w = img.shape[:2]

        # 计算缩放比例，保持长宽比
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 调整大小
        if len(img.shape) == 3:
            resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        # 计算填充量
        pad_h = target_h - new_h
        pad_w = target_w - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 填充
        if len(img.shape) == 3:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=pad_value)
        else:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=pad_value)

        return padded, scale, pad_left, pad_top


# NOTE: The legacy ``DataCollatorForPI0Omni3DConsumerDataset`` has been removed.
# All four detection datasets (omni6d / omni3d / bop / clutter) now share the
# unified ``CollatorForDetectionDataset`` defined in ``collators.py``. Use:
#     from collators import CollatorForDetectionDataset
#     collator = CollatorForDetectionDataset()


# NOTE: Visualization helpers (save_image_chw, visualize_depth_simple_comparison,
# visualize_rays_as_rgb, compute_3d_bbox_vertices_batch, project_3d_to_2d) and a
# duplicate ``quaternion_to_rotation_matrix`` definition have been removed. See
# git history if needed; the canonical helpers live in dataset_omni6d.py.


# ==================== 测试代码 ====================
import hydra
@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="base",
)
def main(cfg):
    bin_tokenizer = BinTokenizer(cfg.statistics_path_6d_dataset)
    train_det_dataset = Omni3DConsumerDataset(config=cfg, tokenizer=bin_tokenizer)
    # Unified collator shared by all four detection datasets.
    # Parameter-free: ``num_cameras`` is inferred from each sample.
    from data.collators import CollatorForDetectionDataset
    data_det_collator = CollatorForDetectionDataset()
    train_det_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=train_det_dataset, collate_fn=data_det_collator)
    for batch in train_det_dataloader:
        print(batch["observation.images.top_head"].shape)
        print(torch.max(batch["observation.depth_priors.top_head"]), torch.min(batch["observation.depth_priors.top_head"]))
        print(batch["task"])
        print(batch["text_label"])
        # res = text_to_class_attr_dict(batch["text_label"][0])
        if cfg.uniform_mapping_6d_dataset:
            res = text_to_class_attr_dict(batch["text_label"][0])
        else:
            res = text_to_class_attr_dict_tokenizer(batch["text_label"][0], bin_tokenizer)
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

        # plot_2d_3d_objects(images, res, res, batch["task"][0], intrinsics)
        visualize_2d_3d_all(images, res, res, intrinsics, batch["task"][0],)
        visualize_views(images, depths, rays)
        break
if __name__ == "__main__":
    main()
