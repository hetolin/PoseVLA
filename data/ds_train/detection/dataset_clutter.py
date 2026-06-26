"""
Clutter 数据集加载器,输出与 PI0 模型兼容的格式
基于 GraspClutter6D 标准格式,参考 dataset_omni6d.py 的输出结构
支持多摄像头、3D 检测、深度图等功能
"""

import glob
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, ListConfig, DictConfig
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from utils.vis import visualize_2d_3d_all, visualize_views
from utils.mapping_token import text_to_class_attr_dict, map_3d_label_to_string
from utils.mapping_token import text_to_class_attr_dict_tokenizer, map_3d_label_to_string_tokenizer, BinTokenizer
# BOP toolkit 导入
import sys
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).parent.parent / "bop_toolkit"))
from bop_toolkit_lib import inout
import tempfile
import shutil

# GraspClutter6D API 导入（用于加载抓取数据）
GRASPCLUTTER6D_AVAILABLE = True
GRASP_HEIGHT = 0.02
from data.ds_train.detection.graspclutter6dAPI import GraspGroup, generate_views, transform_points, batch_viewpoint_params_to_matrix


def remap_pose(trans, rot_matrix):
    """
    将抓取位姿从 [x front, y right, z down] (trans, rot) 映射到 [z front, x right, y down] 坐标系。
    输入:
        trans: (N, 3) np.ndarray, translation
        rot_matrix:  (N,3,3) np.ndarray, rotation matrix
    输出: (N,7) np.ndarray, [trans_remap, quat_wxyz]
    """
    # 1. translation: [x, y, z] → [z, x, y]
    # trans_remap = trans[:, [2, 0, 1]]  # (N, 3)
    trans_remap = trans[:, [0, 1, 2]]  # (N, 3)

    # 2. rotation
    T = np.array([
        [0, 0, 1],  # new_x maps to old_z
        [1, 0, 0],  # new_y maps to old_x
        [0, 1, 0],  # new_z maps to old_y
    ])
    # 对每个rot_matrix做映射
    rot_remap = rot_matrix @ T # (N, 3, 3)

    # 四元数转换 batch
    quats_xyzw = Rotation.from_matrix(rot_remap).as_quat()  # (N, 4) x y z w
    quats_wxyz = np.concatenate([quats_xyzw[:, 3:4], quats_xyzw[:, :3]], axis=1)  # (N, 4), w x y z

    # 合并
    grasp_pose_remap = np.concatenate([quats_wxyz, trans_remap], axis=1)  # (N, 7)
    return grasp_pose_remap


# ---------------------------------------------------------------------------
# Re-use the canonical helper implementations defined in ``dataset_omni6d``.
# Keeping a single source of truth avoids subtle drift between the four
# detection datasets (omni6d / omni3d / bop / clutter).
# ---------------------------------------------------------------------------
from data.ds_train.detection.dataset_omni6d import (  # noqa: E402  -- intentional re-export
    _intrinsics_to_matrix,
    _generate_coord_grid,
    _depth_to_pcl,
    _generate_rays,
    _random_crop_coords,
    _deterministic_crop_coords,
    _crop_resize_with_pad,
    _resize_with_pad,
    _threshold_depth_map,
    _build_sparse_depth,
    quaternion_to_rotation_matrix,
    CropParams,
    CropParamManager,
)


def _compute_bbox_from_mask(mask_path: Path) -> List[int]:
    """从 mask 文件计算 bounding box [x, y, w, h]

    参数:
        mask_path: mask 图像文件路径

    返回:
        [x, y, w, h] - bbox 坐标，如果 mask 不存在或为空则返回 [0, 0, 1, 1]
    """
    try:
        import cv2
        if not mask_path.exists():
            return [0, 0, 1, 1]

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.size == 0:
            return [0, 0, 1, 1]

        # 找到所有非零像素
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return [0, 0, 1, 1]

        # 计算 bbox
        x_min = int(cols.min())
        y_min = int(rows.min())
        x_max = int(cols.max())
        y_max = int(rows.max())

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        return [x_min, y_min, width, height]
    except Exception as e:
        # 如果出错，返回默认值
        return [0, 0, 1, 1]


@dataclass
class _SampleEntry:
    """单个样本的索引信息（每个物体类别一个entry）"""
    scene_id: int
    im_id: int
    scene_dir: Path
    dataset_root: Path  # 记录数据集根目录（用于多数据集支持）
    dataset_name: str  # 记录数据集名称（如 ycbv, hb）
    obj_id: int  # 物体类别ID
    ann_id: int = None  # GraspClutter6D 标注ID（用于多相机随机选择）


class BopClutterConsumerDataset(Dataset):
    """
    BOP 数据集,输出与 PI0 模型兼容的格式

    基于 BOP 场景格式,将每一帧展开为独立样本
    支持多摄像头、深度图、3D 检测等功能
    """

    def __init__(self, config: Any, tokenizer: BinTokenizer, yaml_name="clutter_train") -> None:
        super().__init__()
        self.config = config

        # 检查配置
        if not hasattr(config.dataset_clutter, yaml_name):
            raise ValueError(f"config.dataset_clutter{yaml_name} 未配置,无法构建 BOP 数据集")

        # bop_cfg = config.dataset_clutter
        bop_cfg = getattr(config.dataset_clutter, yaml_name)
        self.bop_cfg = bop_cfg

        # GraspClutter6D API 集成（可选）
        # 如果配置中指定使用 GraspClutter6D 格式，则初始化相关属性
        self.use_graspclutter6d = bop_cfg.get("use_graspclutter6d", False)
        if self.use_graspclutter6d:
            # GraspClutter6D 格式参数
            gc6d_camera_cfg = bop_cfg.get("gc6d_camera", "zivid")

            # 支持单个相机或相机列表
            if isinstance(gc6d_camera_cfg, (list, tuple, ListConfig)):
                self.gc6d_cameras = list(gc6d_camera_cfg)
            else:
                self.gc6d_cameras = [gc6d_camera_cfg]

            # 相机名称标准化（兼容性处理）
            standardized_cameras = []
            for cam in self.gc6d_cameras:
                if cam == 'realsense':
                    standardized_cameras.append('realsense-d435')
                elif cam == 'kinect':
                    standardized_cameras.append('azure-kinect')
                else:
                    standardized_cameras.append(cam)
            self.gc6d_cameras = standardized_cameras

            # 为了向后兼容，保留 gc6d_camera（使用第一个相机）
            self.gc6d_camera = self.gc6d_cameras[0]

            if len(self.gc6d_cameras) == 1:
                print(f"[BopConsumerDataset] 启用 GraspClutter6D 加载模式，相机类型: {self.gc6d_cameras[0]}")
            else:
                print(f"[BopConsumerDataset] 启用 GraspClutter6D 加载模式，相机类型（随机选择）: {self.gc6d_cameras}")

        # 数据集路径（支持单个路径或多个路径列表）
        data_root_cfg = bop_cfg.data_root
        # 支持 OmegaConf 的 ListConfig 以及标准的 list/tuple
        if isinstance(data_root_cfg, (list, tuple, ListConfig)):
            # 多数据集模式
            self.data_roots = [Path(p).expanduser() for p in data_root_cfg]
            self.multi_dataset = True
        else:
            # 单数据集模式
            self.data_roots = [Path(data_root_cfg).expanduser()]
            self.multi_dataset = False

        # 验证所有数据集路径都存在
        for data_root in self.data_roots:
            if not data_root.exists():
                raise FileNotFoundError(f"BOP 数据目录不存在: {data_root}")

        self.split = bop_cfg.get("split", "train")

        # 图像参数
        self.image_size = int(bop_cfg.get("image_size", 224))
        self.vggt_image_size = int(bop_cfg.get("vggt_image_size", 518))
        self.img_history_size = int(bop_cfg.get("img_history_size", config.dataset.img_history_size))

        # 多摄像头配置（与 DetConsumerDataset 保持一致）
        default_camera_configs = [
            {"name": "top_head", "crop_scale": (0.90, 0.90), "aspect_ratio": (1.33, 1.33)},
            {"name": "hand_left", "crop_scale": (0.25, 0.25), "aspect_ratio": (1.33, 1.33)},
            {"name": "hand_right", "crop_scale": (0.55, 0.55), "aspect_ratio": (1.33, 1.33)},
        ]
        self.camera_configs = bop_cfg.get("camera_configs", default_camera_configs)
        self.num_cameras = len(self.camera_configs)

        # 深度处理参数
        self.depth_max_percentile = float(bop_cfg.get("depth_max_percentile", 99.0))
        self.depth_min_percentile = float(bop_cfg.get("depth_min_percentile", 1.0))
        self.depth_max_value = float(bop_cfg.get("depth_max_value", -1.0))
        # BOP 深度单位转换因子 (mm->m 使用 1000.0, 0.1mm->m 使用 10000.0)
        self.unit_scale = float(bop_cfg.get("unit_scale", 1000.0))

        # 数据加载参数
        self.load_depth = bool(bop_cfg.get("load_depth", True))
        self.load_masks = bool(bop_cfg.get("load_masks", True))
        self.load_gt = bool(bop_cfg.get("load_gt", True))
        self.max_samples = bop_cfg.get("max_samples", None)
        self.sample_retries = int(bop_cfg.get("max_retries", 10))

        # GraspClutter6D 抓取加载参数
        self.load_grasps = bool(bop_cfg.get("load_grasps", False))  # 是否加载抓取数据
        self.grasp_format = bop_cfg.get("grasp_format", "6d")  # 抓取格式：'6d' 或 'rect'
        self.fric_coef_thresh = float(bop_cfg.get("fric_coef_thresh", 0.4))  # 摩擦系数阈值（越低越好）
        self.remove_invisible_grasps = bool(bop_cfg.get("remove_invisible_grasps", True))  # 移除不可见抓取
        self.max_grasps_per_object = bop_cfg.get("max_grasps_per_object", None)  # 每个物体最多保留的 grasp 数量
        self.grasp_selection_mode = bop_cfg.get("grasp_selection_mode", "top-k")  # grasp 选择模式
        self.max_sample_points = bop_cfg.get("max_sample_points", None)  # 限制采样点数量（性能优化）

        # 预处理抓取路径（离线处理 remove_invisible_grasps）
        self.use_preprocessed_grasps = bool(bop_cfg.get("use_preprocessed_grasps", False))
        self.preprocessed_grasps_dir = bop_cfg.get("preprocessed_grasps_dir", "cache/filtered_grasps")

        # 预加载抓取标签和碰撞标签（如果需要）
        self.grasp_labels_cache = {}
        self.collision_labels_cache = {}
        if self.load_grasps and self.use_graspclutter6d and GRASPCLUTTER6D_AVAILABLE:
            print(f"[BopConsumerDataset] 启用抓取数据加载")
            print(f"  - 抓取格式: {self.grasp_format}")
            print(f"  - 摩擦系数阈值: {self.fric_coef_thresh}")
            print(f"  - 移除不可见抓取: {self.remove_invisible_grasps}")

            # 检查预处理抓取
            if self.use_preprocessed_grasps:
                self.preprocessed_grasps_dir = Path(self.preprocessed_grasps_dir)
                if not self.preprocessed_grasps_dir.exists():
                    print(f"  Warning: 预处理抓取目录不存在: {self.preprocessed_grasps_dir}")
                    self.use_preprocessed_grasps = False
                else:
                    print(f"  - 使用预处理抓取: {self.preprocessed_grasps_dir}")
                    # 如果使用预处理数据，则跳过实时可见性过滤
                    if self.remove_invisible_grasps:
                        print(f"  - 已禁用实时可见性过滤（使用预处理结果）")
                        self.remove_invisible_grasps = False

            if self.max_grasps_per_object is not None:
                print(f"  - 每个物体最多保留 grasp 数: {self.max_grasps_per_object}")
                print(f"  - Grasp 选择模式: {self.grasp_selection_mode}")
            if self.max_sample_points is not None:
                print(f"  - 每个物体最多采样点数: {self.max_sample_points} (性能优化)")

        # ID到名称的映射 (支持多数据集)
        id2name_cfg = bop_cfg.get("id2name", {})
        # 支持 OmegaConf 的 ListConfig 以及标准的 list/tuple
        if self.multi_dataset and isinstance(id2name_cfg, (list, tuple, ListConfig)):
            # 多数据集模式：id2name 为列表，每个数据集一个映射
            if len(id2name_cfg) != len(self.data_roots):
                raise ValueError(f"id2name 列表长度 ({len(id2name_cfg)}) 与数据集数量 ({len(self.data_roots)}) 不匹配")
            # 转换为标准 Python list（如果是 ListConfig）
            self.id2name_list = list(id2name_cfg)
        else:
            # 单数据集模式或共享映射
            self.id2name_list = [id2name_cfg] * len(self.data_roots)

        # 加载物体模型信息 (3D尺寸等) - 支持多数据集
        self.models_info_list = []
        for data_root in self.data_roots:
            models_info = {}
            models_info_path = data_root / "models" / "models_info.json"
            if models_info_path.exists():
                with open(models_info_path, 'r', encoding='utf-8') as f:
                    models_info_raw = json.load(f)
                    # 转换键为整数
                    models_info = {int(k): v for k, v in models_info_raw.items()}
                print(f"  - [{data_root.name}] 已加载 {len(models_info)} 个物体模型信息")
            else:
                print(f"  - [{data_root.name}] Warning: models_info.json 未找到,将使用默认bbox尺寸")
            self.models_info_list.append(models_info)

        # 缓存
        self.coord_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        # Crop参数管理器（支持随机crop和参数复用）
        self.enable_random_crop = bool(bop_cfg.get("enable_random_crop", False))
        self.crop_param_manager = CropParamManager(enable_random=self.enable_random_crop)

        # 构建样本索引
        self.samples = self._build_index(bop_cfg)
        if len(self.samples) == 0:
            raise RuntimeError("未在指定目录中找到可用的 BOP 样本")

        print(f"[BopConsumerDataset] 索引完毕,共 {len(self.samples)} 个样本")
        if self.multi_dataset:
            print(f"  - 数据集模式: 多数据集 ({len(self.data_roots)} 个)")
            for idx, data_root in enumerate(self.data_roots):
                dataset_samples = [s for s in self.samples if s.dataset_root == data_root]
                print(f"    - {data_root.name}: {len(dataset_samples)} 个样本")
        else:
            print(f"  - 数据集模式: 单数据集")
            print(f"  - 数据根目录: {self.data_roots[0]}")
        print(f"  - 数据划分: {self.split}")
        print(f"  - 图像尺寸: {self.image_size}")
        print(f"  - VGGT 图像尺寸: {self.vggt_image_size}")

        self.tokenizer = tokenizer

        # Step 1: 获取所有类别名，去重，预处理
        category_file = 'cache/clutter_category_list.json'
        if not os.path.exists(category_file):
            category_all = set()
            for d in self.id2name_list:
                # 预处理每个value
                for v in d.values():
                    category_all.add(v.replace("_", " ").lower())
            self.category_all_list = list(category_all)
            self.category_all_set = category_all

            with open(category_file, 'w', encoding='utf-8') as fp:
                json.dump(self.category_all_list, fp, ensure_ascii=False, indent=4)
            print(f'Category list saved to clutter_category_list.json, Total {len(self.category_all_list)} categories')
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

    def _compute_gc6d_img_num(self, ann_id: int, camera: str) -> int:
        """根据标注ID和相机类型计算GraspCluster6D图像编号

        GraspClutter6D 图像编号规则：img_num = 4 * ann_id + camera_offset

        参数:
            ann_id: 标注ID (0-12)
            camera: 相机类型 (realsense-d415/realsense-d435/azure-kinect/zivid)

        返回:
            img_num: 图像编号
        """
        img_num = 4 * ann_id
        if camera == 'realsense-d415':
            img_num += 1
        elif camera == 'realsense-d435':
            img_num += 2
        elif camera == 'azure-kinect':
            img_num += 3
        elif camera == 'zivid':
            img_num += 4
        return img_num

    def _im_id_to_ann_id(self, im_id: int, camera: Optional[str] = None) -> int:
        """将 im_id 转换回 annId（GraspClutter6D 格式）

        GraspClutter6D 图像编号规则：img_num = 4 * ann_id + camera_offset
        因此：ann_id = (img_num - camera_offset) / 4

        参数:
            im_id: 图像ID
            camera: 相机类型（可选，默认使用 self.gc6d_camera）

        返回:
            annId: 标注ID (0-12)
        """
        if not self.use_graspclutter6d:
            return 0  # 标准 BOP 格式不需要 annId

        # 如果未指定相机，使用默认相机
        if camera is None:
            camera = self.gc6d_camera

        # 根据相机类型确定偏移量
        camera_offset = {
            'realsense-d415': 1,
            'realsense-d435': 2,
            'azure-kinect': 3,
            'zivid': 4,
        }.get(camera, 0)

        # 计算 annId
        ann_id = (im_id - camera_offset) // 4
        return ann_id

    def _discover_scenes(self, data_root: Path) -> List[int]:
        """发现指定数据集的所有可用场景"""
        # GraspClutter6D 格式：场景在 scenes/ 目录
        if self.use_graspclutter6d:
            split_path = data_root / "scenes"
        else:
            split_path = data_root / self.split

        if not split_path.exists():
            return []

        scene_dirs = sorted([
            d for d in split_path.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        return [int(d.name) for d in scene_dirs]

    def _build_index(self, bop_cfg) -> List[_SampleEntry]:
        """构建样本索引: 扫描所有数据集的所有场景的所有帧"""
        all_entries: List[_SampleEntry] = []

        # 遍历所有数据集
        for dataset_idx, data_root in enumerate(self.data_roots):
            dataset_name = data_root.name
            print(f"  - 正在索引数据集: {dataset_name}_{self.split}")

            # GraspClutter6D 格式：所有场景在 scenes/ 目录，无 train/test 子目录
            # 标准 BOP 格式：场景在 train/ 或 test/ 子目录
            if self.use_graspclutter6d:
                split_path = data_root / "scenes"
                if not split_path.exists():
                    print(f"    Warning: GraspClutter6D scenes 目录不存在: {split_path}")
                    continue
            else:
                split_path = data_root / self.split
                if not split_path.exists():
                    print(f"    Warning: 数据划分路径不存在: {split_path}")
                    continue

            # 检查是否使用缓存
            cache_path = bop_cfg.get("cache_index_path")
            use_cache = bool(bop_cfg.get("use_cache", False))

            # 如果是多数据集模式，为每个数据集创建独立的缓存文件
            if use_cache and cache_path and self.multi_dataset:
                cache_path_base, cache_ext = os.path.splitext(cache_path)
                cache_path = f"{cache_path_base}_{dataset_name}_{self.split}{cache_ext}"

            entries_from_cache = False
            if use_cache and cache_path and os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
                if payload.get("data_root") == str(data_root) and payload.get("split") == self.split:
                    # 从缓存加载时需要将字符串路径转换为 Path 对象
                    # 检查缓存版本是否包含 obj_id 字段
                    sample_list = payload.get("samples", [])
                    if sample_list and "obj_id" in sample_list[0]:
                        # 新版本缓存：包含 obj_id（可能包含 ann_id）
                        entries = [
                            _SampleEntry(
                                scene_id=item["scene_id"],
                                im_id=item["im_id"],
                                scene_dir=Path(item["scene_dir"]),
                                dataset_root=Path(item["dataset_root"]),
                                dataset_name=item["dataset_name"],
                                obj_id=item["obj_id"],
                                ann_id=item.get("ann_id", None)  # 兼容旧缓存
                            )
                            for item in sample_list
                        ]
                        all_entries.extend(entries)
                        entries_from_cache = True
                        print(f"    从缓存加载: {len(entries)} 个样本")
                    else:
                        # 旧版本缓存：不包含新字段，需要重新构建索引
                        print(f"    缓存版本过旧，重新构建索引")
                        entries_from_cache = False

            if not entries_from_cache:
                # 获取场景ID
                scene_ids = bop_cfg.get("scene_ids", None)
                if scene_ids is None:
                    # 如果使用 GraspClutter6D 格式，从 split_info 加载场景ID
                    if self.use_graspclutter6d:
                        # 处理 split="all" 的情况：合并训练集和测试集
                        if self.split == "all":
                            scene_ids = []
                            for split_type in ["train", "test"]:
                                split_info_path = data_root / "split_info" / f"grasp_{split_type}_scene_ids.json"
                                if split_info_path.exists():
                                    with open(split_info_path, 'r') as f:
                                        split_scene_ids = [int(x) for x in json.load(f)]
                                        scene_ids.extend(split_scene_ids)
                                        print(f"    从 GraspClutter6D split_info 加载了 {len(split_scene_ids)} 个 {split_type} 场景")
                            if not scene_ids:
                                print(f"    Warning: 无法从 split_info 加载场景，使用自动发现")
                                scene_ids = self._discover_scenes(data_root)
                        else:
                            # train 或 test 单独加载
                            split_info_path = data_root / "split_info" / f"grasp_{self.split}_scene_ids.json"
                            if split_info_path.exists():
                                with open(split_info_path, 'r') as f:
                                    scene_ids = [int(x) for x in json.load(f)]
                                print(f"    从 GraspClutter6D split_info 加载了 {len(scene_ids)} 个场景")
                            else:
                                print(f"    Warning: GraspClutter6D split_info 文件不存在: {split_info_path}")
                                scene_ids = self._discover_scenes(data_root)
                    else:
                        scene_ids = self._discover_scenes(data_root)

                entries: List[_SampleEntry] = []

                for scene_id in scene_ids:
                    scene_dir = split_path / f"{scene_id:06d}"
                    scene_camera_path = scene_dir / "scene_camera.json"

                    if not scene_camera_path.exists():
                        print(f"    Warning: scene_camera.json not found in {scene_dir}")
                        continue

                    # 加载 GT 数据（用于为每个物体实例创建 entry）
                    scene_gt_path = scene_dir / "scene_gt.json"
                    scene_gt_info_path = scene_dir / "scene_gt_info.json"

                    scene_gt = None
                    scene_gt_info = None
                    if scene_gt_path.exists() and scene_gt_info_path.exists():
                        scene_gt = inout.load_scene_gt(str(scene_gt_path))
                        scene_gt_info = inout.load_scene_gt(str(scene_gt_info_path))

                    # 根据数据集格式选择不同的索引方式
                    if self.use_graspclutter6d:
                        # GraspClutter6D 格式：每个场景有13个标注（ann_id=0-12）
                        # 每个标注对应4个相机的图像
                        # 注意：不同相机的可见性、bbox等GT信息可能不同
                        # 策略：只为在至少一个相机下满足过滤条件的物体类别创建entry
                        for ann_id in range(13):
                            # 检查所有相机，收集满足条件的物体ID
                            valid_obj_ids = set()  # 在至少一个相机下有效的物体ID集合

                            for camera in self.gc6d_cameras:
                                img_num = self._compute_gc6d_img_num(ann_id, camera)

                                if scene_gt is not None and img_num in scene_gt:
                                    gt_info_list = scene_gt_info[img_num]
                                    gt_list = scene_gt[img_num]

                                    for gt_idx, (gt, gt_info) in enumerate(zip(gt_list, gt_info_list)):
                                        obj_id = gt['obj_id']

                                        # 应用过滤条件
                                        visib_fract = gt_info.get('visib_fract', 0)
                                        if visib_fract < 0.3:
                                            continue

                                        # 此物体类别在当前相机下有效
                                        valid_obj_ids.add(obj_id)

                            # 为每个有效的物体类别创建一个entry（去重）
                            # 使用第一个相机的img_num作为占位符（加载时会重新计算）
                            img_num = self._compute_gc6d_img_num(ann_id, self.gc6d_cameras[0])

                            for obj_id in valid_obj_ids:
                                entries.append(_SampleEntry(
                                    scene_id=scene_id,
                                    im_id=img_num,
                                    scene_dir=scene_dir,
                                    dataset_root=data_root,
                                    dataset_name=dataset_name,
                                    obj_id=obj_id,
                                    ann_id=ann_id  # 存储标注ID用于随机相机选择
                                ))
                    else:
                        # 标准 BOP 格式：加载场景相机参数获取所有图像ID
                        scene_camera = inout.load_scene_camera(str(scene_camera_path))
                        im_ids = sorted(scene_camera.keys())

                        # 添加到样本列表
                        for im_id in im_ids:
                            # 为该图像的每个物体类别创建一个 entry（去重）
                            if scene_gt is not None and im_id in scene_gt:
                                gt_list = scene_gt[im_id]
                                gt_info_list = scene_gt_info[im_id]

                                # 收集有效的物体ID（去重）
                                valid_obj_ids = set()
                                for gt_idx, (gt, gt_info) in enumerate(zip(gt_list, gt_info_list)):
                                    obj_id = gt['obj_id']

                                    # 应用过滤条件（与 _load_sample 一致）
                                    visib_fract = gt_info.get('visib_fract', 0)
                                    if visib_fract < 0.3:
                                        continue

                                    # 从 mask_visib 计算 bbox 而不是从 gt_info 获取
                                    mask_visib_path = scene_dir / 'mask_visib' / f"{im_id:06d}_{gt_idx:06d}.png"
                                    bbox_obj = _compute_bbox_from_mask(mask_visib_path)
                                    bbox_area = bbox_obj[2] * bbox_obj[3]
                                    if bbox_area < 150:
                                        continue

                                    valid_obj_ids.add(obj_id)

                                # 为每个有效的物体类别创建一个 entry
                                for obj_id in valid_obj_ids:
                                    entries.append(_SampleEntry(
                                        scene_id=scene_id,
                                        im_id=im_id,
                                        scene_dir=scene_dir,
                                        dataset_root=data_root,
                                        dataset_name=dataset_name,
                                        obj_id=obj_id
                                    ))

                all_entries.extend(entries)
                print(f"    索引完成: {len(entries)} 个样本")

                # 保存缓存
                if use_cache and cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    # 使用临时文件 + 原子重命名，避免损坏
                    temp_fd, temp_path = tempfile.mkstemp(
                        dir=os.path.dirname(cache_path),
                        prefix=".tmp_cache_",
                        suffix=".json"
                    )
                    os.close(temp_fd)  # 关闭这个 file descriptor，释放占用
                    try:
                        with open(temp_path, "w", encoding="utf-8") as fp:
                            json.dump(
                                {
                                    "data_root": str(data_root),
                                    "split": self.split,
                                    "samples": [{
                                        "scene_id": e.scene_id,
                                        "im_id": e.im_id,
                                        "scene_dir": str(e.scene_dir),
                                        "dataset_root": str(e.dataset_root),
                                        "dataset_name": e.dataset_name,
                                        "obj_id": e.obj_id,
                                        "ann_id": e.ann_id  # 保存 ann_id (GraspClutter6D)
                                    } for e in entries],
                                },
                                fp,
                                ensure_ascii=False,
                            )
                        # 原子重命名，确保完整性
                        shutil.move(temp_path, cache_path)
                    except Exception as e:
                        # 清理临时文件
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                        raise

        # 限制样本数量（应用于所有数据集）
        if self.max_samples:
            all_entries = all_entries[: int(self.max_samples)]

        return all_entries

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取单个样本,带重试机制"""
        for _ in range(self.sample_retries):
            entry = self.samples[index % len(self.samples)]
            sample = self._load_sample(entry, sample_idx=index)
            if sample is not None:
                return sample
            index = torch.randint(0, len(self.samples), ()).item()
        raise RuntimeError("多次尝试后仍无法加载有效样本,请检查数据完整性")

    def _load_mask(self, mask_path: Path) -> Optional[np.ndarray]:
        """加载mask图像,自动处理不同格式"""
        if not mask_path.exists():
            return None

        mask = inout.load_im(str(mask_path))

        # 如果是3通道图像,取第一个通道
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        return mask

    def _load_sample(self, entry: _SampleEntry, sample_idx: int) -> Optional[Dict[str, Any]]:
        """加载单个样本的具体实现

        参数:
            entry: 样本条目
            sample_idx: 样本索引（用于crop参数缓存管理）
        """
        scene_dir = entry.scene_dir
        im_id = entry.im_id
        scene_id = entry.scene_id
        dataset_root = entry.dataset_root
        dataset_name = entry.dataset_name
        target_obj_id = entry.obj_id  # 目标物体类别ID

        # GraspClutter6D: 如果有多个相机，随机选择一个并重新计算 im_id
        selected_camera = None
        if self.use_graspclutter6d and entry.ann_id is not None:
            if len(self.gc6d_cameras) > 1:
                # 随机选择一个相机
                selected_camera = random.choice(self.gc6d_cameras)
                # 根据 ann_id 和选择的相机重新计算 im_id
                im_id = self._compute_gc6d_img_num(entry.ann_id, selected_camera)
            else:
                selected_camera = self.gc6d_cameras[0]

        # 根据数据集来源选择对应的 models_info 和 id2name
        # 确保 dataset_root 是 Path 对象，以匹配 self.data_roots 中的类型
        if not isinstance(dataset_root, Path):
            dataset_root = Path(dataset_root)

        try:
            dataset_idx = self.data_roots.index(dataset_root)
        except ValueError:
            # 如果找不到，尝试通过字符串比较查找
            dataset_idx = None
            for idx, root in enumerate(self.data_roots):
                if str(root) == str(dataset_root):
                    dataset_idx = idx
                    break
            if dataset_idx is None:
                print(f"Warning: 找不到数据集根目录 {dataset_root}")
                return None

        models_info = self.models_info_list[dataset_idx]
        id2name = self.id2name_list[dataset_idx]

        # 加载RGB图像
        rgb_path = scene_dir / 'rgb' / f"{im_id:06d}.png"
        if not rgb_path.exists():
            rgb_path = scene_dir / 'rgb' / f"{im_id:06d}.jpg"
            if not rgb_path.exists():
                return None

        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        if rgb is None or rgb.size == 0:
            print(f"{rgb_path}错误,返回None")
            return None

        # 加载深度图
        depth = None
        depth_mask = None
        if self.load_depth:
            depth_path = scene_dir / 'depth' / f"{im_id:06d}.png"
            if depth_path.exists():
                try:
                    depth_raw = inout.load_depth(str(depth_path)).astype(np.float32)
                    # GraspClutter6D 深度单位转换：不同相机类型有不同的单位
                    if self.use_graspclutter6d:
                        # 使用选择的相机（如果有）或默认相机
                        camera_for_depth = selected_camera if selected_camera else self.gc6d_camera
                        if camera_for_depth in ['realsense-d415', 'realsense-d435']:
                            depth_scale = 1000.0  # mm -> m
                        elif camera_for_depth in ['azure-kinect', 'zivid']:
                            depth_scale = 10000.0  # 0.1mm -> m
                        else:
                            depth_scale = self.unit_scale  # 回退到配置值
                    else:
                        # 标准 BOP 格式使用配置的 unit_scale
                        depth_scale = self.unit_scale

                    depth_raw = depth_raw / depth_scale
                    depth = _threshold_depth_map(
                        depth_raw,
                        max_percentile=self.depth_max_percentile,
                        min_percentile=self.depth_min_percentile,
                        max_depth=self.depth_max_value,
                    )
                    depth_mask = (depth > 0).astype(np.uint8)
                except Exception as e:
                    print(e)
                    print(f"{depth_path}错误, 创建全为 0 的 depth")
                    # depth_path 不存在时，创建全为 0 的 depth
                    depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
                    depth_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            else:
                # depth_path 不存在时，创建全为 0 的 depth
                depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
                depth_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

        # 加载相机参数
        scene_camera_path = scene_dir / "scene_camera.json"
        scene_camera = inout.load_scene_camera(str(scene_camera_path))

        if im_id not in scene_camera:
            return None

        cam_info = scene_camera[im_id]
        cam_K = np.array(cam_info['cam_K']).reshape(3, 3)

        # 加载GT (物体姿态和bbox)
        objects_data = []
        if self.load_gt:
            scene_gt_path = scene_dir / "scene_gt.json"
            scene_gt_info_path = scene_dir / "scene_gt_info.json"

            if scene_gt_path.exists() and scene_gt_info_path.exists():
                scene_gt = inout.load_scene_gt(str(scene_gt_path))
                scene_gt_info = inout.load_scene_gt(str(scene_gt_info_path))

                if im_id in scene_gt:
                    gt_list = scene_gt[im_id]
                    gt_info_list = scene_gt_info[im_id]

                    for gt_idx, (gt, gt_info) in enumerate(zip(gt_list, gt_info_list)):
                        obj_id = gt['obj_id']

                        # 过滤1: 可见度太低的物体（与官方BOP toolkit一致）
                        visib_fract = gt_info.get('visib_fract', 0)
                        if visib_fract < 0.3:
                            continue

                        # 2D bbox: 从 mask_visib 计算而不是从 gt_info 获取
                        mask_visib_path = scene_dir / 'mask_visib' / f"{im_id:06d}_{gt_idx:06d}.png"
                        bbox_obj = _compute_bbox_from_mask(mask_visib_path)

                        # 过滤2: bbox面积太小的物体（与官方BOP toolkit一致）
                        bbox_area = bbox_obj[2] * bbox_obj[3]  # width * height
                        if bbox_area < 150:
                            continue

                        # 旋转和平移
                        cam_R_m2c = np.array(gt['cam_R_m2c']).reshape(3, 3)

                        # GraspClutter6D 姿态单位转换：统一使用 mm->m (1000.0)，与相机类型无关
                        # 注意：深度单位与姿态单位不同！
                        if self.use_graspclutter6d:
                            pose_scale = 1000.0  # mm -> m (所有相机统一)
                        else:
                            pose_scale = self.unit_scale

                        cam_t_m2c = np.array(gt['cam_t_m2c']).reshape(3) / pose_scale

                        objects_data.append({
                            'obj_id': obj_id,
                            'cam_R_m2c': cam_R_m2c,
                            'cam_t_m2c': cam_t_m2c,
                            'bbox_obj': bbox_obj,
                            'visib_fract': visib_fract,
                            'gt_idx': gt_idx,  # 记录在GT列表中的原始索引
                        })

        if len(objects_data) == 0:
            return None

        # 使用 entry 中的 obj_id 作为主要检测目标（确定性选择）
        obj_id = target_obj_id

        # 过滤出与目标物体相同类别的所有实例（与 dataset_omni6d.py 一致）
        same_class_objects = [obj for obj in objects_data if obj['obj_id'] == obj_id]

        # 验证目标物体是否在过滤后的列表中
        if len(same_class_objects) == 0:
            return None

        # ============================================================
        # 数据处理流程 (多摄像头独立crop):
        # 1. 为N个摄像头生成不同crop的图像（根据配置）
        # 2. 原始RGB/depth → crop + resize + pad → 518x518 (vggt)
        # 3. vggt_518 → resize + pad → 224x224
        # 4. 基于224分辨率生成所有标注: bbox, rays, K等
        # ============================================================

        camera_configs = self.camera_configs

        # 为每个摄像头生成不同crop的图像
        camera_images_224 = []
        camera_images_518 = []
        camera_depths_224 = []
        camera_valid_masks_224 = []
        camera_intrinsics = []

        for cam_idx, cam_config in enumerate(camera_configs):
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
                debug=True,  # 使用中心crop（但会被preset_crop_params覆盖）
                preset_crop_params=preset_crop_params_518
            )

            # 深度图也要用相同的crop参数
            if depth is not None:
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
            else:
                depth_518_cam = np.zeros((self.vggt_image_size, self.vggt_image_size), dtype=np.float32)
                depth_mask_518_cam = np.zeros((self.vggt_image_size, self.vggt_image_size), dtype=np.uint8)

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

            # 调整相机内参 (原始 → crop → 518 → 224)
            fx_orig = cam_K[0, 0]
            fy_orig = cam_K[1, 1]
            cx_orig = cam_K[0, 2]
            cy_orig = cam_K[1, 2]

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

        # 将每个摄像头的518图像转换为tensor列表
        images_vggt_list = [
            torch.from_numpy(rgb_518).permute(2, 0, 1).contiguous().float() / 255.0
            for rgb_518 in camera_images_518
        ]

        # 使用top_head (第0个摄像头) 的数据作为主要数据
        rgb_224 = camera_images_224[0]
        depth_224 = camera_depths_224[0]
        depth_mask_224 = (camera_depths_224[0] > 0).astype(np.uint8)
        valid_mask_224 = camera_valid_masks_224[0]

        # 使用top_head的内参
        fx, fy, cx, cy = camera_intrinsics[0]
        K = torch.from_numpy(_intrinsics_to_matrix(fx, fy, cx, cy))

        # 转换为tensor
        rgb_tensor = torch.from_numpy(rgb_224).permute(2, 0, 1).contiguous()
        depth_tensor = torch.from_numpy(depth_224).unsqueeze(0).float()
        depth_mask_tensor = torch.from_numpy(depth_mask_224).unsqueeze(0)
        img_mask_tensor = torch.from_numpy(valid_mask_224).unsqueeze(0)

        # 生成坐标网格
        coord_map = self.coord_cache.get((self.image_size, self.image_size))
        if coord_map is None:
            coord_map = _generate_coord_grid(self.image_size, self.image_size)
            self.coord_cache[(self.image_size, self.image_size)] = coord_map
        coord_map = coord_map.to(torch.device("cpu"))

        # 为每个摄像头生成3D感知数据（rays, pcl, know_depth等）
        camera_rays_list = []
        camera_pcl_list = []
        camera_know_depth_list = []
        camera_depth_tensors = []
        camera_depth_mask_tensors = []
        camera_K_list = []

        # 用top_head的深度来计算avg_scale（统一尺度）
        top_depth_tensor = torch.from_numpy(camera_depths_224[0]).unsqueeze(0).float()
        top_depth_mask_tensor = torch.from_numpy((camera_depths_224[0] > 0).astype(np.uint8)).unsqueeze(0)
        top_K = torch.from_numpy(_intrinsics_to_matrix(*camera_intrinsics[0]))

        top_pcl = _depth_to_pcl(top_depth_tensor.squeeze(0), top_K, coord_map, top_depth_mask_tensor.squeeze(0))
        if top_depth_mask_tensor.sum() > 0:
            dist = torch.linalg.norm(top_pcl, dim=-1)
            # avg_scale = dist[top_depth_mask_tensor.squeeze(0) > 0].mean().clamp(min=1e-3, max=1e3)
            if self.config.max_norm_6d_dataset:
                avg_scale = dist[top_depth_mask_tensor.squeeze(0) > 0].max().clamp(min=1e-3, max=1e3)
            else:
                avg_scale = torch.tensor(1.0)

        # 为每个摄像头生成对应的3D数据
        for cam_idx in range(len(camera_images_224)):
            cam_depth_224 = camera_depths_224[cam_idx]
            cam_valid_mask_224 = camera_valid_masks_224[cam_idx]
            cam_fx, cam_fy, cam_cx, cam_cy = camera_intrinsics[cam_idx]

            # 构建K矩阵
            cam_K = torch.from_numpy(_intrinsics_to_matrix(cam_fx, cam_fy, cam_cx, cam_cy))
            camera_K_list.append(cam_K)

            # 深度张量
            cam_depth_tensor = torch.from_numpy(cam_depth_224).unsqueeze(0).float()
            cam_depth_mask_tensor = torch.from_numpy((cam_depth_224 > 0).astype(np.uint8)).unsqueeze(0)
            cam_img_mask_tensor = torch.from_numpy(cam_valid_mask_224).unsqueeze(0)

            # 生成点云
            cam_pcl = _depth_to_pcl(cam_depth_tensor.squeeze(0), cam_K, coord_map, cam_depth_mask_tensor.squeeze(0))

            # 生成rays
            cam_rays = _generate_rays(self.image_size, self.image_size, cam_K, cam_img_mask_tensor.squeeze(0).bool())

            # 生成稀疏深度
            cam_know_depth = torch.cat([cam_depth_tensor, cam_depth_mask_tensor], dim=0) #(2,h,w)

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

        if random.random() > self.bop_cfg.depth_full_prob:
            camera_know_depth_list = [torch.zeros_like(kd) for kd in camera_know_depth_list]

        if random.random() > self.bop_cfg.ray_prob:
            camera_rays_list = [torch.zeros_like(r) for r in camera_rays_list]

        # 向后兼容：使用top_head（第0个相机）的数据
        rays = camera_rays_list[0]
        depth_tensor = camera_depth_tensors[0]
        depth_mask_tensor = camera_depth_mask_tensors[0]
        pcl = camera_pcl_list[0]
        know_depth = camera_know_depth_list[0]

        # 为每个摄像头独立计算2D bbox和过滤物体（与DetConsumerDataset一致）
        camera_bboxes_all = []
        camera_pose_list = []  # [num_cameras] 每个元素是该摄像头的pose列表
        camera_bbox_side_len_list = []  # [num_cameras] 每个元素是该摄像头的bbox_side_len列表
        camera_class_labels = []  # [num_cameras] 每个元素是该摄像头的class_label列表
        camera_visibility_list = []  # [num_cameras] 每个元素是该摄像头的visibility列表

        # 为每个摄像头记录通过过滤的物体索引，用于后续分别加载 grasps
        camera_filtered_objects_indices = {}  # {cam_idx: [obj_idx1, obj_idx2, ...]}

        for cam_idx, cam_config in enumerate(camera_configs):
            # 使用与图像处理相同的crop参数（从缓存中获取，确保完全一致）
            crop_params_518 = self.crop_param_manager.get_or_create_params(
                sample_idx=sample_idx,
                cam_idx=cam_idx,
                size_key="518",
                orig_width=rgb.shape[1],
                orig_height=rgb.shape[0],
                crop_scale=cam_config["crop_scale"],
                aspect_ratio=cam_config["aspect_ratio"]
            )

            preset_crop_params_518 = (
                crop_params_518.crop_width,
                crop_params_518.crop_height,
                crop_params_518.crop_left,
                crop_params_518.crop_top
            )

            _, _, resize_ratio_518_cam, pad_l_518_cam, pad_t_518_cam, crop_l_518_cam, crop_t_518_cam = _crop_resize_with_pad(
                rgb, self.vggt_image_size, self.vggt_image_size,
                pad_value=0, interpolation=cv2.INTER_LINEAR,
                crop_scale=cam_config["crop_scale"],
                aspect_ratio=cam_config["aspect_ratio"],
                debug=True,
                preset_crop_params=preset_crop_params_518
            )

            # 计算518->224的参数
            dummy_518 = np.zeros((self.vggt_image_size, self.vggt_image_size, 3), dtype=np.uint8)
            _, _, resize_ratio_224_cam, pad_l_224_cam, pad_t_224_cam = _resize_with_pad(
                dummy_518, self.image_size, self.image_size, pad_value=0, interpolation=cv2.INTER_LINEAR
            )

            # 为当前摄像头过滤并处理物体
            cam_bbox_list = []
            cam_pose_list = []
            cam_bbox_side_len_list = []
            cam_class_label_list = []
            cam_visibility_list = []  # 存储每个物体的可见性
            cam_valid_mask_224 = camera_valid_masks_224[cam_idx]

            # 计算有效区域边界（排除pad区域）
            valid_rows, valid_cols = np.where(cam_valid_mask_224 > 0)
            if len(valid_rows) > 0:
                valid_x_min = float(valid_cols.min())
                valid_y_min = float(valid_rows.min())
                valid_x_max = float(valid_cols.max() + 1)  # +1 因为是exclusive边界
                valid_y_max = float(valid_rows.max() + 1)
            else:
                # 如果没有有效区域，使用整个图像范围（回退方案）
                valid_x_min = 0.0
                valid_y_min = 0.0
                valid_x_max = float(self.image_size)
                valid_y_max = float(self.image_size)

            # 初始化当前摄像头的过滤索引列表
            camera_filtered_objects_indices[cam_idx] = []

            for obj_idx, obj_data in enumerate(same_class_objects):
                gt_idx = obj_data['gt_idx']
                bbox_orig = np.array(obj_data['bbox_obj'])  # [x, y, w, h]

                # 加载该物体的 visible_mask 和 amodal_mask（用于可见性过滤）
                visible_mask_path = scene_dir / 'mask_visib' / f"{im_id:06d}_{gt_idx:06d}.png"
                amodal_mask_path = scene_dir / 'mask' / f"{im_id:06d}_{gt_idx:06d}.png"

                # 加载masks（直接加载，不使用缓存）
                # 注意：运行时缓存已禁用，避免多进程内存浪费
                visible_mask_orig = cv2.imread(str(visible_mask_path), cv2.IMREAD_GRAYSCALE)
                amodal_mask_orig = cv2.imread(str(amodal_mask_path), cv2.IMREAD_GRAYSCALE)

                # 如果mask加载失败，跳过该物体
                if visible_mask_orig is None or amodal_mask_orig is None:
                    continue

                # 计算visibility: visible_mask考虑截断（crop内可见像素），amodal_mask不考虑截断（完整轮廓）
                # 1. 分母：使用原始分辨率的amodal_mask（物体完整轮廓）
                amodal_mask_sum_orig = np.sum(amodal_mask_orig > 0)

                # 2. 分子：计算visible_mask在crop区域内的像素数
                if amodal_mask_sum_orig > 0:
                    # 获取crop区域的边界（在原始分辨率上）
                    # crop_left和crop_top已经是原始图像的像素坐标
                    H_orig, W_orig = visible_mask_orig.shape[:2]
                    crop_t_orig = crop_t_518_cam
                    crop_l_orig = crop_l_518_cam
                    crop_b_orig = crop_t_518_cam + crop_params_518.crop_height
                    crop_r_orig = crop_l_518_cam + crop_params_518.crop_width

                    # 确保crop边界在有效范围内
                    crop_t_orig = max(0, min(crop_t_orig, H_orig))
                    crop_l_orig = max(0, min(crop_l_orig, W_orig))
                    crop_b_orig = max(0, min(crop_b_orig, H_orig))
                    crop_r_orig = max(0, min(crop_r_orig, W_orig))

                    # 提取crop区域内的visible_mask
                    visible_mask_in_crop = visible_mask_orig[crop_t_orig:crop_b_orig, crop_l_orig:crop_r_orig]
                    visible_mask_sum_in_crop = np.sum(visible_mask_in_crop > 0)

                    # visibility = crop内可见像素 / 完整轮廓像素
                    visibility = visible_mask_sum_in_crop / amodal_mask_sum_orig
                else:
                    visibility = 0.0

                # 对masks应用与图像相同的crop变换（crop → resize+pad → 224）
                visible_mask_518, _, _, _, _, _, _ = _crop_resize_with_pad(
                    visible_mask_orig,
                    self.vggt_image_size,
                    self.vggt_image_size,
                    pad_value=0,
                    interpolation=cv2.INTER_NEAREST,
                    crop_scale=cam_config["crop_scale"],
                    aspect_ratio=cam_config["aspect_ratio"],
                    debug=True,
                    preset_crop_params=preset_crop_params_518
                )

                amodal_mask_518, _, _, _, _, _, _ = _crop_resize_with_pad(
                    amodal_mask_orig,
                    self.vggt_image_size,
                    self.vggt_image_size,
                    pad_value=0,
                    interpolation=cv2.INTER_NEAREST,
                    crop_scale=cam_config["crop_scale"],
                    aspect_ratio=cam_config["aspect_ratio"],
                    debug=True,
                    preset_crop_params=preset_crop_params_518
                )

                # 518 → 224
                visible_mask_224, _, _, _, _ = _resize_with_pad(
                    visible_mask_518,
                    self.image_size,
                    self.image_size,
                    pad_value=0,
                    interpolation=cv2.INTER_NEAREST
                )

                amodal_mask_224, _, _, _, _ = _resize_with_pad(
                    amodal_mask_518,
                    self.image_size,
                    self.image_size,
                    pad_value=0,
                    interpolation=cv2.INTER_NEAREST
                )

                # 原始 → crop
                x1_after_crop = bbox_orig[0] - crop_l_518_cam
                y1_after_crop = bbox_orig[1] - crop_t_518_cam
                x2_after_crop = x1_after_crop + bbox_orig[2]
                y2_after_crop = y1_after_crop + bbox_orig[3]

                # crop → 518
                x1_518 = x1_after_crop / resize_ratio_518_cam + pad_l_518_cam
                y1_518 = y1_after_crop / resize_ratio_518_cam + pad_t_518_cam
                x2_518 = x2_after_crop / resize_ratio_518_cam + pad_l_518_cam
                y2_518 = y2_after_crop / resize_ratio_518_cam + pad_t_518_cam

                # 518 → 224
                x1 = x1_518 / resize_ratio_224_cam + pad_l_224_cam
                y1 = y1_518 / resize_ratio_224_cam + pad_t_224_cam
                x2 = x2_518 / resize_ratio_224_cam + pad_l_224_cam
                y2 = y2_518 / resize_ratio_224_cam + pad_t_224_cam

                # 裁剪到有效区域内（排除pad区域）
                x1 = max(valid_x_min, min(x1, valid_x_max))
                y1 = max(valid_y_min, min(y1, valid_y_max))
                x2 = max(valid_x_min, min(x2, valid_x_max))
                y2 = max(valid_y_min, min(y2, valid_y_max))

                # 过滤3: 变换后的bbox面积检查（与官方BOP toolkit第919行一致）
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area_transformed = bbox_width * bbox_height
                if bbox_area_transformed < 200 or bbox_width < 10 or bbox_height < 10:
                    continue

                # 过滤4: visibility过滤
                # visible_mask考虑截断（crop内可见），amodal_mask不考虑截断（完整轮廓）
                # visibility = crop内可见像素数 / 完整轮廓像素数
                if visibility < 0.4 or np.isnan(visibility):
                    continue

                # 该物体在此摄像头上有效，添加到列表
                cam_bbox_list.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))

                # 记录该物体索引到当前摄像头的过滤列表中
                camera_filtered_objects_indices[cam_idx].append(obj_idx)

                # 计算pose
                rot = R.from_matrix(obj_data['cam_R_m2c'])
                quat_xyzw = rot.as_quat()
                quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                trans = obj_data['cam_t_m2c'] / avg_scale.item()
                pose = np.concatenate([quat_wxyz, trans])
                cam_pose_list.append(pose)

                # 计算3D bbox尺寸
                if obj_data['obj_id'] in models_info:
                    model_info = models_info[obj_data['obj_id']]
                    if 'size_x' in model_info and 'size_y' in model_info and 'size_z' in model_info:
                        bbox_side_len = np.array([
                            model_info['size_x'] / 1000.0,
                            model_info['size_y'] / 1000.0,
                            model_info['size_z'] / 1000.0
                        ]) / avg_scale.item()
                    elif 'diameter' in model_info:
                        diam = model_info['diameter'] / 1000.0
                        bbox_side_len = np.array([diam, diam, diam]) / avg_scale.item()
                    else:
                        bbox_side_len = np.array([0.1, 0.1, 0.1]) / avg_scale.item()
                else:
                    bbox_side_len = np.array([0.1, 0.1, 0.1]) / avg_scale.item()
                cam_bbox_side_len_list.append(bbox_side_len)

                cam_class_label_list.append(obj_id)

                # 存储该物体的可见性（crop内可见像素 / 完整轮廓像素）
                cam_visibility_list.append(visibility)

            camera_bboxes_all.append(cam_bbox_list)
            camera_pose_list.append(cam_pose_list)
            camera_bbox_side_len_list.append(cam_bbox_side_len_list)
            camera_class_labels.append(cam_class_label_list)
            camera_visibility_list.append(cam_visibility_list)

        # 转换为tensor（每个摄像头独立，与DetConsumerDataset一致）
        camera_bbox_side_len_tensors = []
        camera_pose_tensors = []
        for cam_idx in range(self.num_cameras):
            if camera_bbox_side_len_list[cam_idx]:  # 如果该相机有可见物体
                cam_bbox_side_len = torch.tensor(np.array(camera_bbox_side_len_list[cam_idx]), dtype=torch.float32)
                cam_pose = torch.tensor(np.array(camera_pose_list[cam_idx]), dtype=torch.float32)
            else:  # 如果该相机没有可见物体，创建空张量
                cam_bbox_side_len = torch.zeros((0, 3), dtype=torch.float32)
                cam_pose = torch.zeros((0, 7), dtype=torch.float32)
            camera_bbox_side_len_tensors.append(cam_bbox_side_len)
            camera_pose_tensors.append(cam_pose)

        # 向后兼容：使用top_head（第0个相机）的3D标注
        bbox_side_len_tensor = camera_bbox_side_len_tensors[0] if camera_bbox_side_len_tensors[0].numel() > 0 else torch.zeros((0, 3), dtype=torch.float32)
        pose_tensor = camera_pose_tensors[0] if camera_pose_tensors[0].numel() > 0 else torch.zeros((0, 7), dtype=torch.float32)

        # 为每个摄像头生成对应的图像和深度张量（同一时间的多视角）
        images: List[torch.Tensor] = [
            torch.from_numpy(camera_images_224[cam_idx]).permute(2, 0, 1).contiguous()
            for cam_idx in range(self.num_cameras)
        ]
        depths: List[torch.Tensor] = [
            torch.from_numpy(camera_depths_224[cam_idx]).unsqueeze(0).float()
            for cam_idx in range(self.num_cameras)
        ]

        # 组装样本
        class_name = id2name.get(obj_id, f"object_{obj_id}")

        # camera_bboxes, camera_pose_tensors, camera_bbox_side_len_tensors, camera_class_labels
        # 已在上面独立为每个摄像头计算和过滤

        # 创建空的mask（BOP数据集不提供mask）
        camera_masks = [[] for _ in range(self.num_cameras)]
        mask_category_level = torch.zeros((1, self.image_size, self.image_size), dtype=torch.uint8)
        mask_instance_level = torch.zeros((0, 1, self.image_size, self.image_size), dtype=torch.uint8)

        class_name_clean = class_name.replace("_", " ").lower()
        use_grasp = (random.random() < self.bop_cfg.use_grasp_prob)
        if not use_grasp:
            task = f"detect the {class_name_clean}"
        else:
            task = f"grasp the {class_name_clean} with no collision"

        if class_name_clean not in self.category_all_set:
            print(class_name_clean, 'clutter', '==================')
            return None

        # 10%负样本
        if random.random() < self.bop_cfg.neg_prob and len(self.category_full_set) > 1:
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
                text_label = map_3d_label_to_string(class_name_clean, images, camera_bboxes_all, camera_pose_tensors, camera_bbox_side_len_tensors)
            else:
                text_label = map_3d_label_to_string_tokenizer(class_name_clean, images, camera_bboxes_all, camera_pose_tensors, camera_bbox_side_len_tensors, self.tokenizer)


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
            "task":        task,
            "text_label":  text_label,

            # ===== 多相机 2D / 3D 标注 =====
            "bbox_2d":              camera_bboxes_all,              # List[List[Tensor(4,)]]
            "camera_pose":          camera_pose_tensors,            # List[Tensor(N,7)]
            "camera_bbox_side_len": camera_bbox_side_len_tensors,   # List[Tensor(N,3)]
            "camera_class_labels":  camera_class_labels,            # List[List[int]]
        }

        # GraspClutter6D 抓取数据加载（为每个摄像头分别加载）
        if self.load_grasps and self.use_graspclutter6d and GRASPCLUTTER6D_AVAILABLE:
            try:
                # 为每个摄像头加载对应物体的 grasp
                camera_grasps = []  # List[List[GraspGroup]]，外层是摄像头，内层是物体实例

                for cam_idx in range(self.num_cameras):
                    # 获取该摄像头过滤后的物体列表
                    cam_filtered_indices = camera_filtered_objects_indices.get(cam_idx, [])
                    if cam_filtered_indices:
                        cam_objects = [same_class_objects[i] for i in cam_filtered_indices]

                        # 获取该摄像头的 crop 参数
                        cam_config = self.camera_configs[cam_idx]
                        crop_params_for_cam = self.crop_param_manager.get_or_create_params(
                            sample_idx=sample_idx,
                            cam_idx=cam_idx,
                            size_key="518",
                            orig_width=rgb.shape[1],
                            orig_height=rgb.shape[0],
                            crop_scale=cam_config["crop_scale"],
                            aspect_ratio=cam_config["aspect_ratio"]
                        )

                        # 加载 grasp 数据，传入对应的 crop 参数
                        cam_grasp_data = self._load_grasps(
                            scene_id=scene_id,
                            im_id=im_id,
                            scene_dir=scene_dir,
                            objects_data=cam_objects,
                            scene_cloud=None,  # scene_cloud 会在 _load_grasps 内部加载
                            camera=selected_camera,  # 传递选择的相机类型
                            crop_params=crop_params_for_cam,  # 传入 crop 参数
                            orig_img_shape=(rgb.shape[0], rgb.shape[1])  # 原始图像尺寸
                        )
                        camera_grasps.append(cam_grasp_data if cam_grasp_data is not None else [])
                    else:
                        # 该摄像头没有可见物体
                        camera_grasps.append([])

                # 多摄像头 grasp 数据（与 camera_bbox_side_len, camera_pose 对应）
                camera_grasps_quat_trans = []
                camera_grasps_box = []
                for each_view_grasp in camera_grasps:
                    # 当前view无grasp
                    if len(each_view_grasp) == 0:
                        camera_grasps_quat_trans.append(np.zeros((0, 7)))
                        continue

                    # each_view_grasp: list of grasp objects
                    camera_grasps_each_view = []
                    camera_grasps_box_each_view = []

                    for grasp in each_view_grasp:
                        if grasp is None:
                            grasp_pose = np.full((1, 7), 0.)  # 用 0 占位
                            grasp_box = np.full((1, 3), 0.)  # 用 0 占位
                        else:
                            grasp_rot = grasp.rotation_matrices  # (top-1,3,3)
                            grasp_trans = grasp.translations  # (top-1,3)
                            grasp_pose = remap_pose(grasp_trans, grasp_rot)
                            grasp_box = np.stack([grasp.widths, grasp.heights, grasp.depths], axis=1) #(N,3)
                        camera_grasps_each_view.append(grasp_pose[0])  # append (7,)
                        camera_grasps_box_each_view.append(grasp_box[0]) # append (3,)
                        # 合并各 grasp 对象得到 (num_grasp * top-1, 7)
                    camera_grasps_each_view = np.stack(camera_grasps_each_view, axis=0)  # (N, 7)
                    camera_grasps_each_view = torch.from_numpy(camera_grasps_each_view)
                    camera_grasps_box_each_view = np.stack(camera_grasps_box_each_view, axis=0) # (N,3)
                    camera_grasps_box_each_view = torch.from_numpy(camera_grasps_box_each_view)

                    camera_grasps_quat_trans.append(camera_grasps_each_view)  # 可以是list of array，每个视角(N,7)
                    camera_grasps_box.append(camera_grasps_box_each_view)

            except Exception as e:
                print(f"Warning: 加载抓取数据失败 (scene_id={scene_id}, im_id={im_id}): {e}")
                import traceback
                traceback.print_exc()
                camera_grasps_quat_trans = [np.zeros((0, 7)) for _ in range(self.num_cameras)]
                camera_grasps_box = [torch.zeros((0, 3)) for _ in range(self.num_cameras)]

            if use_grasp:
                #TODO: grasp now not support negative samples
                sample["camera_pose"] = camera_grasps_quat_trans
                sample["camera_bbox_side_len"] = camera_grasps_box
                if self.config.uniform_mapping_6d_dataset:
                    text_label = map_3d_label_to_string(class_name_clean, images, camera_bboxes_all, camera_grasps_quat_trans, camera_grasps_box)
                else:
                    text_label = map_3d_label_to_string_tokenizer(class_name_clean, images, camera_bboxes_all, camera_grasps_quat_trans, camera_grasps_box, self.tokenizer)
                sample["text_label"] = text_label

        self.crop_param_manager.clear_cache()

        return sample

    def _load_preprocessed_grasps(
        self,
        scene_id: int,
        im_id: int,
        objects_data: List[Dict],
        camera: Optional[str] = None
    ) -> Optional[List[GraspGroup]]:
        """加载预处理的抓取数据（离线过滤后的）

        预处理文件已包含：
        - 展开后的grasp候选
        - 摩擦系数和碰撞过滤
        - 变换到相机坐标系
        - 旋转矩阵计算

        参数:
            scene_id: 场景ID
            im_id: 图像ID
            objects_data: 物体数据列表
            camera: 相机类型

        返回:
            List[GraspGroup]，每个元素对应 objects_data 中的一个物体实例
        """
        if camera is None:
            camera = self.gc6d_camera

        # 计算 ann_id (从 im_id 反推，使用对应的 camera)
        ann_id = self._im_id_to_ann_id(im_id, camera=camera)

        grasp_list = []

        for obj_idx, obj in enumerate(objects_data):
            obj_id = obj['obj_id']
            gt_idx = obj.get('gt_idx', 0)

            # 构建预处理文件路径
            # 格式: scene_{scene_id:06d}_ann_{ann_id:02d}_cam_{camera}_obj_{obj_id:06d}_gt_{gt_idx:02d}.npz
            preprocessed_file = self.preprocessed_grasps_dir / f"scene_{scene_id:06d}_ann_{ann_id:02d}_cam_{camera}_obj_{obj_id:06d}_gt_{gt_idx:02d}.npz"

            if not preprocessed_file.exists():
                # 文件不存在（该物体的所有 grasp 被过滤掉了），返回 None
                # 这是正常情况：某些物体由于碰撞、摩擦系数或可见性等原因没有有效 grasp
                grasp_list.append(None)
                continue

            try:
                # 加载预处理数据（已展开、已过滤、已变换）
                data = np.load(str(preprocessed_file))

                translations = data['translations']  # [N, 3] 已在相机坐标系
                rotations = data['rotations']        # [N, 3, 3] 旋转矩阵
                widths = data['widths']              # [N]
                depths = data['depths']              # [N]
                scores = data['scores']              # [N] 已计算 (1.1 - fric_coef)

                num_grasp = translations.shape[0]

                if num_grasp == 0:
                    grasp_list.append(None)
                    continue

                # 构建 GraspGroup 数组
                # 格式: [score, width, height, depth, rotation(9), translation(3), object_id]
                widths_col = widths.reshape(-1, 1)
                heights_col = GRASP_HEIGHT * np.ones((num_grasp, 1))
                depths_col = depths.reshape(-1, 1)
                scores_col = scores.reshape(-1, 1)
                rotations_flat = rotations.reshape((-1, 9))
                object_ids = obj_id * np.ones((num_grasp, 1), dtype=np.int32)

                obj_grasp_array = np.hstack([
                    scores_col, widths_col, heights_col, depths_col,
                    rotations_flat, translations, object_ids
                ]).astype(np.float32)

                # 创建 GraspGroup
                obj_grasp_group = GraspGroup()
                obj_grasp_group.grasp_group_array = obj_grasp_array

                # 可选：应用 top-k 或随机选择
                if self.max_grasps_per_object is not None and len(obj_grasp_group) > self.max_grasps_per_object:
                    if self.grasp_selection_mode == "top-k":
                        # 按分数降序排列，选择 top-k
                        grasp_scores = obj_grasp_group.scores
                        top_k_indices = np.argsort(grasp_scores)[::-1][:self.max_grasps_per_object]
                        obj_grasp_group.grasp_group_array = obj_grasp_group.grasp_group_array[top_k_indices]
                    elif self.grasp_selection_mode == "random":
                        # 随机采样
                        indices = np.random.choice(len(obj_grasp_group), self.max_grasps_per_object, replace=False)
                        obj_grasp_group.grasp_group_array = obj_grasp_group.grasp_group_array[indices]

                # 添加到列表
                if len(obj_grasp_group) > 0:
                    grasp_list.append(obj_grasp_group)
                else:
                    grasp_list.append(None)

            except Exception as e:
                print(f"Warning: 加载预处理抓取失败 {preprocessed_file}: {e}")
                import traceback
                traceback.print_exc()
                grasp_list.append(None)

        return grasp_list if len(grasp_list) > 0 else None

    def _load_grasps(
        self,
        scene_id: int,
        im_id: int,
        scene_dir: Path,
        objects_data: List[Dict],
        scene_cloud: Optional[np.ndarray] = None,
        camera: Optional[str] = None,
        crop_params: Optional[CropParams] = None,
        orig_img_shape: Optional[Tuple[int, int]] = None
    ) -> Optional[List[GraspGroup]]:
        """加载 GraspCluster6D 抓取数据

        参数:
            scene_id: 场景ID
            im_id: 图像ID
            scene_dir: 场景目录路径
            objects_data: 物体数据列表（包含 obj_id, cam_R_m2c, cam_t_m2c 等）
            scene_cloud: 场景点云（image_space格式），用于移除不可见抓取
            camera: 相机类型（用于确定深度单位），如果为None则使用默认相机
            crop_params: crop参数（用于对mask应用相同的crop变换）
            orig_img_shape: 原始图像尺寸 (height, width)

        返回:
            List[GraspGroup]，每个元素对应 objects_data 中的一个物体实例
            如果某个物体没有有效 grasp，对应位置为 None
            如果整体加载失败，返回 None
        """
        if not GRASPCLUTTER6D_AVAILABLE:
            return None

        if self.grasp_format != '6d':
            print(f"Warning: 目前只支持 6d 格式的抓取加载")
            return None

        # 快速路径：如果使用预处理的抓取数据，直接加载
        if self.use_preprocessed_grasps:
            return self._load_preprocessed_grasps(scene_id, im_id, objects_data, camera)

        # 【步骤1】加载场景点云（用于移除不可见抓取）- 直接加载
        # 注意：运行时缓存已禁用，避免多进程内存浪费
        if self.remove_invisible_grasps and scene_cloud is None:
            # 直接加载场景点云（不使用缓存）
            scene_cloud = self._load_scene_cloud_for_grasps(scene_id, im_id, scene_dir, camera=camera)

        # 2. 提取物体列表和姿态
        obj_list = [obj['obj_id'] for obj in objects_data]
        pose_list = []
        for obj in objects_data:
            pose = np.eye(4)
            pose[:3, :3] = obj['cam_R_m2c']
            # 注意：cam_t_m2c 已经在 _load_sample 中转换为米（与场景点云单位一致）
            # grasp 标签中的点坐标也是米单位，所以这里保持米单位
            pose[:3, 3] = obj['cam_t_m2c']  # 保持 m 单位
            pose_list.append(pose)

        # 3. 加载抓取标签（延迟加载，带缓存）
        # scene_dir 是 .../scenes/XXXXXX/，需要 .../GraspClutter6D/ 作为根目录
        dataset_root = scene_dir.parent.parent
        grasp_labels = self._get_grasp_labels(obj_list, dataset_root)

        # 4. 加载碰撞标签（延迟加载，带缓存）
        collision_labels = self._get_collision_labels(scene_id, dataset_root)

        if grasp_labels is None or collision_labels is None:
            return None

        # 5. 生成抓取视角模板
        num_views, num_angles, num_depths = 300, 12, 4
        template_views = generate_views(num_views)
        template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
        template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

        collision_dump = collision_labels.get(str(scene_id).zfill(6))
        if collision_dump is None:
            print(f"Warning: 未找到场景 {scene_id} 的碰撞标签")
            return None

        # 【步骤6】为每个物体实例生成抓取（分别存储）
        # 初始化列表，长度与 objects_data 一致
        grasp_list = []

        for enum_i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):
            # 为当前物体实例创建独立的 GraspGroup
            obj_grasp_group = None

            # 获取原始GT索引（用于碰撞标签）
            gt_idx = objects_data[enum_i].get('gt_idx', enum_i)

            # 检查可见性
            if self.remove_invisible_grasps:
                visible_mask_path = scene_dir / 'mask_visib' / f"{im_id:06d}_{gt_idx:06d}.png"
                amodal_mask_path = scene_dir / 'mask' / f"{im_id:06d}_{gt_idx:06d}.png"

                if not visible_mask_path.exists() or not amodal_mask_path.exists():
                    grasp_list.append(None)
                    continue

                # 直接加载原始 mask（不使用缓存）
                # 注意：运行时缓存已禁用，避免多进程内存浪费
                visible_mask_orig = cv2.imread(str(visible_mask_path))
                amodal_mask_orig = cv2.imread(str(amodal_mask_path))

                if visible_mask_orig is None or amodal_mask_orig is None:
                    grasp_list.append(None)
                    continue

                # 应用与图像相同的 crop 变换
                if crop_params is not None:
                    preset_crop = (
                        crop_params.crop_width,
                        crop_params.crop_height,
                        crop_params.crop_left,
                        crop_params.crop_top
                    )

                    # 对 visible_mask 应用 crop
                    visible_mask, _, _, _, _, _, _ = _crop_resize_with_pad(
                        visible_mask_orig,
                        self.vggt_image_size,
                        self.vggt_image_size,
                        pad_value=0,
                        interpolation=cv2.INTER_NEAREST,
                        debug=True,
                        preset_crop_params=preset_crop
                    )

                    # 对 amodal_mask 也应用相同的 crop
                    amodal_mask, _, _, _, _, _, _ = _crop_resize_with_pad(
                        amodal_mask_orig,
                        self.vggt_image_size,
                        self.vggt_image_size,
                        pad_value=0,
                        interpolation=cv2.INTER_NEAREST,
                        debug=True,
                        preset_crop_params=preset_crop
                    )
                else:
                    # 如果没有提供 crop 参数，使用原始 mask
                    visible_mask = visible_mask_orig
                    amodal_mask = amodal_mask_orig
            else:
                visible_mask = None
                amodal_mask = None

            # 【步骤6b】物体级别可见性过滤已移至_load_sample，此处不再过滤
            # visible_mask和amodal_mask保留用于后续移除不可见的grasp点（_remove_invisible_grasps）

            # 获取抓取标签
            if obj_idx not in grasp_labels:
                grasp_list.append(None)
                continue

            sampled_points, offsets, fric_coefs = grasp_labels[obj_idx]

            # 5. 从标签数据推断维度（而不是硬编码）
            # offsets 的形状: [num_points, num_views, num_angles, num_depths, 3]
            # fric_coefs 的形状: [num_points, num_views, num_angles, num_depths]
            if offsets.ndim != 5 or fric_coefs.ndim != 4:
                print(f"Warning: 物体 {obj_idx} 的标签格式不正确，跳过")
                grasp_list.append(None)
                continue

            num_points_in_label = offsets.shape[0]
            num_views = offsets.shape[1]
            num_angles = offsets.shape[2]
            num_depths = offsets.shape[3]

            # 生成抓取视角模板（使用从标签推断的维度）
            template_views = generate_views(num_views)
            template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
            template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

            # 获取碰撞标签（使用原始GT索引）
            if gt_idx >= len(collision_dump):
                grasp_list.append(None)
                continue
            collision = collision_dump[gt_idx]

            # 【性能优化】限制采样点数量（如果配置了 max_sample_points）
            if self.max_sample_points is not None and sampled_points.shape[0] > self.max_sample_points:
                # 随机选择子集，保持顺序以便复现
                indices = np.random.choice(
                    sampled_points.shape[0],
                    self.max_sample_points,
                    replace=False
                )
                indices = np.sort(indices)  # 保持原始顺序
                # 同步子采样所有相关数组
                sampled_points = sampled_points[indices]
                offsets = offsets[indices]
                fric_coefs = fric_coefs[indices]
                collision = collision[indices]  # collision 的第一维度也是 num_points

            # 【步骤6c】移除不可见抓取点
            if self.remove_invisible_grasps and scene_cloud is not None and visible_mask is not None:
                sampled_points, offsets, fric_coefs, collision = self._remove_invisible_grasps(
                    scene_cloud, visible_mask, sampled_points, offsets, fric_coefs, collision, trans
                )

            # 生成抓取姿态
            point_inds = np.arange(sampled_points.shape[0])
            num_points = len(point_inds)

            if num_points == 0:
                grasp_list.append(None)
                continue

            target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
            target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
            views = np.tile(template_views, [num_points, 1, 1, 1, 1])
            angles = offsets[:, :, :, :, 0]
            depths = offsets[:, :, :, :, 1]
            widths = offsets[:, :, :, :, 2]

            # 过滤抓取（摩擦系数阈值 + 无碰撞）
            mask1 = ((fric_coefs <= self.fric_coef_thresh) & (fric_coefs > 0) & ~collision)
            target_points = target_points[mask1]
            views = views[mask1]
            angles = angles[mask1]
            depths = depths[mask1]
            widths = widths[mask1]
            fric_coefs = fric_coefs[mask1]

            if target_points.shape[0] == 0:
                grasp_list.append(None)
                continue

            # 变换到相机坐标系
            target_points = transform_points(target_points, trans)

            # 计算旋转矩阵
            Rs = batch_viewpoint_params_to_matrix(-views, angles)
            Rs = np.matmul(trans[np.newaxis, :3, :3], Rs)

            # 构建当前物体的抓取数组
            num_grasp = widths.shape[0]
            scores = (1.1 - fric_coefs).reshape(-1, 1)
            widths = widths.reshape(-1, 1)
            heights = GRASP_HEIGHT * np.ones((num_grasp, 1))
            depths = depths.reshape(-1, 1)
            rotations = Rs.reshape((-1, 9))
            object_ids = obj_idx * np.ones((num_grasp, 1), dtype=np.int32)

            obj_grasp_array = np.hstack([
                scores, widths, heights, depths, rotations, target_points, object_ids
            ]).astype(np.float32)

            # 为当前物体创建 GraspGroup
            obj_grasp_group = GraspGroup()
            obj_grasp_group.grasp_group_array = obj_grasp_array

            # 对当前物体的 grasp 应用 top-k 过滤
            if self.max_grasps_per_object is not None and len(obj_grasp_group) > self.max_grasps_per_object:
                if self.grasp_selection_mode == "top-k":
                    # 按分数降序排列，选择 top-k
                    scores = obj_grasp_group.scores
                    top_k_indices = np.argsort(scores)[::-1][:self.max_grasps_per_object]
                    obj_grasp_group.grasp_group_array = obj_grasp_group.grasp_group_array[top_k_indices]
                elif self.grasp_selection_mode == "random":
                    # 随机采样
                    indices = np.random.choice(len(obj_grasp_group), self.max_grasps_per_object, replace=False)
                    obj_grasp_group.grasp_group_array = obj_grasp_group.grasp_group_array[indices]

            # 添加到列表（如果为空则添加 None）
            if len(obj_grasp_group) > 0:
                grasp_list.append(obj_grasp_group)
            else:
                grasp_list.append(None)

        # 返回 grasp 列表
        # 注意：即使所有元素都是 None，也返回列表以保持与 objects_data 的对应关系
        # 只有在发生错误时才返回 None
        return grasp_list if len(grasp_list) > 0 else None

    def _load_scene_cloud_for_grasps(self, scene_id: int, im_id: int, scene_dir: Path, camera: str = None) -> Optional[
        np.ndarray]:
        """加载场景点云（image_space格式）用于抓取可见性检测

        参数:
            scene_id: 场景ID
            im_id: 图像ID
            scene_dir: 场景目录
            camera: 相机类型（用于确定深度单位），如果为None则使用默认相机
        """
        try:
            # 加载深度图
            depth_path = scene_dir / 'depth' / f"{im_id:06d}.png"
            if not depth_path.exists():
                return None

            depth_raw = inout.load_depth(str(depth_path)).astype(np.float32)

            # 深度单位转换
            camera_for_depth = camera if camera else self.gc6d_camera
            if camera_for_depth in ['realsense-d415', 'realsense-d435']:
                depth_scale = 1000.0
            elif camera_for_depth in ['azure-kinect', 'zivid']:
                depth_scale = 10000.0
            else:
                depth_scale = 1000.0

            # 加载相机内参
            scene_camera_path = scene_dir / "scene_camera.json"
            scene_camera = inout.load_scene_camera(str(scene_camera_path))
            cam_info = scene_camera[im_id]
            cam_K = np.array(cam_info['cam_K']).reshape(3, 3)
            fx, fy = cam_K[0, 0], cam_K[1, 1]
            cx, cy = cam_K[0, 2], cam_K[1, 2]

            # 生成image_space点云
            H, W = depth_raw.shape
            xmap, ymap = np.arange(W), np.arange(H)
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_z = depth_raw / depth_scale
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            # 堆叠并展平为 (H*W, 3)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points.reshape(-1, 3)

            # 过滤掉无效深度点（z <= 0）
            valid_mask = points[:, 2] > 0
            points = points[valid_mask]

            return points

        except Exception as e:
            print(f"Warning: 加载场景点云失败: {e}")
            return None

    def _get_grasp_labels(self, obj_list: List[int], dataset_root: Path) -> Optional[Dict]:
        """获取抓取标签（带缓存）"""
        # 使用frozenset作为缓存键（因为list不可哈希）
        cache_key = frozenset(obj_list)

        if cache_key not in self.grasp_labels_cache:
            try:
                grasp_labels = {}
                for obj_id in obj_list:
                    # GraspClutter6D 标签文件格式: obj_XXXXXX_labels.npz
                    grasp_label_path = dataset_root / 'grasp_label' / f'obj_{obj_id:06d}_labels.npz'
                    if grasp_label_path.exists():
                        data = np.load(str(grasp_label_path))
                        grasp_labels[obj_id] = (data['points'], data['offsets'], data['scores'])
                    else:
                        print(f"Warning: 未找到物体 {obj_id} 的抓取标签: {grasp_label_path}")

                self.grasp_labels_cache[cache_key] = grasp_labels if grasp_labels else None
            except Exception as e:
                print(f"Warning: 加载抓取标签失败: {e}")
                self.grasp_labels_cache[cache_key] = None

        return self.grasp_labels_cache[cache_key]

    def _get_collision_labels(self, scene_id: int, dataset_root: Path) -> Optional[Dict]:
        """获取碰撞标签（带缓存）

        返回格式: {
            "scene_id_str": [arr_0, arr_1, arr_2, ...]  # 每个物体的碰撞标签数组
        }
        """
        if scene_id not in self.collision_labels_cache:
            try:
                # GraspClutter6D 碰撞标签文件格式: XXXXXX.npz
                collision_label_path = dataset_root / 'collision_label' / f'{scene_id:06d}.npz'
                if collision_label_path.exists():
                    data = np.load(str(collision_label_path), allow_pickle=True)
                    # 按照 GraspClutter6D API 格式构建：键是场景ID字符串，值是列表
                    collision_list = [data[f'arr_{j}'] for j in range(len(data.files))]
                    scene_key = f'{scene_id:06d}'
                    self.collision_labels_cache[scene_id] = {scene_key: collision_list}
                else:
                    print(f"Warning: 未找到场景 {scene_id} 的碰撞标签: {collision_label_path}")
                    self.collision_labels_cache[scene_id] = None
            except Exception as e:
                print(f"Warning: 加载碰撞标签失败: {e}")
                import traceback
                traceback.print_exc()
                self.collision_labels_cache[scene_id] = None

        return self.collision_labels_cache[scene_id]

    def _remove_invisible_grasps(
            self,
            scene_cloud: np.ndarray,
            visible_mask: np.ndarray,
            sampled_points: np.ndarray,
            offsets: np.ndarray,
            fric_coefs: np.ndarray,
            collision: np.ndarray,
            trans: np.ndarray,
            th: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """移除不可见的抓取点

        参考 GraspCluster6D API 的 removeInvisibleGrasp 方法
        scene_cloud: (N, 3) 已经是展平并过滤后的点云
        visible_mask: (H, W, 3) 物体可见性掩码（未使用，因为scene_cloud已经是有效点）
        """
        # 将抓取点变换到相机坐标系
        grasp_points_cam = transform_points(sampled_points, trans)

        # scene_cloud 已经是 (N, 3) 格式且已过滤无效点
        # 如果点太多，随机采样
        if scene_cloud.shape[0] > 20000:
            indices = np.random.choice(scene_cloud.shape[0], 20000, replace=False)
            scene_cloud_sampled = scene_cloud[indices]
        else:
            scene_cloud_sampled = scene_cloud

        if scene_cloud_sampled.shape[0] == 0:
            # 没有有效场景点，返回空
            return (np.zeros((0, 3)), np.zeros((0, 300, 12, 4, 3)),
                    np.zeros((0, 300, 12, 4)), np.zeros((0, 300, 12, 4), dtype=bool))

        # 计算每个抓取点到最近场景点的距离
        # grasp_points_cam: (M, 3), scene_cloud_sampled: (N, 3)
        grasp_points_expanded = grasp_points_cam[:, np.newaxis, :]  # (M, 1, 3)
        scene_cloud_expanded = scene_cloud_sampled[np.newaxis, :, :]  # (1, N, 3)
        dists = np.linalg.norm(grasp_points_expanded - scene_cloud_expanded, axis=-1)  # (M, N)
        min_dists = dists.min(axis=1)  # (M,)

        # 保留距离小于阈值的抓取点
        visible_point_mask = min_dists < th

        return (sampled_points[visible_point_mask],
                offsets[visible_point_mask],
                fric_coefs[visible_point_mask],
                collision[visible_point_mask])


# NOTE: Visualization helpers (add_axis_to_image, compute_3d_bbox_vertices_batch,
# draw_text_with_background, visualize_rays_as_rgb, visualize_depth_simple_comparison,
# project_3d_to_2d, save_image_chw) have been removed. They were only used by
# ad-hoc debugging scripts. See git history if needed.


# ==================== DataCollator ====================

# NOTE: The legacy ``DataCollatorForPI0BopConsumerDataset`` (clutter variant)
# has been removed. All four detection datasets (omni6d / omni3d / bop /
# clutter) now share the unified ``CollatorForDetectionDataset`` defined in
# ``collators.py``. Use:
#     from collators import CollatorForDetectionDataset
#     collator = CollatorForDetectionDataset()


# ==================== 测试代码 ====================
import hydra
@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="base",
)
def main(cfg):
    bin_tokenizer = BinTokenizer(cfg.statistics_path_6d_dataset)
    train_det_dataset = BopClutterConsumerDataset(config=cfg, tokenizer=bin_tokenizer)
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

