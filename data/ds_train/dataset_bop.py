"""
BOP 数据集加载器,输出与 PI0 模型兼容的格式

基于 BOP 标准格式,参考 dataset_det.py 的输出结构
支持多摄像头、3D 检测、深度图等功能
"""
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf, ListConfig, DictConfig
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

# BOP toolkit 导入
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "bop_toolkit"))
from bop_toolkit_lib import inout
from utils.vis import visualize_2d_3d_all, visualize_views
from utils.mapping_token import text_to_class_attr_dict, map_3d_label_to_string
from utils.mapping_token import text_to_class_attr_dict_tokenizer, map_3d_label_to_string_tokenizer, BinTokenizer
import tempfile
import shutil


# ---------------------------------------------------------------------------
# Re-use the canonical helper implementations defined in ``dataset_omni6d``.
# Keeping a single source of truth avoids subtle drift between the four
# detection datasets (omni6d / omni3d / bop / clutter).
# ---------------------------------------------------------------------------
from data.ds_train.dataset_omni6d import (  # noqa: E402  -- intentional re-export
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


@dataclass
class _SampleEntry:
    """单个样本的索引信息"""
    scene_id: int
    im_id: int
    scene_dir: Path
    dataset_root: Path  # 记录数据集根目录（用于多数据集支持）
    dataset_name: str   # 记录数据集名称（如 ycbv, hb）


class BopConsumerDataset(Dataset):
    """
    BOP 数据集,输出与 PI0 模型兼容的格式

    基于 BOP 场景格式,将每一帧展开为独立样本
    支持多摄像头、深度图、3D 检测等功能
    """

    def __init__(self, config: Any, tokenizer: BinTokenizer, yaml_name="bop_train") -> None:
        super().__init__()
        self.config = config

        # 检查配置
        if not hasattr(config.dataset_bop, yaml_name):
            raise ValueError(f"config.dataset_bop.{yaml_name} 未配置,无法构建 BOP 数据集")

        # bop_cfg = config.dataset_bop
        bop_cfg = getattr(config.dataset_bop, yaml_name)
        self.bop_cfg = bop_cfg

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
            {"name": "top_head",    "crop_scale": (0.90, 0.90), "aspect_ratio": (1.33, 1.33)},
            {"name": "hand_left",   "crop_scale": (0.25, 0.25), "aspect_ratio": (1.33, 1.33)},
            {"name": "hand_right",  "crop_scale": (0.55, 0.55), "aspect_ratio": (1.33, 1.33)},
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
        category_file = 'cache/bop_category_list.json'
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
            print(f'Category list saved to bop_category_list.json, Total {len(self.category_all_list)} categories')
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

    def _discover_scenes(self, data_root: Path) -> List[int]:
        """发现指定数据集的所有可用场景"""
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
                    entries = [
                        _SampleEntry(
                            scene_id=item["scene_id"],
                            im_id=item["im_id"],
                            scene_dir=Path(item["scene_dir"]),
                            dataset_root=Path(item["dataset_root"]),
                            dataset_name=item["dataset_name"]
                        )
                        for item in payload.get("samples", [])
                    ]
                    all_entries.extend(entries)
                    entries_from_cache = True
                    print(f"    从缓存加载: {len(entries)} 个样本")

            if not entries_from_cache:
                # 获取场景ID
                scene_ids = bop_cfg.get("scene_ids", None)
                if scene_ids is None:
                    scene_ids = self._discover_scenes(data_root)

                entries: List[_SampleEntry] = []

                for scene_id in scene_ids:
                    scene_dir = split_path / f"{scene_id:06d}"
                    scene_camera_path = scene_dir / "scene_camera.json"

                    if not scene_camera_path.exists():
                        print(f"    Warning: scene_camera.json not found in {scene_dir}")
                        continue

                    # 加载场景相机参数获取图像ID
                    scene_camera = inout.load_scene_camera(str(scene_camera_path))
                    im_ids = sorted(scene_camera.keys())

                    # 添加到样本列表
                    for im_id in im_ids:
                        entries.append(_SampleEntry(
                            scene_id=scene_id,
                            im_id=im_id,
                            scene_dir=scene_dir,
                            dataset_root=data_root,
                            dataset_name=dataset_name
                        ))

                all_entries.extend(entries)
                print(f"    索引完成: {len(entries)} 个样本")

                # 保存缓存
                # if use_cache and cache_path:
                #     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                #     with open(cache_path, "w", encoding="utf-8") as fp:
                #         json.dump(
                #             {
                #                 "data_root": str(data_root),
                #                 "split": self.split,
                #                 "samples": [{
                #                     "scene_id": e.scene_id,
                #                     "im_id": e.im_id,
                #                     "scene_dir": str(e.scene_dir),
                #                     "dataset_root": str(e.dataset_root),
                #                     "dataset_name": e.dataset_name
                #                 } for e in entries],
                #             },
                #             fp,
                #             ensure_ascii=False,
                #         )

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
                                        "dataset_name": e.dataset_name
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
            return None

        # 加载深度图
        depth = None
        depth_mask = None
        if self.load_depth:
            depth_path = scene_dir / 'depth' / f"{im_id:06d}.png"
            if depth_path.exists():
                try:
                    depth_raw = inout.load_depth(str(depth_path)).astype(np.float32)
                    # BOP 深度单位转换: 除以 unit_scale 转换为米 (如 mm->m: /1000.0)
                    depth_raw = depth_raw / self.unit_scale
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

                        # 2D bbox
                        bbox_obj = gt_info.get('bbox_obj', [0, 0, rgb.shape[1], rgb.shape[0]])

                        # 过滤2: bbox面积太小的物体（与官方BOP toolkit一致）
                        bbox_area = bbox_obj[2] * bbox_obj[3]  # width * height
                        if bbox_area < 150:
                            continue

                        # 旋转和平移
                        cam_R_m2c = np.array(gt['cam_R_m2c']).reshape(3, 3)
                        cam_t_m2c = np.array(gt['cam_t_m2c']).reshape(3) / self.unit_scale  # 单位转换 (如 mm->m)

                        objects_data.append({
                            'obj_id': obj_id,
                            'gt_idx': gt_idx,  # 添加 gt_idx，用于加载mask文件
                            'cam_R_m2c': cam_R_m2c,
                            'cam_t_m2c': cam_t_m2c,
                            'bbox_obj': bbox_obj,
                            'visib_fract': visib_fract,
                        })

        if len(objects_data) == 0:
            return None

        # 随机选择一个物体类别作为主要检测目标
        target_obj = random.choice(objects_data)
        obj_id = target_obj['obj_id']

        # 过滤出与目标物体相同类别的所有实例（与 dataset_det.py 一致）
        same_class_objects = [obj for obj in objects_data if obj['obj_id'] == obj_id]

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
            # avg_scale = dist[top_depth_mask_tensor.squeeze(0) > 0].max().clamp(min=1e-3, max=1e3)
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
            # cam_know_depth = _build_sparse_depth(cam_depth_tensor, cam_depth_mask_tensor,
            #                                      self.bop_cfg.depth_full_prob, self.bop_cfg.depth_sparse_prob)
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

            for obj_data in same_class_objects:
                gt_idx = obj_data['gt_idx']
                bbox_orig = np.array(obj_data['bbox_obj'])  # [x, y, w, h]

                # 加载该物体的 visible_mask 和 amodal_mask（用于可见性过滤）
                visible_mask_path = scene_dir / 'mask_visib' / f"{im_id:06d}_{gt_idx:06d}.png"
                amodal_mask_path = scene_dir / 'mask' / f"{im_id:06d}_{gt_idx:06d}.png"

                # 加载masks
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

            camera_bboxes_all.append(cam_bbox_list)
            camera_pose_list.append(cam_pose_list)
            camera_bbox_side_len_list.append(cam_bbox_side_len_list)
            camera_class_labels.append(cam_class_label_list)

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
        if class_name_clean not in self.category_all_set:
            print(class_name_clean, 'bop', '==================')
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
            "task":        f"detect the {class_name_clean}",
            "text_label":  text_label,

            # ===== 多相机 2D / 3D 标注 =====
            "bbox_2d":              camera_bboxes_all,              # List[List[Tensor(4,)]]
            "camera_pose":          camera_pose_tensors,            # List[Tensor(N,7)]
            "camera_bbox_side_len": camera_bbox_side_len_tensors,   # List[Tensor(N,3)]
            "camera_class_labels":  camera_class_labels,            # List[List[int]]
        }

        self.crop_param_manager.clear_cache()

        return sample


# NOTE: Visualization helpers (add_axis_to_image, compute_3d_bbox_vertices_batch,
# visualize_rays_as_rgb, visualize_depth_simple_comparison, project_3d_to_2d,
# save_image_chw) have been removed. They were only used by ad-hoc debugging
# scripts. See git history if needed.


# ==================== DataCollator ====================

# NOTE: The legacy ``DataCollatorForPI0BopConsumerDataset`` has been removed.
# All four detection datasets (omni6d / omni3d / bop / clutter) now share the
# unified ``CollatorForDetectionDataset`` defined in ``collators.py``. Use:
#     from collators import CollatorForDetectionDataset
#     collator = CollatorForDetectionDataset(config=cfg, sub_cfg_key="dataset_bop")


# ==================== 测试代码 ====================
import hydra
@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="base",
)
def main(cfg):
    bin_tokenizer = BinTokenizer(cfg.statistics_path_6d_dataset)
    train_det_dataset = BopConsumerDataset(config=cfg, tokenizer=bin_tokenizer, yaml_name="bop_train")
    # Unified collator shared by all four detection datasets.
    # For BOP we read camera_configs from ``cfg.dataset_bop``.
    from data.collators import CollatorForDetectionDataset
    data_det_collator = CollatorForDetectionDataset(config=cfg, sub_cfg_key="dataset_bop")
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
