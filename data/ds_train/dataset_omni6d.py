import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import random

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

try:
    import OpenEXR  # type: ignore
    import Imath  # type: ignore
except ImportError as exc:  # pragma: no cover - 环境缺少依赖时给出明确提示
    raise ImportError("缺少 OpenEXR 依赖，无法读取 .exr 深度/掩码文件，请先安装 openexr 与 imath 包") from exc
from cutoop.data_loader import Dataset as OmniDataset
from utils.vis import visualize_2d_3d_all, visualize_views
from mapping_token import text_to_class_attr_dict, map_3d_label_to_string
from mapping_token import text_to_class_attr_dict_tokenizer, map_3d_label_to_string_tokenizer, BinTokenizer
from tqdm import tqdm
import tempfile
import shutil

def _random_crop_coords(crop_scale: Tuple[float, float], aspect_ratio: Tuple[float, float],
                        orig_width: int, orig_height: int, seed: int | None = None) -> Tuple[int, int, int, int]:
    """计算随机裁剪的坐标和尺寸

    参数:
        crop_scale: 裁剪比例范围，如 (0.75, 0.75) 表示固定裁剪比例
        aspect_ratio: 宽高比范围，如 (1.33, 1.33) 表示固定宽高比
        orig_width: 原始图像宽度
        orig_height: 原始图像高度
        seed: 随机种子，用于保证可重复性

    返回:
        crop_width, crop_height, left, top
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 随机选择裁剪区域比例
    scale = random.uniform(*crop_scale)
    # 随机选择宽高比 (在对数空间均匀采样)
    log_ratio = (np.log(aspect_ratio[0]), np.log(aspect_ratio[1]))
    target_ratio = np.exp(random.uniform(*log_ratio))

    # 计算裁剪尺寸
    crop_area = orig_width * orig_height * scale
    crop_width = int(round(np.sqrt(crop_area * target_ratio)))
    crop_height = int(round(np.sqrt(crop_area / target_ratio)))

    # 确保不超过原始尺寸
    crop_width = min(crop_width, orig_width)
    crop_height = min(crop_height, orig_height)

    # 随机选择裁剪位置
    left = random.randint(0, orig_width - crop_width) if orig_width > crop_width else 0
    top = random.randint(0, orig_height - crop_height) if orig_height > crop_height else 0

    return crop_width, crop_height, left, top


def _deterministic_crop_coords(crop_scale: Tuple[float, float], aspect_ratio: Tuple[float, float],
                                orig_width: int, orig_height: int) -> Tuple[int, int, int, int]:
    """计算确定性（中心）裁剪的坐标和尺寸，用于调试

    参数:
        crop_scale: 裁剪比例范围，使用中间值
        aspect_ratio: 宽高比范围，使用中间值
        orig_width: 原始图像宽度
        orig_height: 原始图像高度

    返回:
        crop_width, crop_height, left, top
    """
    # # 使用裁剪比例范围的中间值
    # scale = (crop_scale[0] + crop_scale[1]) / 2.0
    # 随机选择裁剪区域比例
    scale = random.uniform(*crop_scale)

    # 使用宽高比范围的中间值（在对数空间）
    log_ratio_min = np.log(aspect_ratio[0])
    log_ratio_max = np.log(aspect_ratio[1])
    target_ratio = np.exp((log_ratio_min + log_ratio_max) / 2.0)

    # 计算裁剪尺寸
    crop_area = orig_width * orig_height * scale
    crop_width = int(round(np.sqrt(crop_area * target_ratio)))
    crop_height = int(round(np.sqrt(crop_area / target_ratio)))

    # 确保不超过原始尺寸
    crop_width = min(crop_width, orig_width)
    crop_height = min(crop_height, orig_height)

    # 使用中心位置（确定性）
    left = (orig_width - crop_width) // 2
    top = (orig_height - crop_height) // 2

    return crop_width, crop_height, left, top


@dataclass
class CropParams:
    """Crop变换参数

    包含完整的crop变换链所需的所有参数：
    - resize_ratio: 缩放比例
    - pad_left/pad_top: 填充偏移
    - crop_left/crop_top: 裁剪起始位置
    - crop_width/crop_height: 裁剪区域尺寸
    - output_width/output_height: 输出图像尺寸
    """
    resize_ratio: float     # 缩放比例
    pad_left: int           # 填充左侧偏移
    pad_top: int            # 填充顶部偏移
    crop_left: int          # 裁剪区域左上角X坐标
    crop_top: int           # 裁剪区域左上角Y坐标
    crop_width: int         # 裁剪区域宽度
    crop_height: int        # 裁剪区域高度
    output_width: int       # 输出图像宽度
    output_height: int      # 输出图像高度


class CropParamManager:
    """管理crop参数的生成和缓存

    负责为每个样本的每个摄像头的每个尺寸生成crop参数，并缓存复用
    确保同一样本同一摄像头的RGB、depth、bbox变换使用相同的crop参数
    """
    def __init__(self, enable_random: bool = False):
        """初始化crop参数管理器

        参数:
            enable_random: 是否启用随机crop（True=随机crop，False=确定性中心crop）
        """
        self.enable_random = enable_random
        # 缓存键: (sample_idx, cam_idx, size_key) -> CropParams
        self.cache: Dict[Tuple[int, int, str], CropParams] = {}

    def get_or_create_params(
        self,
        sample_idx: int,
        cam_idx: int,
        size_key: str,
        orig_width: int,
        orig_height: int,
        crop_scale: Tuple[float, float],
        aspect_ratio: Tuple[float, float]
    ) -> CropParams:
        """获取或创建crop参数

        参数:
            sample_idx: 样本索引
            cam_idx: 摄像头索引
            size_key: 尺寸标识符（如 "224", "518"）
            orig_width: 原始图像宽度
            orig_height: 原始图像高度
            crop_scale: 裁剪比例范围
            aspect_ratio: 宽高比范围

        返回:
            CropParams对象
        """
        cache_key = (sample_idx, cam_idx, size_key)

        if cache_key not in self.cache:
            # 生成crop参数
            if self.enable_random:
                crop_width, crop_height, crop_left, crop_top = _random_crop_coords(
                    crop_scale, aspect_ratio, orig_width, orig_height
                )
            else:
                crop_width, crop_height, crop_left, crop_top = _deterministic_crop_coords(
                    crop_scale, aspect_ratio, orig_width, orig_height
                )

            # 计算目标尺寸（从size_key解析）
            target_size = int(size_key)

            # 创建参数对象（resize_ratio和pad稍后在实际处理时计算）
            params = CropParams(
                resize_ratio=0.0,  # 占位符，稍后更新
                pad_left=0,        # 占位符，稍后更新
                pad_top=0,         # 占位符，稍后更新
                crop_left=crop_left,
                crop_top=crop_top,
                crop_width=crop_width,
                crop_height=crop_height,
                output_width=target_size,
                output_height=target_size
            )

            self.cache[cache_key] = params

        return self.cache[cache_key]

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()


def _load_exr(path: str, channel: str | None = None) -> np.ndarray:
    """读取 EXR 文件并返回单通道数组。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 EXR 文件: {path}")

    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = list(header["channels"].keys())
    target_channel = channel
    if target_channel is None:
        # 深度常用 'Y' 通道，mask 多为 'R'
        if "Y" in channels:
            target_channel = "Y"
        else:
            target_channel = channels[0]
    elif target_channel not in channels:
        # 如果指定通道不存在，则退回到第一个可用通道
        fallback = channels[0]
        # print(f"[WARN] EXR 文件 {path} 中不存在通道 '{target_channel}'，改用 '{fallback}'")
        target_channel = fallback

    buffer = exr_file.channel(target_channel, pixel_type)
    array = np.frombuffer(buffer, dtype=np.float32)
    array.shape = (height, width)
    return array


def _resize_and_center_crop(
    array: np.ndarray,
    target_size: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """缩放并中心裁剪到目标尺寸，确保输出正方形图像。"""
    orig_h, orig_w = array.shape[:2]

    # 先缩放到短边等于目标尺寸
    scale = target_size / min(orig_h, orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    resized = cv2.resize(array, (new_w, new_h), interpolation=interpolation)

    # 中心裁剪到目标尺寸
    start_h = (new_h - target_size) // 2
    start_w = (new_w - target_size) // 2

    cropped = resized[start_h:start_h + target_size, start_w:start_w + target_size]

    return cropped


def _crop_resize_with_pad(
    array: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: float,
    interpolation: int,
    crop_scale: Tuple[float, float] = (0.75, 0.75),
    aspect_ratio: Tuple[float, float] = (1.33, 1.33),
    debug: bool = True,
    seed: int | None = None,
    preset_crop_params: Tuple[int, int, int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, int, int, int, int]:
    """先crop再resize再pad，保持纵横比缩放并填充，同时返回有效区域掩码与缩放信息。

    流程: crop → resize → pad

    参数:
        array: 输入数组 (H, W) 或 (H, W, C)
        target_h: 目标高度
        target_w: 目标宽度
        pad_value: 填充值
        interpolation: 插值方法
        crop_scale: 裁剪比例范围
        aspect_ratio: 宽高比范围
        debug: 是否使用确定性裁剪（中心裁剪）
        seed: 随机种子
        preset_crop_params: 预设的crop参数 (crop_w, crop_h, crop_left, crop_top)，如果提供则直接使用

    返回:
        (padded_array, valid_mask, resize_ratio, pad_left, pad_top, crop_left, crop_top)
    """
    orig_h, orig_w = array.shape[:2]
    if orig_h == 0 or orig_w == 0:
        raise ValueError("输入尺寸非法，无法执行 crop_resize_with_pad")

    # Step 1: Crop
    if preset_crop_params is not None:
        # 使用预设的crop参数
        crop_w, crop_h, crop_left, crop_top = preset_crop_params
    elif debug:
        crop_w, crop_h, crop_left, crop_top = _deterministic_crop_coords(
            crop_scale, aspect_ratio, orig_width=orig_w, orig_height=orig_h
        )
    else:
        crop_w, crop_h, crop_left, crop_top = _random_crop_coords(
            crop_scale, aspect_ratio, orig_width=orig_w, orig_height=orig_h, seed=seed
        )

    # 执行裁剪
    if array.ndim == 2:
        cropped = array[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
    else:
        cropped = array[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w, :]

    # Step 2: Resize (保持宽高比)
    resize_ratio = max(crop_w / target_w, crop_h / target_h)
    new_h = max(1, int(round(crop_h / resize_ratio)))
    new_w = max(1, int(round(crop_w / resize_ratio)))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interpolation)

    # Step 3: Pad
    if resized.ndim == 2:
        padded = np.full((target_h, target_w), pad_value, dtype=resized.dtype)
    else:
        padded = np.full((target_h, target_w, resized.shape[2]), pad_value, dtype=resized.dtype)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    if resized.ndim == 2:
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    else:
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w, :] = resized

    # 生成有效区域掩码
    valid_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    valid_mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = 1

    return padded, valid_mask, resize_ratio, pad_left, pad_top, crop_left, crop_top


def _resize_with_pad(
    array: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: float,
    interpolation: int,
) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """保持纵横比缩放并填充，同时返回有效区域掩码与缩放信息。

    返回:
        (padded_array, valid_mask, ratio, pad_left, pad_top)
        其中 ratio = max(orig_h / target_h, orig_w / target_w)，表示缩放比例（用于内参调整）
    """
    orig_h, orig_w = array.shape[:2]
    if orig_h == 0 or orig_w == 0:
        raise ValueError("输入尺寸非法，无法执行 resize_with_pad")

    scale = min(target_h / orig_h, target_w / orig_w)
    new_h = max(1, int(round(orig_h * scale)))
    new_w = max(1, int(round(orig_w * scale)))

    if array.ndim == 2:
        resized = cv2.resize(array, (new_w, new_h), interpolation=interpolation)
        padded = np.full((target_h, target_w), pad_value, dtype=resized.dtype)
    else:
        resized = cv2.resize(array, (new_w, new_h), interpolation=interpolation)
        padded = np.full((target_h, target_w, array.shape[2]), pad_value, dtype=resized.dtype)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w, ...] = resized

    valid_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    valid_mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = 1

    # 返回 ratio（用于内参调整）= 1/scale，即 orig_size / target_size
    ratio = 1.0 / scale if scale > 0 else 1.0
    return padded, valid_mask, ratio, pad_left, pad_top


def _threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float,
    min_percentile: float,
    max_depth: float = -1.0,
) -> np.ndarray:
    """参考 Omni6D 官方实现，使用分位数阈值筛除异常深度。"""
    depth = depth_map.astype(np.float32).copy()

    if max_depth > 0:
        depth[depth > max_depth] = 0.0

    valid = depth[depth > 0]
    if valid.size == 0:
        return depth

    if max_percentile > 0:
        max_thr = np.nanpercentile(valid, max_percentile)
        if max_thr > 0:
            depth[depth > max_thr] = 0.0

    if min_percentile > 0:
        min_thr = np.nanpercentile(valid, min_percentile)
        if min_thr > 0:
            depth[depth < min_thr] = 0.0

    return depth


def _intrinsics_to_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """将相机内参转换为 3x3 矩阵。"""
    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K


def _generate_coord_grid(width: int, height: int) -> torch.Tensor:
    """生成像素坐标网格 (2, H, W)，第一行为 x，第二行为 y。"""
    xs = torch.linspace(0, width - 1, width)
    ys = torch.linspace(0, height - 1, height)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=0)


def _depth_to_pcl(depth: torch.Tensor, K: torch.Tensor, coord_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """将深度转换为点云 (H, W, 3)。"""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (coord_map[0] - cx) * depth / fx
    y = (coord_map[1] - cy) * depth / fy
    z = depth
    pcl = torch.stack([x, y, z], dim=-1)
    pcl[mask == 0] = 0
    return pcl


def _generate_rays(height: int, width: int, K: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """生成每个像素的单位视线 (3, H, W), valid_mask (H,W)"""
    coord_map = _generate_coord_grid(width, height).to(K)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    dirs = torch.stack(
        [
            (coord_map[0] - cx) / fx,
            (coord_map[1] - cy) / fy,
            torch.ones_like(coord_map[0]),
        ],
        dim=0,
    )
    norm = torch.linalg.norm(dirs, dim=0, keepdim=True).clamp_min(1e-6)
    # dirs = dirs / norm
    dirs[:, valid_mask == 0] = 0
    return dirs


def _build_sparse_depth(depth: torch.Tensor, depth_mask: torch.Tensor, probality_full, probality_sparse) -> torch.Tensor:
    """构造基础稀疏深度张量 (1, 2, H, W)。"""
    depth_map = depth.squeeze(0)
    mask_map = depth_mask.squeeze(0).float()
    stacked = torch.stack([depth_map, mask_map], dim=0).unsqueeze(0) #(B,2,h,w)
    sparse_depth = sample_points_from_sparse_mask_v2_vectorized(
        [stacked], debug=False, probality_full=probality_full, probality_sparse=probality_sparse
    )
    return sparse_depth[0][0]


@dataclass
class _SampleEntry:
    meta_path: str
    class_name: str
    subset: str


class Omni6dConsumerDataset(Dataset):
    """Omni6D 检测数据集，输出与 dataset_vlm 相同的多摄像头键以及 3D 检测所需字段。"""

    def __init__(self, config: Any, tokenizer: BinTokenizer) -> None:
        super().__init__()
        self.config = config
        if not hasattr(config, "dataset_det"):
            raise ValueError("config.dataset_det 未配置，无法构建检测数据集")

        det_cfg = config.dataset_det
        self.det_cfg = det_cfg
        self.data_root = Path(det_cfg.data_root).expanduser()
        if not self.data_root.exists():
            raise FileNotFoundError(f"Omni6D 数据目录不存在: {self.data_root}")

        self.image_size = int(det_cfg.get("image_size", 224))
        self.vggt_image_size = int(det_cfg.get("vggt_image_size", 518))

        # 从配置读取相机配置（支持任意数量）
        default_camera_configs = [
            {"name": "top_head",    "crop_scale": (0.90, 0.90), "aspect_ratio": (1.33, 1.33)},
            {"name": "hand_left",   "crop_scale": (0.25, 0.25), "aspect_ratio": (1.33, 1.33)},
            {"name": "hand_right",  "crop_scale": (0.55, 0.55), "aspect_ratio": (1.33, 1.33)},
        ]
        self.camera_configs = det_cfg.get("camera_configs", default_camera_configs)
        self.num_cameras = len(self.camera_configs)

        self.depth_max_percentile = float(det_cfg.get("depth_max_percentile", det_cfg.get("depth_clip_percentile", 99.0)))
        self.depth_min_percentile = float(det_cfg.get("depth_min_percentile", 1.0))
        self.depth_max_value = float(det_cfg.get("depth_max_value", -1.0))
        self.max_samples = det_cfg.get("max_samples", None)
        self.sample_retries = int(det_cfg.get("max_retries", 10))

        self.meta_cache: Dict[str, Dict[str, Any]] = {}
        self.coord_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        # Crop参数管理器（支持随机crop和参数复用）
        self.enable_random_crop = bool(det_cfg.get("enable_random_crop", False))
        self.crop_param_manager = CropParamManager(enable_random=self.enable_random_crop)

        self.samples = self._build_index(det_cfg)
        if len(self.samples) == 0:
            raise RuntimeError("未在指定目录中找到可用的 Omni6D 样本")

        print(f"[Omni6dConsumerDataset] 索引完毕，共 {len(self.samples)} 个样本")

        self.tokenizer = tokenizer

        category_file = 'cache/omni6d_category_list.json'
        if not os.path.exists(category_file):
            meta_folder = os.path.join(self.data_root, 'Meta')
            meta_files = ['obj_meta.json', 'real_obj_meta.json']
            category_all = set()
            for meta_file in meta_files:
                json_path = os.path.join(meta_folder, meta_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    for cat in meta_data['class_list']:
                        name = cat['name'].replace("_", " ").lower()
                        category_all.add(name)

            self.category_all_list = list(category_all)
            self.category_all_set = category_all

            # 存到json
            with open(os.path.join(category_file), "w", encoding="utf-8") as f:
                json.dump(self.category_all_list, f, ensure_ascii=False, indent=4)
            print(f'Category list saved to omni6d_category_list.json, Total {len(self.category_all_list)} categories')
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

    def _build_index(self, det_cfg) -> List[_SampleEntry]:
        """扫描 meta.json，按照类别展开样本索引。"""
        cache_path = det_cfg.get("cache_index_path")
        use_cache = bool(det_cfg.get("use_cache", False))
        raw_patterns = det_cfg.get("meta_patterns", None)
        if raw_patterns is None:
            patterns = ["ROPE_single2/*/*_meta.json"]
        else:
            obj = OmegaConf.to_object(raw_patterns)
            if isinstance(obj, str):
                patterns = [obj]
            else:
                patterns = [str(p) for p in obj]
        if use_cache and cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
                _old_data_root = payload.get("data_root", "")
                _new_data_root = str(self.data_root)
                payload["data_root"] = _new_data_root
            if payload.get("data_root") == _new_data_root and payload.get("patterns") == patterns:
                samples = payload.get("samples", [])
                for item in samples:
                    if "meta_path" in item:
                        item["meta_path"] = item["meta_path"].replace(_old_data_root, _new_data_root)
                return [_SampleEntry(**item) for item in samples]

        entries: List[_SampleEntry] = []
        for pattern in patterns:
            glob_path = str(self.data_root / pattern)
            for meta_path in tqdm(sorted(glob.glob(glob_path))):
                meta_obj = self._load_meta(meta_path)
                prefix = self._resolve_prefix(meta_path)
                try:
                    mask_raw = _load_exr(prefix + "mask.exr", channel="R")
                    mask_ids = np.rint(mask_raw * 255).astype(np.int16)
                    mask_ids[mask_ids == 255] = 0
                except FileNotFoundError:
                    mask_ids = None
                valid_names = {
                    obj.meta.class_name
                    for obj in meta_obj.objects
                    if obj.is_valid and (
                        mask_ids is None or np.any(mask_ids == int(obj.mask_id))
                    )
                }
                rel_subset = os.path.relpath(Path(meta_path).parent.parent, self.data_root)
                for cls in valid_names:
                    entries.append(_SampleEntry(meta_path=meta_path, class_name=cls, subset=rel_subset))

        if self.max_samples:
            entries = entries[: int(self.max_samples)]

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
                            "data_root": str(self.data_root),
                            "patterns": patterns,
                            "samples": [entry.__dict__ for entry in entries],
                        },
                        fp,
                        ensure_ascii=False,
                        indent=2,  # 便于调试
                    )
                # 原子重命名，确保完整性
                shutil.move(temp_path, cache_path)
            except Exception as e:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise


        return entries

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for _ in range(self.sample_retries):
            sample_idx = index % len(self.samples)
            entry = self.samples[sample_idx]
            sample = self._load_sample(entry, sample_idx)
            if sample is not None:
                return sample
            index = torch.randint(0, len(self.samples), ()).item()
        raise RuntimeError("多次尝试后仍无法加载有效样本，请检查数据完整性")

    def _load_meta(self, meta_path: str):
        if meta_path not in self.meta_cache:
            self.meta_cache[meta_path] = OmniDataset.load_meta(meta_path)
        return self.meta_cache[meta_path]

    @staticmethod
    def _resolve_prefix(meta_path: str) -> str:
        if not meta_path.endswith("meta.json"):
            raise ValueError(f"非法 meta 命名: {meta_path}")
        return meta_path[: -len("meta.json")]

    def _load_sample(self, entry: _SampleEntry, sample_idx: int) -> Dict[str, Any] | None:
        meta = self._load_meta(entry.meta_path)
        prefix = self._resolve_prefix(entry.meta_path)

        # 先读取 BGR 图像，避免文件缺失时在 cvtColor 中直接抛异常
        bgr = cv2.imread(prefix + "color.png", cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth_raw = _load_exr(prefix + "depth.exr", channel="Y").astype(np.float32)
        depth = _threshold_depth_map(
            depth_raw,
            max_percentile=self.depth_max_percentile,
            min_percentile=self.depth_min_percentile,
            max_depth=self.depth_max_value,
        )
        mask_raw = _load_exr(prefix + "mask.exr", channel="R")
        mask_ids = np.rint(mask_raw * 255).astype(np.int16)
        mask_ids[mask_ids == 255] = 0

        objects = [
            obj
            for obj in meta.objects
            if obj.is_valid and obj.meta.class_name == entry.class_name
        ]
        if not objects:
            return None

        inst_masks = []
        bbox_list = []
        bbox_side_len = []
        pose_list = []
        class_labels: List[int] = []

        for obj in objects:
            obj_mask = (mask_ids == int(obj.mask_id)).astype(np.uint8)
            inst_masks.append(obj_mask)
            bbox_side_len.append(obj.meta.bbox_side_len)
            pose_list.append(list(obj.quaternion_wxyz) + list(obj.translation))
            class_labels.append(int(obj.meta.class_label))
            ys, xs = np.where(obj_mask > 0)
            if ys.size > 0 and xs.size > 0:
                bbox_list.append([xs.min(), ys.min(), xs.max(), ys.max()])
            else:
                bbox_list.append([0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1])

        if not inst_masks:
            return None
        class_label = class_labels[0] if class_labels else -1

        # ============================================================
        # 数据加载流程 (多摄像头不同crop):
        # 1. 为N个摄像头生成不同crop的图像（根据配置）
        # 2. 原始RGB/depth/mask → crop + resize + pad → 518x518 (vggt)
        # 3. vggt_518 → resize + pad → 224x224
        # 4. 基于224分辨率生成所有标注: bbox, rays, K等
        # ============================================================

        # 使用从配置读取的相机设置
        camera_configs = self.camera_configs

        # 为每个摄像头生成不同crop的图像
        camera_images_224 = []
        camera_images_518 = []  # 新增：保存每个摄像头的518图像
        camera_depths_224 = []
        camera_valid_masks_224 = []
        camera_intrinsics = []

        depth_mask = (depth > 0).astype(np.uint8)

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
            # 使用预设的crop参数确保RGB、depth、mask使用相同的crop
            rgb_518_cam, valid_mask_518_cam, resize_ratio_518_cam, pad_l_518_cam, pad_t_518_cam, crop_l_518_cam, crop_t_518_cam = _crop_resize_with_pad(
                rgb, self.vggt_image_size, self.vggt_image_size,
                pad_value=0, interpolation=cv2.INTER_LINEAR,
                crop_scale=cam_config["crop_scale"],
                aspect_ratio=cam_config["aspect_ratio"],
                debug=True,  # 使用中心crop（但会被preset_crop_params覆盖）
                preset_crop_params=preset_crop_params_518
            )

            # 深度图也要用相同的crop参数（所有摄像头都处理同一个深度图，只是crop不同）
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

            # 调整相机内参 (原始 → crop → 518 → 224)
            intr = meta.camera.intrinsics
            scale_raw = rgb.shape[0] / intr.height
            fx_orig = intr.fx * scale_raw
            fy_orig = intr.fy * scale_raw
            cx_orig = intr.cx * scale_raw
            cy_orig = intr.cy * scale_raw

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
            camera_images_518.append(rgb_518_cam)  # 新增：保存518图像
            camera_depths_224.append(depth_224_cam)
            camera_valid_masks_224.append(valid_mask_224_cam)
            camera_intrinsics.append([fx_cam, fy_cam, cx_cam, cy_cam])

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

        # 步骤3: 为每个摄像头独立处理实例mask、bbox和3D标注
        # 为每个摄像头生成对应crop下的mask、bbox、bbox_side_len、pose
        camera_masks = []  # 每个摄像头的mask列表
        camera_bboxes = []  # 每个摄像头的bbox列表
        camera_bbox_side_len = []  # 每个摄像头的3D尺寸列表
        camera_pose = []  # 每个摄像头的位姿列表
        camera_class_labels = []  # 每个摄像头的类别标签列表

        for cam_idx, cam_config in enumerate(camera_configs):
            # 获取与图像处理相同的crop参数（从缓存中获取，确保完全一致）
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

            inst_mask_tensor_list = []
            bbox_tensor_list = []
            bbox_side_len_list = []
            pose_list_cam = []
            class_label_list = []

            for obj_idx, mask_np in enumerate(inst_masks):
                # mask使用当前摄像头的crop参数: 原始 → 518 → 224
                # 使用预设参数确保与RGB、depth使用相同的crop
                mask_518, _, _, _, _, _, _ = _crop_resize_with_pad(
                    mask_np, self.vggt_image_size, self.vggt_image_size,
                    pad_value=0, interpolation=cv2.INTER_NEAREST,
                    crop_scale=cam_config["crop_scale"],
                    aspect_ratio=cam_config["aspect_ratio"],
                    debug=True,
                    preset_crop_params=preset_crop_params_518
                )
                mask_224, _, _, _, _ = _resize_with_pad(
                    mask_518, self.image_size, self.image_size, pad_value=0, interpolation=cv2.INTER_NEAREST
                )
                mask_224 = (mask_224 > 0).astype(np.float32)

                # 从224分辨率计算bbox
                ys, xs = np.where(mask_224 > 0.5)
                if ys.size == 0 or xs.size == 0:
                    # 物体在当前crop下不可见，跳过
                    continue

                # 计算bbox
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                bbox_width = x2 - x1 + 1
                bbox_height = y2 - y1 + 1
                bbox_area = bbox_width * bbox_height

                # 过滤策略：仅保留面积大于 150 的 bbox。
                # （历史上还试过基于 mask_bbox_ratio、edge_touch、min_size_ratio 的多种规则，
                #  实测下来单一面积阈值已经足够，故保留最简版。）
                if bbox_area < 150:
                    continue

                # 通过过滤，添加到列表（2D和3D标注）
                inst_mask_tensor_list.append(torch.from_numpy(mask_224).unsqueeze(0))
                bbox_tensor_list.append(
                    torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
                )

                # 添加对应物体的3D标注
                bbox_side_len_list.append(bbox_side_len[obj_idx])
                pose_list_cam.append(pose_list[obj_idx])
                class_label_list.append(class_labels[obj_idx])

            # 保存当前摄像头的mask、bbox和3D标注
            camera_masks.append(inst_mask_tensor_list)
            camera_bboxes.append(bbox_tensor_list)
            camera_bbox_side_len.append(bbox_side_len_list)
            camera_pose.append(pose_list_cam)
            camera_class_labels.append(class_label_list)

        # 使用top_head (第0个摄像头) 的mask和bbox作为主要标注
        if camera_masks[0]:
            mask_category = torch.stack(camera_masks[0], dim=0).sum(dim=0).clamp(max=1)
            mask_instance_level = torch.stack(camera_masks[0], dim=0)
        else:
            # 如果top_head没有可见物体，创建空mask
            mask_category = torch.zeros((1, self.image_size, self.image_size), dtype=torch.uint8)
            mask_instance_level = torch.zeros((0, 1, self.image_size, self.image_size), dtype=torch.uint8)

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
        top_fx, top_fy, top_cx, top_cy = camera_intrinsics[0]
        top_K = torch.from_numpy(_intrinsics_to_matrix(top_fx, top_fy, top_cx, top_cy))

        coord_map = self.coord_cache.get((self.image_size, self.image_size))
        if coord_map is None:
            coord_map = _generate_coord_grid(self.image_size, self.image_size)
            self.coord_cache[(self.image_size, self.image_size)] = coord_map
        coord_map = coord_map.to(torch.device("cpu"))

        top_pcl = _depth_to_pcl(top_depth_tensor.squeeze(0), top_K, coord_map, top_depth_mask_tensor.squeeze(0))
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

        if random.random() > self.det_cfg.depth_full_prob:
            camera_know_depth_list = [torch.zeros_like(kd) for kd in camera_know_depth_list]

        if random.random() > self.det_cfg.ray_prob:
            camera_rays_list = [torch.zeros_like(r) for r in camera_rays_list]

        # 归一化每个相机的3D标注（使用统一的avg_scale）
        camera_bbox_side_len_tensors = []
        camera_pose_tensors = []
        for cam_idx in range(len(camera_configs)):
            if camera_bbox_side_len[cam_idx]:  # 如果该相机有可见物体
                cam_bbox_side_len = torch.tensor(camera_bbox_side_len[cam_idx], dtype=torch.float32) / avg_scale
                cam_pose = torch.tensor(camera_pose[cam_idx], dtype=torch.float32)
                cam_pose[:, 4:7] = cam_pose[:, 4:7] / avg_scale
            else:  # 如果该相机没有可见物体，创建空张量
                cam_bbox_side_len = torch.zeros((0, 3), dtype=torch.float32)
                cam_pose = torch.zeros((0, 7), dtype=torch.float32)
            camera_bbox_side_len_tensors.append(cam_bbox_side_len)
            camera_pose_tensors.append(cam_pose)

        # 多相机 RGB 张量 (3,224,224)；多相机深度由 camera_depth_tensors 提供，无需再生成单数 depth/depths。
        images: List[torch.Tensor] = [
            torch.from_numpy(camera_images_224[cam_idx]).permute(2, 0, 1).contiguous()
            for cam_idx in range(self.num_cameras)
        ]

        class_name_clean = entry.class_name.replace("_", " ").lower()
        if class_name_clean not in self.category_all_set:
            print(class_name_clean, 'omni6d', '==================')
            return None

        # 10%负样本
        if random.random() < self.det_cfg.neg_prob and len(self.category_full_set) > 1:
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
                text_label = map_3d_label_to_string(class_name_clean, images, camera_bboxes, camera_pose_tensors, camera_bbox_side_len_tensors)
            else:
                text_label = map_3d_label_to_string_tokenizer(class_name_clean, images, camera_bboxes, camera_pose_tensors, camera_bbox_side_len_tensors, self.tokenizer)
            # print(text_label)

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
            "bbox_2d":              camera_bboxes,                  # List[List[Tensor(4,)]]
            "camera_pose":          camera_pose_tensors,            # List[Tensor(N,7)]
            "camera_bbox_side_len": camera_bbox_side_len_tensors,   # List[Tensor(N,3)]
            "camera_class_labels":  camera_class_labels,            # List[List[int]]
        }

        self.crop_param_manager.clear_cache()

        return sample


# NOTE: The legacy ``DataCollatorForPI0Omni6dConsumerDataset`` has been removed.
# All four detection datasets (omni6d / omni3d / bop / clutter) now share the
# unified ``CollatorForDetectionDataset`` defined in ``collators.py``. Use:
#     from collators import CollatorForDetectionDataset
#     collator = CollatorForDetectionDataset(config=cfg, sub_cfg_key="dataset_det")


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """将四元数转换为旋转矩阵。

    Args:
        quaternion: 四元数张量 [..., 4]，格式为 [w, x, y, z]

    Returns:
        旋转矩阵张量 [..., 3, 3]
    """
    if not isinstance(quaternion, torch.Tensor):
        quaternion = torch.tensor(quaternion, dtype=torch.float32)

    w, x, y, z = torch.unbind(quaternion, dim=-1)

    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    rotation_matrix = torch.stack([
        1.0 - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, 1.0 - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, 1.0 - (txx + tyy)
    ], dim=-1).reshape(quaternion.shape[:-1] + (3, 3))

    return rotation_matrix


# NOTE: Visualization helpers (compute_3d_bbox_vertices_batch, visualize_rays_as_rgb,
# visualize_depth_simple_comparison, project_3d_to_2d, save_image_chw) have been
# removed. They were only used by ad-hoc debugging scripts. See git history if needed.


import hydra
@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="base",
)
def main(cfg):
    bin_tokenizer = BinTokenizer(cfg.statistics_path_6d_dataset)
    train_det_dataset = Omni6dConsumerDataset(config=cfg, tokenizer=bin_tokenizer)
    # Unified collator shared by all four detection datasets.
    # ``sub_cfg_key`` selects which sub-config block the camera_configs
    # are read from (``cfg.dataset_det`` here for omni6d).
    from collators import CollatorForDetectionDataset
    data_det_collator = CollatorForDetectionDataset(config=cfg, sub_cfg_key="dataset_det")
    train_det_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=train_det_dataset, collate_fn=data_det_collator)
    for batch in train_det_dataloader:
        print(batch["observation.images.top_head"].shape)
        print(torch.max(batch["observation.depth_priors.top_head"]), torch.min(batch["observation.depth_priors.top_head"]))
        print(batch["task"])
        print(batch["text_label"])
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
        print(batch["observation.depth_priors.top_head"].shape)
        print(batch["observation.rays.top_head"].shape)
        # plot_2d_3d_objects(images, res, res, batch["task"][0], intrinsics)
        visualize_2d_3d_all(images, res, res, intrinsics, batch["task"][0],)

        visualize_views(images, depths, rays)
        break
if __name__ == "__main__":
    # test()
    main()
