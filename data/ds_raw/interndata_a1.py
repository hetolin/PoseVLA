import torch
from torch.utils.data import Dataset
from collections import defaultdict
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import numpy as np
from mapping_token import text_to_class_attr_dict_tokenizer, map_3d_label_to_string_tokenizer, BinTokenizer, repeat_attribute_to_match, text_to_class_attr_dict, map_3d_label_to_string_tokenizer_ablation
from utils.vis import visualize_2d_3d_all, visualize_traj, visualize_views
import random
import time
import os

REWARD = "next.reward"
ACTION = "action"
OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"

FEATURE_MAPPING = defaultdict(
    lambda : {
        OBS_STATE: ["observation.state"],
        ACTION: ["action"],
    },
    a2d={
        OBS_STATE: [
            "observation.states.joint.position",
            "observation.states.effector.position",
        ],
        ACTION: [
            "actions.joint.position",
            "actions.effector.position",
        ],
    },
    genie1={
        OBS_STATE: [
            "states.left_joint.position",
            "states.right_joint.position",
            "states.left_gripper.position",
            "states.right_gripper.position",
        ],
        ACTION: [
            "actions.left_joint.position",
            "actions.right_joint.position",
            "actions.left_gripper.position",
            "actions.right_gripper.position",
        ],
    },
    arx_lift2={
        OBS_STATE: [
            "states.left_joint.position",
            "states.left_gripper.position",
            "states.right_joint.position",
            "states.right_gripper.position",
        ],
        ACTION: [
            "actions.left_joint.position",
            "actions.left_gripper.position",
            "actions.right_joint.position",
            "actions.right_gripper.position",
        ],
    },
    piper={
        OBS_STATE: [
            "states.left_joint.position",
            "states.left_gripper.position",
            "states.right_joint.position",
            "states.right_gripper.position",
        ],
        ACTION: [
            "actions.left_joint.position",
            "actions.left_gripper.position",
            "actions.right_joint.position",
            "actions.right_gripper.position",
        ],
    },
    r1lite={
        OBS_STATE: [
            'observation.state.left_arm',
            'observation.state.right_arm',
            'observation.state.left_gripper',
            'observation.state.right_gripper',
        ],
        ACTION: [
            "action.left_arm",
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
        ],
    },
    aloha={
        OBS_STATE: [
            'observation.state',
        ],
        ACTION: [
            'action',
        ],
    },
    franka={
        OBS_STATE: [
            "states.joint.position",
            "states.gripper.position",
        ],
        ACTION: [
            "actions.joint.position",
            "actions.gripper.position",
        ],
    },
    panda={
        OBS_STATE: [
            "observation.state",
        ],
        ACTION: [
            "action",
        ],
    }
)
# a1 new
FEATURE_MAPPING["Franka"] = {
    OBS_STATE: [
            "states.joint.position",
            "states.gripper.position",
            'states.tcp_to_robot_pose',
    ],
    ACTION: [
        "actions.joint.position",
        "actions.gripper.position",
        "actions.tcp_to_robot_pose"
    ],
}
FEATURE_MAPPING["ARX Lift-2"] = {
    OBS_STATE: [
            "states.left_joint.position",
            "states.left_gripper.position",
            "states.left_tcp_to_robot_pose",

            "states.right_joint.position",
            "states.right_gripper.position",
            "states.right_tcp_to_robot_pose"
        ],
    ACTION: [
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.left_tcp_to_robot_pose",

        "actions.right_joint.position",
        "actions.right_gripper.position",
        "actions.right_tcp_to_robot_pose",
    ],
}
FEATURE_MAPPING["Genie-1"] = {
    OBS_STATE: [
        "states.left_joint.position",
        "states.left_gripper.position",
        "states.left_tcp_to_robot_pose",

        "states.right_joint.position",
        "states.right_gripper.position",
        "states.right_tcp_to_robot_pose",
    ],
    ACTION: [
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.left_tcp_to_robot_pose",

        "actions.right_joint.position",
        "actions.right_gripper.position",
        "actions.right_tcp_to_robot_pose",
    ],
}
FEATURE_MAPPING["AgileX Split Aloha"] = {
    OBS_STATE: [
        "states.left_joint.position",
        "states.left_gripper.position",
        "states.left_tcp_to_robot_pose",

        "states.right_joint.position",
        "states.right_gripper.position",
        "states.right_tcp_to_robot_pose",
    ],
    ACTION: [
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.left_tcp_to_robot_pose",

        "actions.right_joint.position",
        "actions.right_gripper.position",
        "actions.right_tcp_to_robot_pose",
    ],
}
FEATURE_MAPPING["ARX AC One"] = {
    OBS_STATE: [
        "states.left_joint.position",
        "states.left_gripper.position",
        "states.left_tcp_to_robot_pose",

        "states.right_joint.position",
        "states.right_gripper.position",
        "states.right_tcp_to_robot_pose",
    ],
    ACTION: [
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.left_tcp_to_robot_pose",

        "actions.right_joint.position",
        "actions.right_gripper.position",
        "actions.right_tcp_to_robot_pose",
    ],
}


IMAGE_MAPPING = defaultdict(
    lambda : {
        "observation.image": f"{OBS_IMAGES}.image0",
    },
    arx_lift2={
        "images.rgb.head": f"{OBS_IMAGES}.image0",
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1",
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2",
    },
    piper={
        "images.rgb.head": f"{OBS_IMAGES}.image0",
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1",
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2",
    },
    genie1={
        "images.rgb.head": f"{OBS_IMAGES}.image0",
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1",
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2",
    },
    a2d={
        "observation.images.head": f"{OBS_IMAGES}.image0",
        "observation.images.hand_left": f"{OBS_IMAGES}.image1",
        "observation.images.hand_right": f"{OBS_IMAGES}.image2",
    },
    # todo, make sure what the key names are for franka
    franka={
        "images.rgb.head": f"{OBS_IMAGES}.image0",
        "images.rgb.hand": f"{OBS_IMAGES}.image1",
    },
    r1lite={
        "observation.images.head_rgb": f"{OBS_IMAGES}.image0",
        "observation.images.left_wrist_rgb": f"{OBS_IMAGES}.image1",
        "observation.images.right_wrist_rgb": f"{OBS_IMAGES}.image2",
    },

    aloha={
        "observation.images.cam_high": f"{OBS_IMAGES}.image0",
        "observation.images.cam_left_wrist": f"{OBS_IMAGES}.image1",
        "observation.images.cam_right_wrist": f"{OBS_IMAGES}.image2",
    },
    panda={
        "observation.images.image": f"{OBS_IMAGES}.image0",
        "observation.images.image2": f"{OBS_IMAGES}.image1",
    }
)
# a1 new
IMAGE_MAPPING["Franka"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0",
    "images.rgb.hand": f"{OBS_IMAGES}.image1",
}
IMAGE_MAPPING["ARX Lift-2"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0",
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1",
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["Genie-1"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0",
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1",
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["AgileX Split Aloha"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0",
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1",
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["ARX AC One"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0",
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1",
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2",
}

def to_3x3_intrinsics(intrinsics):
    # intrinsics 预期为 [fx, fy, cx, cy]
    K = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype)
    K[0, 0] = intrinsics[0] # fx
    K[1, 1] = intrinsics[1] # fy
    K[0, 2] = intrinsics[2] # cx
    K[1, 2] = intrinsics[3] # cy
    return K


def pose_to_mat(pose):
    """
    将 [x, y, z, w, qx, qy, qz] 转换为 (N, 4, 4) 或 (4, 4) 矩阵
    """
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    N = pose.shape[0]
    device = pose.device

    # 提取平移和旋转
    res = torch.eye(4, device=device).repeat(N, 1, 1)
    res[:, :3, 3] = pose[:, :3]  # xyz

    # 旋转部分 (w, qx, qy, qz) -> Rotation Matrix
    # Scipy 使用 [qx, qy, qz, w] 格式，需要调整顺序
    quat_wxyz = pose[:, 3:]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]].cpu().numpy()
    rot_mats = R.from_quat(quat_xyzw).as_matrix()

    res[:, :3, :3] = torch.from_numpy(rot_mats).to(device)
    return res.squeeze(0) if N == 1 else res


def project_points(points_3d, K):
    """
    将 (N, 3) 的 3D 点投影到 2D
    points_3d: 在相机坐标系下的坐标
    K: (3, 3) 内参矩阵
    """
    # points_3d: (N, 3) -> (3, N)
    p_cam = points_3d.T
    p_2d = K @ p_cam
    p_2d = p_2d[:2, :] / (p_2d[2, :].clamp(min=1e-6))  # 归一化 z
    return p_2d.T  # (N, 2)

# inv(Extrinsic) @ Action @ T_gripper2ee
def transform_action(action_mat, cam_extrinsic_mat, T_offset):
    # cam_extrinsic_mat 是 cam_to_robot, 所以求逆得到 robot_to_cam
    inv_extrinsic = torch.inverse(cam_extrinsic_mat)
    # 批量矩阵乘法: (N, 4, 4)
    return inv_extrinsic @ action_mat @ T_offset


def mat44_to_quat_trans(mats):
    """
    mats: (N, 4, 4) torch tensor or numpy
    returns: (N, 7) numpy array [w, qx, qy, qz, x, y, z]
    """
    if isinstance(mats, torch.Tensor):
        mats = mats.detach().cpu().numpy()

    N = mats.shape[0]
    pos = mats[:, :3, 3]
    rot_mats = mats[:, :3, :3]

    # scipy 使用 xyzw，我们需要转为 wxyz
    quat_xyzw = R.from_matrix(rot_mats).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    return np.concatenate([quat_wxyz, pos], axis=-1).astype(np.float32)


def is_group_valid(action_2d, W, H):
    """
    判断一组投影点是否在图像内
    """
    if action_2d is None:
        return False
    if len(action_2d) == 0:
        return False
    if isinstance(action_2d, torch.Tensor):
        action_2d = action_2d.detach().cpu().numpy()

    u = action_2d[:, 0]
    v = action_2d[:, 1]
    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.all(valid_mask)


def resize_with_pad(
        img,
        width,
        height,
        depth=None,
        random_crop_ratio_range=(0.8, 1.0),
        pad_value=-1,
        depth_pad_value=0,
        mode="bilinear",
        intrinsic=None,
        pts=None
):
    """
    全流程图像增强：随机裁剪(比例) -> 等比例缩放 -> 居中填充。

    参数:
        img (torch.Tensor): (B, C, H, W) 输入图像。
        width (int): 目标输出宽度。
        height (int): 目标输出高度。
        depth (torch.Tensor, optional): (B, 1, H, W) 或 (B, H, W) 深度图。
        random_crop_ratio_range (tuple, optional): 裁剪比例范围，如 (0.8, 1.0)。
        pad_value (float): RGB 填充值。
        depth_pad_value (float): 深度图填充值。
        mode (str): RGB 缩放模式 ("bilinear", "nearest")。
        intrinsic (torch.Tensor, optional): (B, 3, 3) 或 (3, 3) 相机内参。
        pts (list | np.ndarray | torch.Tensor, optional): 2D 点坐标，支持 []。

    返回:
        padded_img, depth_new, valid_mask, new_intrinsic, pts_new
    """
    if img.ndim != 4:
        raise ValueError(f"img: (b,c,h,w) expected, but got {img.shape}")

    device = img.device
    b, c, cur_height, cur_width = img.shape

    # --- 1. 初始化与备份 ---
    depth_new = None
    if depth is not None:
        depth_new = depth.clone()
        if depth_new.ndim == 3:
            depth_new = depth_new.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

    new_intrinsic = None
    if intrinsic is not None:
        if intrinsic.dim() == 2:
            new_intrinsic = intrinsic[None].expand(b, -1, -1).clone()
        else:
            new_intrinsic = intrinsic.clone()

    # 处理 pts 为 [] 的情况
    pts_new = None
    if pts is not None:
        # 如果是空列表或空数组，直接标记为 None 避开后续计算
        if isinstance(pts, (list, tuple, np.ndarray)) and len(pts) == 0:
            pts_new = pts  # 保持原样返回 []
        else:
            pts_new = torch.as_tensor(pts, device=device, dtype=torch.float32).clone()
            if pts_new.dim() == 2:
                pts_new = pts_new[None].expand(b, -1, -1).clone()

    # # ==========================================
    # # 2. Random Crop 阶段 (按比例裁剪)
    # # ==========================================
    # if random_crop_ratio_range is not None:
    #     min_ratio, max_ratio = random_crop_ratio_range
    #     scale = random.uniform(min_ratio, max_ratio)
    #
    #     crop_h, crop_w = int(cur_height * scale), int(cur_width * scale)
    #     y_start = random.randint(0, cur_height - crop_h)
    #     x_start = random.randint(0, cur_width - crop_w)
    #
    #     # 裁剪操作
    #     img = img[:, :, y_start:y_start + crop_h, x_start:x_start + crop_w]
    #     if depth_new is not None:
    #         depth_new = depth_new[:, :, y_start:y_start + crop_h, x_start:x_start + crop_w]
    #
    #     # 几何更新：减去左上角偏移
    #     if new_intrinsic is not None:
    #         new_intrinsic[:, 0, 2] -= x_start
    #         new_intrinsic[:, 1, 2] -= y_start
    #     if isinstance(pts_new, torch.Tensor):
    #         pts_new[..., 0] -= x_start
    #         pts_new[..., 1] -= y_start
    #
    #     cur_height, cur_width = crop_h, crop_w

    # ==========================================
    # 2. Random Crop 阶段 (不按比例裁剪)
    # ==========================================
    if random_crop_ratio_range is not None:
        min_ratio, max_ratio = random_crop_ratio_range

        # --- 核心修改：分别随机生成高度和宽度的缩放比例 ---
        scale_h = random.uniform(min_ratio, max_ratio)
        scale_w = random.uniform(min_ratio, max_ratio)

        crop_h, crop_w = int(cur_height * scale_h), int(cur_width * scale_w)

        # 随机选择左上角起点
        y_start = random.randint(0, cur_height - crop_h)
        x_start = random.randint(0, cur_width - crop_w)

        # 裁剪操作
        img = img[:, :, y_start:y_start + crop_h, x_start:x_start + crop_w]
        if depth_new is not None:
            depth_new = depth_new[:, :, y_start:y_start + crop_h, x_start:x_start + crop_w]

        # 几何更新：减去左上角偏移 (逻辑不变)
        if new_intrinsic is not None:
            new_intrinsic[:, 0, 2] -= x_start
            new_intrinsic[:, 1, 2] -= y_start
        if isinstance(pts_new, torch.Tensor):
            pts_new[..., 0] -= x_start
            pts_new[..., 1] -= y_start

        cur_height, cur_width = crop_h, crop_w


    # ==========================================
    # 3. Resize 阶段 (等比例缩放)
    # ==========================================
    ratio = max(cur_width / width, cur_height / height)
    resized_height, resized_width = int(cur_height / ratio), int(cur_width / ratio)

    # RGB 缩放
    img = F.interpolate(img, size=(resized_height, resized_width), mode=mode,
                        align_corners=(False if mode != "nearest" else None))
    # Depth 缩放 (强制 nearest)
    if depth_new is not None:
        depth_new = F.interpolate(depth_new, size=(resized_height, resized_width), mode="nearest")

    scale_x, scale_y = resized_width / cur_width, resized_height / cur_height

    # 几何更新：缩放焦距和主点
    if new_intrinsic is not None:
        new_intrinsic[:, 0, 0] *= scale_x
        new_intrinsic[:, 1, 1] *= scale_y
        new_intrinsic[:, 0, 2] *= scale_x
        new_intrinsic[:, 1, 2] *= scale_y
    if isinstance(pts_new, torch.Tensor):
        pts_new[..., 0] *= scale_x
        pts_new[..., 1] *= scale_y

    # ==========================================
    # 4. Pad 阶段 (居中填充)
    # ==========================================
    pad_h, pad_w = max(0, height - resized_height), max(0, width - resized_width)
    ph, pw = pad_h // 2, pad_w // 2

    padded_img = F.pad(img, (pw, pad_w - pw, ph, pad_h - ph), value=pad_value)
    if depth_new is not None:
        depth_new = F.pad(depth_new, (pw, pad_w - pw, ph, pad_h - ph), value=depth_pad_value)

    # 几何更新：加上填充偏移
    if new_intrinsic is not None:
        new_intrinsic[:, 0, 2] += pw
        new_intrinsic[:, 1, 2] += ph
    if isinstance(pts_new, torch.Tensor):
        pts_new[..., 0] += pw
        pts_new[..., 1] += ph

    # ==========================================
    # 5. Mask 生成
    # ==========================================
    valid_mask = torch.zeros((b, 1, height, width), device=device, dtype=padded_img.dtype)
    valid_mask[..., ph:ph + resized_height, pw:pw + resized_width] = 1

    return padded_img, depth_new, valid_mask, new_intrinsic, pts_new


def _generate_coord_grid(width: int, height: int):
    """生成像素坐标网格 (2, H, W)"""
    # 生成 y 坐标 (0~H-1) 和 x 坐标 (0~W-1)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij'
    )
    return torch.stack([x, y], dim=0)  # (2, H, W)


def _generate_rays(height: int, width: int, K: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    根据内参生成每个像素的视线向量 (3, H, W)
    K: (3, 3)
    valid_mask: (H, W) 或 (1, H, W)
    """
    device = K.device
    # 1. 生成坐标网格并移至对应设备
    coord_map = _generate_coord_grid(width, height).to(device)  # (2, H, W)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 2. 计算归一化平面坐标 (x, y, 1)
    dirs = torch.stack(
        [
            (coord_map[0] - cx) / fx,
            (coord_map[1] - cy) / fy,
            torch.ones_like(coord_map[0]),
        ],
        dim=0,
    )  # (3, H, W)

    # 3. 归一化为单位向量 (根据你的需求，如果需要单位向量则取消下面注释)
    # norm = torch.linalg.norm(dirs, dim=0, keepdim=True).clamp_min(1e-6)
    # dirs = dirs / norm

    # 4. 将 Padding 区域清零
    # 确保 mask 是 (H, W) 形状
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.squeeze(0)

    dirs[:, valid_mask == 0] = 0

    return dirs


def pad_vector(vector, new_dim):
    """填充特征维度 (..., current_dim) -> (..., new_dim)"""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = np.zeros(shape, dtype=vector.dtype)
    new_vector[..., :current_dim] = vector
    return new_vector


def pad_action_sequence(actions, chunk_size):
    """填充时间维度：如果动作序列不够长，重复最后一个动作"""
    curr_len = actions.shape[0]
    if curr_len >= chunk_size:
        return actions[:chunk_size]

    pad_len = chunk_size - curr_len
    last_action = actions[-1:]  # 保持 (1, dim) 形状
    padding = np.repeat(last_action, pad_len, axis=0)
    return np.concatenate([actions, padding], axis=0)

class InternDataA1LeRobotDatasetWrapper(Dataset):
    def __init__(self, cfg, lerobot_dataset, tokenizer: BinTokenizer = None,):
        self.cfg = cfg
        self.dataset = lerobot_dataset
        self.tokenizer = tokenizer

        # 存储：(子数据集对象, 该 Episode 的起始帧, 该 Episode 的结束帧)
        self.episode_ranges = []

        # 遍历所有子数据集
        for sub_ds in self.dataset._datasets:
            # 获取你刚才打印出来的那个 episodes 表
            episodes_df = sub_ds.meta.episodes

            # 直接提取起始和结束索引列
            # dataset_from_index: 该 episode 的第一帧位置
            # dataset_to_index: 该 episode 的最后一帧位置 + 1
            starts = episodes_df["dataset_from_index"]
            ends = episodes_df["dataset_to_index"]

            # 将它们存入映射表
            for s, e in zip(starts, ends):
                # 确保是整数
                self.episode_ranges.append((sub_ds, int(s), int(e)))

    def __len__(self):
        # return len(self.dataset)
        # return self.dataset.num_episodes
        return len(self.episode_ranges)

    def __getitem__(self, idx):
        # # 1. 获取原始数据
        # item = self.dataset[idx]

        # 设定一个最大尝试次数，防止整个数据集都坏了导致死循环
        max_episode_retries = 5

        for _ in range(max_episode_retries):
            sub_ds, start_frame, end_frame = self.episode_ranges[idx]

            # 1. 尝试在当前 Episode 里随机找一帧
            for _ in range(10):
                try:
                    random_idx = random.randint(start_frame, end_frame - 1)
                    res = sub_ds[random_idx]
                    return self._post_process(res)
                except Exception:
                    # 如果这一帧报错（时间戳对不齐等），换一帧再试
                    continue

            new_idx = random.randint(0, len(self.episode_ranges) - 1)
            print(f"Warning: Episode {idx} seems broken, switching to Episode {new_idx}")
            idx = new_idx

        raise RuntimeError(
            "Failed to load data after trying multiple episodes. Please check your dataset or environment.")

        return batch

    def _post_process(self, item):
        data_dict = {}
        device = item["images.rgb.head"].device
        target_W = self.cfg.dataset.image_size
        target_H = self.cfg.dataset.image_size
        # 新增配置
        CHUNK_SIZE = self.cfg.dataset.action_chunk_size
        STATE_DIM = self.cfg.dataset.state_dim

        sample_step = 10

        # --- 0. 基础信息提取 (区分单双臂) ---
        if item.get("images.rgb.hand_left") is not None:
            # 【双臂模式 Dual Arm】
            images = [item["images.rgb.head"], item["images.rgb.hand_left"], item["images.rgb.hand_right"]]
            intri_head = to_3x3_intrinsics(item["head_camera_intrinsics"])
            intri_left = to_3x3_intrinsics(item["hand_left_camera_intrinsics"])
            intri_right = to_3x3_intrinsics(item["hand_right_camera_intrinsics"])
            extrinsic_head = pose_to_mat(item["head_camera_to_robot_extrinsics"])
            extrinsic_left = pose_to_mat(item["hand_left_camera_to_robot_extrinsics"])
            extrinsic_right = pose_to_mat(item["hand_right_camera_to_robot_extrinsics"])

            action_l_mat = pose_to_mat(item["actions.left_tcp_to_robot_pose"][::sample_step, ...])
            action_r_mat = pose_to_mat(item["actions.right_tcp_to_robot_pose"][::sample_step, ...])

            # 提取 State/Action (双臂)
            s_l, g_l = item["states.left_tcp_to_robot_pose"], item["states.left_gripper.position"]
            s_r, g_r = item["states.right_tcp_to_robot_pose"], item["states.right_gripper.position"]
            a_l, ag_l = item["actions.left_tcp_to_robot_pose"], item["actions.left_gripper.position"]
            a_r, ag_r = item["actions.right_tcp_to_robot_pose"], item["actions.right_gripper.position"]

        elif item.get("images.rgb.hand") is not None:
            # 【单臂模式 Single Arm】
            images = [item["images.rgb.head"], item["images.rgb.hand"], torch.zeros_like(item["images.rgb.head"])]
            intri_head = to_3x3_intrinsics(item["head_camera_intrinsics"])
            intri_left = to_3x3_intrinsics(item["hand_camera_intrinsics"])
            intri_right = []
            extrinsic_head = pose_to_mat(item["head_camera_to_robot_extrinsics"])
            extrinsic_left = pose_to_mat(item["hand_camera_to_robot_extrinsics"])

            extrinsic_right = []
            action_l_mat = pose_to_mat(item["actions.tcp_to_robot_pose"])[::sample_step, ...]
            action_r_mat = []

            # 提取 State/Action (单臂，右臂补零)
            s_l, g_l = item["states.tcp_to_robot_pose"], item["states.gripper.position"]
            s_r, g_r = np.zeros_like(s_l), np.zeros_like(g_l)
            a_l, ag_l = item["actions.tcp_to_robot_pose"], item["actions.gripper.position"]
            a_r, ag_r = np.zeros_like(a_l), np.zeros_like(ag_l)

        else:
            return {}

        # --- State & Action 拼接与填充 (顺序: 左臂, 左夹爪, 右臂, 右夹爪) ---
        # 使用 np.atleast_1d 确保标量变为 (1,) 向量，以便与 (7,) 向量拼接
        combined_state = np.concatenate([
            s_l, np.atleast_1d(g_l),
            s_r, np.atleast_1d(g_r)
        ], axis=-1)[None, ...]

        # Action 通常是 (T, 7) 和 (T,)，需要将 (T,) 变为 (T, 1)
        # 如果 ag_l 已经是 (T, 1) 则 reshape 不起作用，如果是 (T,) 则变为 (T, 1)
        combined_action = np.concatenate([
            a_l, ag_l.reshape(-1, 1),
            a_r, ag_r.reshape(-1, 1)
        ], axis=-1)

        # Action 时间轴填充 (Repeat Last)
        combined_action = pad_action_sequence(combined_action, CHUNK_SIZE)

        # 特征维度填充 (Pad to STATE_DIM)
        data_dict["states"] = torch.from_numpy(pad_vector(combined_state, STATE_DIM)).float().to(device)
        data_dict["actions"] = torch.from_numpy(pad_vector(combined_action, STATE_DIM)).float().to(device)

        _, H, W = images[0].shape
        T_gripper2tcp = torch.eye(4, device=device)

        # --- 1. 计算投影 (3相机 x 2手臂) ---
        poses_3d = [[[] for _ in range(2)] for _ in range(3)]
        poses_2d = [[[] for _ in range(2)] for _ in range(3)]
        intrinsics_list = [intri_head, intri_left, intri_right]
        extrinsics_list = [extrinsic_head, extrinsic_left, extrinsic_right]
        actions_list = [action_l_mat, action_r_mat]

        for cam_idx in range(3):
            for arm_idx in range(2):
                cam_ext, cam_int, arm_act = extrinsics_list[cam_idx], intrinsics_list[cam_idx], actions_list[arm_idx]
                if len(cam_ext) > 0 and len(arm_act) > 0:
                    p3d = transform_action(arm_act, cam_ext, T_gripper2tcp)
                    p2d = project_points(p3d[:, :3, 3], cam_int)
                    poses_3d[cam_idx][arm_idx] = p3d
                    poses_2d[cam_idx][arm_idx] = p2d

        # --- 2. 视图组织与可见性过滤 ---
        multi_poses, multi_2ds, class_names = [], [], []

        # 视图策略配置 [左臂, 右臂]
        view_policies = [
            # Head: 左右手都启用(Enabled)，但都不强制(AlwaysVisible)，即都检查
            {"enabled": [True, True], "always_visible": [False, False], "labels": ["left arm", "right arm"]},
            # Left Wrist: 启用左手并强制，禁用右手(直接不显示)
            {"enabled": [True, False], "always_visible": [True, False], "labels": ["left arm", "right arm"]},
            # Right Wrist: 禁用左手(直接不显示)，启用右手并强制
            {"enabled": [False, True], "always_visible": [False, True], "labels": ["left arm", "right arm"]}
        ]

        for cam_idx in range(3):
            names_v, poses_v, p2ds_v = [], [], []
            policy = view_policies[cam_idx]

            for arm_idx in range(2):
                # A. 如果该手臂在当前相机策略中未启用，直接跳过
                if not policy["enabled"][arm_idx]:
                    continue

                p3d = poses_3d[cam_idx][arm_idx]
                p2d = poses_2d[cam_idx][arm_idx]
                if len(p3d) == 0: continue

                # B. 校验可见性：强制显示 OR 通过边界检查
                if policy["always_visible"][arm_idx] or is_group_valid(p2d[0:1, :], W, H):
                    label = policy["labels"][arm_idx]
                    names_v += [f"{label} {i}" for i in range(len(p3d))]
                    poses_v.append(p3d)
                    p2ds_v.append(p2d)

            class_names.append(names_v)
            if len(poses_v) > 0:
                c_poses = torch.cat(poses_v, dim=0)
                c_2ds = torch.cat(p2ds_v, dim=0)
                multi_poses.append(torch.from_numpy(mat44_to_quat_trans(c_poses)).to(device))
                multi_2ds.append(c_2ds)
            else:
                multi_poses.append([])
                multi_2ds.append([])


        # --- 3. 缩放、Padding 与 视线生成 ---
        padded_images, padded_masks, padded_intrinsics, padded_multi_2ds = [], [], [], []
        camera_rays = []

        for i in range(3):
            k_in = intrinsics_list[i] if not isinstance(intrinsics_list[i], list) else None
            p_img, _, v_mask, p_K, p_pts = resize_with_pad(
                img=images[i].unsqueeze(0), width=target_W, height=target_H,
                random_crop_ratio_range=self.cfg.dataset.get("random_crop_ratio_range", (0.8, 1.0)),
                pad_value=0, mode="bilinear", intrinsic=k_in, pts=multi_2ds[i]
            )
            padded_images.append(p_img.squeeze(0) * 255.)
            padded_masks.append(v_mask.squeeze(0))

            # --- 修改点：如果 p_K 为空，填充 3x3 零矩阵张量 ---
            # padded_intrinsics.append(p_K.squeeze(0) if p_K is not None else [])
            padded_intrinsics.append(p_K.squeeze(0) if p_K is not None else torch.zeros((3, 3), device=device))
            padded_multi_2ds.append(p_pts.squeeze(0) if len(p_pts) > 0 else [])

            # 使用 .any() 检查张量是否包含非零元素
            if isinstance(padded_intrinsics[-1], torch.Tensor) and padded_intrinsics[-1].any():
                rays = _generate_rays(target_H, target_W, padded_intrinsics[-1], padded_masks[-1])
                camera_rays.append(rays)
            else:
                camera_rays.append(torch.zeros((3, target_H, target_W), device=device))

        if random.random() > self.cfg.ray_prob:
            camera_rays = [torch.zeros_like(ray) for ray in camera_rays]

        # --- 4. 组装输出 ---
        data_dict.update({
            "images": padded_images, #(3,H,W)
            "camera_rays": camera_rays, #(3,H,W)
            "intrinsics": padded_intrinsics, #(3,3)
            "camera_pose": multi_poses #(N,7)
        })

        # Dummy Inputs
        opt = {"device": device, "dtype": images[0].dtype}
        data_dict["depths"] = [torch.zeros((target_H, target_W), **opt) for _ in range(3)]
        data_dict["depth_priors"] = [torch.zeros((2, target_H, target_W), **opt) for _ in range(3)]
        data_dict["dets"] = [torch.zeros((1, target_H, target_W), **opt) for _ in range(3)]

        text_labels = map_3d_label_to_string_tokenizer(class_names=class_names,
                                                       multi_images=data_dict["images"],
                                                       multi_2ds=padded_multi_2ds,
                                                       multi_poses=multi_poses,
                                                       multi_sizes=repeat_attribute_to_match(multi_poses, None),
                                                       tokenizer=self.tokenizer,
                                                       near2far=False)


        #TODO:  ablation: output pose directly
        # class_names = []
        # multi_poses = []
        # if len(action_l_mat) > 0:
        #     class_names_v = []
        #     for i in range(action_l_mat.shape[0]):
        #         class_names_v.append(f"left arm {i}")
        #         multi_poses.append(torch.from_numpy(mat44_to_quat_trans(action_l_mat)))
        #     class_names.append(class_names_v)
        #
        # if len(action_r_mat) > 0:
        #     class_names_v = []
        #     for i in range(action_r_mat.shape[0]):
        #         class_names_v.append(f"right arm {i}")
        #         multi_poses.append(torch.from_numpy(mat44_to_quat_trans(action_r_mat)))
        #     class_names.append(class_names_v)
        #
        # text_labels = map_3d_label_to_string_tokenizer_ablation(class_names=class_names,
        #                                                multi_images=data_dict["images"],
        #                                                multi_2ds=repeat_attribute_to_match(multi_poses, None),
        #                                                multi_poses=multi_poses,
        #                                                multi_sizes=repeat_attribute_to_match(multi_poses, None),
        #                                                tokenizer=self.tokenizer,
        #                                                near2far=False)

        data_dict["text_labels"] = text_labels

        data_dict["dataset_name"] = item["task"]
        data_dict["instructions"] = item["task"]
        data_dict["data_idx"] = item["episode_index"]

        # sanity check
        # text_label = map_3d_label_to_string_tokenizer(class_names=class_names,
        #                                               multi_images=data_dict["images"],
        #                                               multi_2ds=padded_multi_2ds,
        #                                               multi_poses=multi_poses,
        #                                               multi_sizes=repeat_attribute_to_match(multi_poses, None),
        #                                               tokenizer=self.tokenizer,
        #                                               near2far=False)
        #
        # images = {"image0": images[0], "image1": images[1], "image2": images[2]}
        # intrinsics = {"image0": padded_intrinsics[0], "image1": padded_intrinsics[1], "image2": padded_intrinsics[2]}
        # res = text_to_class_attr_dict_tokenizer(text_label, self.tokenizer)
        # print(text_label)
        # visualize_2d_3d_all(images, res, res, intrinsics, data_dict["instructions"])

        return data_dict

    # 关键：属性透传
    # 这样包装后的对象依然可以访问 dataset.stats, dataset.meta 等 LeRobot 特有属性
    def __getattr__(self, name):
        return getattr(self.dataset, name)


def load_interndata_a1(pi0_config, cfg, bin_tokenizer, make_dataset_fn, yaml_names):
    """
    加载 InternData A1 数据集并进行封装

    参数:
        pi0_config: pi0 模型的配置对象
        cfg: 全局配置对象，需包含 dataset_lerobot 属性
        bin_tokenizer: 动作离散化的 tokenizer
        make_dataset_fn: 外部传入的 make_dataset 函数
        yaml_names: 包含需要加载的组名列表 (例如 ["lerobot_group01", ...])

    返回:
        all_wrapped_datasets: 包含所有封装后数据集的列表
    """
    # 强制开启离线模式，跳过 HF 联网检查，提速关键
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    all_wrapped_datasets = []
    start_time = time.time()

    # 统计配置中实际存在的组
    available_groups = [y for y in yaml_names if hasattr(cfg.dataset_lerobot, y)]
    num_to_load = len(available_groups)

    print("\n" + "=" * 60)
    print(f"开始加载 InternData A1 数据集")
    print(f"计划加载组数: {len(yaml_names)} | 实际匹配到配置: {num_to_load}")
    print("=" * 60)

    count = 0
    for yaml_name in yaml_names:
        if hasattr(cfg.dataset_lerobot, yaml_name):
            count += 1
            group_start_time = time.time()
            print(f"[{count}/{num_to_load}] 正在处理: {yaml_name} ... ", end="", flush=True)

            try:
                # 获取对应组的配置
                group_cfg = getattr(cfg.dataset_lerobot, yaml_name)

                # 1. 调用外部传入的 make_dataset 函数
                sub_dataset = make_dataset_fn(pi0_config, group_cfg)

                # 2. 封装数据集
                wrapped_dataset = InternDataA1LeRobotDatasetWrapper(
                    cfg,
                    sub_dataset,
                    bin_tokenizer
                )

                all_wrapped_datasets.append(wrapped_dataset)

                duration = time.time() - group_start_time
                print(f"成功! (耗时: {duration:.2f}s)")

            except Exception as e:
                print(f"\n[错误] 加载 {yaml_name} 失败: {e}")
        else:
            # 如果 yaml_names 里的名字在 cfg 里找不到，打印提示
            print(f"[-] 跳过: {yaml_name} (配置中未定义)")

    total_duration = time.time() - start_time
    print("=" * 60)
    print(f"加载总结:")
    print(f" - 成功封装组数: {len(all_wrapped_datasets)} / {len(yaml_names)}")
    print(f" - 总计耗时: {total_duration:.2f}s")
    print("=" * 60 + "\n")

    return all_wrapped_datasets
