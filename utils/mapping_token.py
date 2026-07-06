import numpy as np
from pathlib import Path
import pickle
import os
import torch
import re
import regex

class BinTokenizer:
    def __init__(self, file='./bin_stats/nonuniform_bins.pkl'):
        # 1. 默认值
        self.EXTRA_LOC_TOKENS = self.gen_tokens("loc", 1024)
        self.EXTRA_SEG_TOKENS = self.gen_tokens("seg", 128)
        self.EXTRA_IMAGE_TOKENS = [item for pair in zip(self.gen_tokens('image', 3),
                                                        self.gen_tokens('/image', 3)) for item in pair]
        self.EXTRA_NO_OBJ_TOKENS = "<no_object>"
        # ====== 硬编码开关：True 使用 HunyuanVL 原生 <box> 格式，False 使用 <loc> 格式 ======
        self.USE_BOX_FORMAT = False
        self.EXTRA_TRANS_XY_TOKENS = self.gen_tokens("transXY", 2048)
        self.EXTRA_TRANS_Z_TOKENS = self.gen_tokens("transZ", 2048)
        self.EXTRA_ROT_XYZ_TOKENS = self.gen_tokens("rot", 360)
        self.EXTRA_SIZE_XYZ_TOKENS = self.gen_tokens("size", 2048)

        # 深度估计 token (2048个, 0-5m, log scale)
        self.EXTRA_DEPTH_TOKENS = self.gen_tokens("depth", 2048)
        self.DEPTH_MAX_METER = 5.0
        self.DEPTH_NUM_BINS = 2048

        # 双目匹配可见性 token
        self.EXTRA_STEREO_TOKENS = {
            "no_pair": "<stereo_no_pair>",
            "invisible": "<stereo_invisible>",
            "visible": "<stereo_visible>",
        }

        # 2. 从文件加载（如有）
        if file and os.path.exists(file):
            with open(file, 'rb') as f:
                data = pickle.load(f)
            # 用 pkl 里有的覆盖，没有的还是默认
            self.EXTRA_TRANS_XY_TOKENS = data.get("token_list_txy", self.EXTRA_TRANS_XY_TOKENS)
            self.EXTRA_TRANS_Z_TOKENS = data.get("token_list_tz", self.EXTRA_TRANS_Z_TOKENS)
            self.EXTRA_ROT_XYZ_TOKENS = data.get("token_list_rxyz", self.EXTRA_ROT_XYZ_TOKENS)
            self.EXTRA_SIZE_XYZ_TOKENS = data.get("token_list_sxyz", self.EXTRA_SIZE_XYZ_TOKENS)

            self.percentiles_txy = data.get("percentiles_txy", None)
            self.percentiles_tz = data.get("percentiles_tz", None)
            self.percentiles_sxyz = data.get("percentiles_sxyz", None)
        else:
            self.percentiles_txy = None
            self.percentiles_tz = None
            self.percentiles_sxyz = None

        print(self.EXTRA_LOC_TOKENS[0], self.EXTRA_LOC_TOKENS[-1])
        print(self.EXTRA_SEG_TOKENS[0], self.EXTRA_SEG_TOKENS[-1])
        print(self.EXTRA_IMAGE_TOKENS[0], self.EXTRA_IMAGE_TOKENS[-1])
        print(self.EXTRA_TRANS_XY_TOKENS[0], self.EXTRA_TRANS_XY_TOKENS[-1])
        print(self.EXTRA_TRANS_Z_TOKENS[0], self.EXTRA_TRANS_Z_TOKENS[-1])
        print(self.EXTRA_ROT_XYZ_TOKENS[0], self.EXTRA_ROT_XYZ_TOKENS[-1])
        print(self.EXTRA_SIZE_XYZ_TOKENS[0], self.EXTRA_SIZE_XYZ_TOKENS[-1])
        print(self.EXTRA_DEPTH_TOKENS[0], self.EXTRA_DEPTH_TOKENS[-1])
        print(self.EXTRA_STEREO_TOKENS)
        print(self.EXTRA_NO_OBJ_TOKENS)
        print(self.USE_BOX_FORMAT)

    def gen_tokens(self, prefix, num):
        width = len(str(num-1))
        return [f"<{prefix}{str(i).zfill(width)}>" for i in range(num)]

    def encode_values(self, values, percentiles, token_list):
        '''
        values: 待编码的一维数组(1,)(N,)
        percentiles: 区间分割点数组(len=N+1)
        token_list: 每一组的token(len=N)
        返回: [tokens, bin_indices]
        '''
        values = np.array(values).reshape(-1)
        bins = np.digitize(values, percentiles, right=False)
        bins = np.clip(bins - 1, 0, len(token_list) - 1)  # 保证在合法区间
        tokens = [token_list[i] for i in bins]

        if len(tokens) == 1:
            tokens = tokens[0]

        return tokens, bins

    def decode_tokens(self, tokens, token_list, percentiles):
        '''
        tokens: 与token_list对应的一组token（可重复）
        token_list: 全体可能token（有顺序）
        percentiles: 区间分割点数组(len=N+1)
        返回: [bin_idx, 每个token的数值区间]
        '''
        tokens = np.array(tokens).reshape(-1)  # 保证是1维
        token_list = list(token_list)  # 保证能用enumerate和index访问
        token2bin = {token: i for i, token in enumerate(token_list)}
        bins = np.array([token2bin[token] for token in tokens])

        # 使用 broadcasting 和 np.where 找索引（不太优雅，但可以避免Python循环）
        # bins = np.searchsorted(token_list, tokens)

        intervals = [ (percentiles[i], percentiles[i+1]) for i in bins ]
        mids = [(percentiles[i] + percentiles[i + 1]) / 2.0 for i in bins]

        if len(mids) == 1:
            mids = mids[0]

        return mids, bins


    # non-uniform encode-decode
    def encode_txy(self, values):
        if self.percentiles_txy is None:
            print("percentiles_txy未初始化！")
        return self.encode_values(values, self.percentiles_txy, self.EXTRA_TRANS_XY_TOKENS)

    def encode_tz(self, values):
        if self.percentiles_tz is None:
            print("percentiles_tz未初始化！")
        return self.encode_values(values, self.percentiles_tz, self.EXTRA_TRANS_Z_TOKENS)

    def encode_sxyz(self, values):
        if self.percentiles_sxyz is None:
            print("percentiles_sxyz未初始化！")
        return self.encode_values(values, self.percentiles_sxyz, self.EXTRA_SIZE_XYZ_TOKENS)

    def decode_txy(self, tokens):
        if self.percentiles_txy is None:
            print("percentiles_txy未初始化！")
        return self.decode_tokens(tokens, self.EXTRA_TRANS_XY_TOKENS, self.percentiles_txy)

    def decode_tz(self, tokens):
        if self.percentiles_tz is None:
            print("percentiles_tz未初始化！")
        return self.decode_tokens(tokens, self.EXTRA_TRANS_Z_TOKENS, self.percentiles_tz)

    def decode_sxyz(self, tokens):
        if self.percentiles_sxyz is None:
            print("percentiles_sxyz未初始化！")
        return self.decode_tokens(tokens, self.EXTRA_SIZE_XYZ_TOKENS, self.percentiles_sxyz)

    # ---- Depth Log-Scale Encode / Decode ----

    def encode_depth_log(self, depth_meter):
        """
        将深度值 (米) 编码为 depth token。
        使用 log scale: bin = log(1 + depth) / log(1 + max_depth) * (num_bins - 1)
        近处分辨率高，远处分辨率低。
        depth_meter: float, 深度值 (米), 范围 [0, DEPTH_MAX_METER]
        返回: (token_str, bin_idx)
        """
        depth_meter = float(np.clip(depth_meter, 0, self.DEPTH_MAX_METER))
        # log scale 映射
        bin_idx = int(np.log1p(depth_meter) / np.log1p(self.DEPTH_MAX_METER) * (self.DEPTH_NUM_BINS - 1))
        bin_idx = int(np.clip(bin_idx, 0, self.DEPTH_NUM_BINS - 1))
        token = self.EXTRA_DEPTH_TOKENS[bin_idx]
        return token, bin_idx

    def decode_depth_log(self, token_str):
        """
        将 depth token 字符串解码为深度值 (米)。
        token_str: str, 如 "0512" (纯数字部分) 或 "<depth0512>"
        返回: float, 深度值 (米)
        """
        # 支持传入纯数字或完整 token
        if isinstance(token_str, str) and token_str.startswith('<'):
            digits = re.findall(r'<depth(\d+)>', token_str)
            if digits:
                bin_idx = int(digits[0])
            else:
                raise ValueError(f"无法解析 depth token: {token_str}")
        else:
            bin_idx = int(token_str)
        bin_idx = int(np.clip(bin_idx, 0, self.DEPTH_NUM_BINS - 1))
        # 反向 log scale
        depth_meter = np.expm1(bin_idx / (self.DEPTH_NUM_BINS - 1) * np.log1p(self.DEPTH_MAX_METER))
        return float(depth_meter)

    # ---- Box Format (HunyuanVL 原生格式) Encode / Decode ----

    def encode_box(self, coords, w, h):
        """
        将2D bbox编码为HunyuanVL原生的<box>(x1,y1),(x2,y2)</box>格式。
        坐标归一化到 [0, 1000] 范围。
        """
        coords = np.array(coords, dtype=np.float64)
        BOX_MAX = 1000
        if len(coords) == 4:
            x1 = int(np.clip(coords[0] / w * BOX_MAX, 0, BOX_MAX))
            y1 = int(np.clip(coords[1] / h * BOX_MAX, 0, BOX_MAX))
            x2 = int(np.clip(coords[2] / w * BOX_MAX, 0, BOX_MAX))
            y2 = int(np.clip(coords[3] / h * BOX_MAX, 0, BOX_MAX))
            return f"<box>({x1},{y1}),({x2},{y2})</box>"
        elif len(coords) == 2:
            x1 = int(np.clip(coords[0] / w * BOX_MAX, 0, BOX_MAX))
            y1 = int(np.clip(coords[1] / h * BOX_MAX, 0, BOX_MAX))
            return f"<box>({x1},{y1})</box>"
        else:
            raise ValueError(f"encode_box: coords长度必须为2或4, 实际为{len(coords)}")

    def decode_box(self, box_string):
        """
        将<box>(x1,y1),(x2,y2)</box>格式解码为归一化坐标 [0, 1]。
        """
        BOX_MAX = 1000
        nums = re.findall(r'\d+', box_string)
        coords = [int(n) / BOX_MAX for n in nums]
        return np.array(coords)

    # uniform encode-decode

    def encode_loc_uniform(self, coords, w, h):
        """
        coords: list 或 np.array, [x1, y1] 或 [x1, y1, x2, y2]
        w: 图像宽度
        h: 图像高度
        返回: tokens, bins
        """
        LOC_MAX = len(self.EXTRA_LOC_TOKENS) - 1
        coords = np.array(coords)
        # 标准化为桶号
        bins = []
        if len(coords) >= 2:
            x1_bin = int(np.clip(coords[0] / w * LOC_MAX, 0, LOC_MAX))
            y1_bin = int(np.clip(coords[1] / h * LOC_MAX, 0, LOC_MAX))
            bins.extend([x1_bin, y1_bin])
        if len(coords) == 4:
            x2_bin = int(np.clip(coords[2] / w * LOC_MAX, 0, LOC_MAX))
            y2_bin = int(np.clip(coords[3] / h * LOC_MAX, 0, LOC_MAX))
            bins.extend([x2_bin, y2_bin])
        tokens = [self.EXTRA_LOC_TOKENS[b] for b in bins]
        return tokens, bins

    def decode_loc_uniform(self, tokens):
        """
        tokens: ["0675", "0123", ...]
        w: 图像宽度
        h: 图像高度
        返回: np.array([x1, y1]) 或 [x1, y1, x2, y2] 0~1
        """
        LOC_MAX = len(self.EXTRA_LOC_TOKENS) - 1
        bins = [int(token) for token in tokens]
        coords = []
        if len(bins) >= 2:
            x1 = bins[0] / LOC_MAX
            y1 = bins[1] / LOC_MAX
            coords.extend([x1, y1])
        if len(bins) == 4:
            x2 = bins[2] / LOC_MAX
            y2 = bins[3] / LOC_MAX
            coords.extend([x2, y2])
        return np.array(coords)

    def encode_txyz_uniform(self, xyz):
        """
        xyz: list 或 np.array, [tx, ty, tz]，要求范围在[-1, 1]
        返回: tokens, bins
        """
        TRANS_MAX = len(self.EXTRA_TRANS_TOKENS) - 1
        xyz = np.array(xyz)
        bins = ((xyz + 1.0) / 2.0 * TRANS_MAX).astype(int)
        bins = np.clip(bins, 0, TRANS_MAX)
        tokens = [self.EXTRA_TRANS_TOKENS[b] for b in bins]
        return tokens, bins

    def decode_txyz_uniform(self, tokens):
        """
        tokens: ["0100", "0909", "2047"] 字符串列表
        返回: np.array([tx, ty, tz])，范围在[-1, 1]
        """
        TRANS_MAX = len(self.EXTRA_TRANS_TOKENS) - 1
        bins = np.array([int(token) for token in tokens])
        xyz = bins / TRANS_MAX * 2.0 - 1.0
        return xyz

    def encode_rxyz_uniform(self, values):
        """
        values: list 或 np.array, 形如 [rx, ry, rz]，取值范围在 [-π, π]
        返回: tokens, bins
        """
        ROT_MAX = len(self.EXTRA_ROT_XYZ_TOKENS) - 1
        values = np.array(values)
        bins = ((values + np.pi) / (2 * np.pi) * ROT_MAX).astype(int)
        bins = np.clip(bins, 0, ROT_MAX)
        tokens = [self.EXTRA_ROT_XYZ_TOKENS[b] for b in bins]
        return tokens, bins

    def decode_rxyz_uniform(self, tokens):
        """
        tokens: ["0003", "0512", ...] 形如字符串列表
        返回: np.array([rx, ry, rz])，范围在 [-π, π]，取token对应bin的中心（mid）
        """
        ROT_MAX = len(self.EXTRA_ROT_XYZ_TOKENS) - 1
        bins = np.array([int(token) for token in tokens])
        rxyz = bins / ROT_MAX * 2 * np.pi - np.pi
        return rxyz


def quantize_value(data, type="loc", mode="norm", w=None, h=None):
    """
    将 loc / rot / trans / size 等数值进行 量化 (norm) 或 反量化 (denorm)。

    参数:
        data: 输入数据 (Tensor)，shape 视 type 而定
        type: "loc" | "rot" | "trans" | "size"
        mode: "norm" 量化到桶号；"denorm" 反量化为归一化坐标 / 物理量
        w, h: 仅 type=="loc" 时使用，原始图像宽高
    """
    LOC_MAX = 1023
    TRANS_MAX = 2047
    ROT_MAX = 255
    SIZE_MAX = 1023
    if type == "loc":
        if mode == "norm":
            # data: (4,) [x1,y1,x2,y2] 原始坐标
            x1 = (data[0].float() / w * LOC_MAX).clamp(0, LOC_MAX).int()
            y1 = (data[1].float() / h * LOC_MAX).clamp(0, LOC_MAX).int()
            if data.shape[0] == 2:
                return x1, y1
            elif data.shape[0] == 4:
                x2 = (data[2].float() / w * LOC_MAX).clamp(0, LOC_MAX).int()
                y2 = (data[3].float() / h * LOC_MAX).clamp(0, LOC_MAX).int()
                return x1, y1, x2, y2
        elif mode == "denorm":
            # data: (x1, y1, x2, y2) 量化坐标
            x1 = data[0].float() / LOC_MAX
            y1 = data[1].float() / LOC_MAX
            if data.shape[0] == 2:
                return x1, y1
            elif data.shape[0] == 4:
                x2 = data[2].float() / LOC_MAX
                y2 = data[3].float() / LOC_MAX
                return x1, y1, x2, y2

    if type == "rot":
        if mode == "norm":
            # data: (3,) 欧拉角
            mapped = ((data + torch.pi) / (2 * torch.pi) * ROT_MAX).clamp(0, ROT_MAX).int()
            return mapped
        elif mode == "denorm":
            # data: (3,) 量化后的角度
            return data.float() / ROT_MAX * (2 * torch.pi) - torch.pi

    if type == "trans":
        if mode == "norm":
            # data: (3,) xyz位移
            return ((data + 1.) / 2. * TRANS_MAX).clamp(0, TRANS_MAX).int()
        elif mode == "denorm":
            return data.float() / TRANS_MAX * 2 - 1

    if type == "size":
        if mode == "norm":
            return (data * SIZE_MAX).clamp(0, SIZE_MAX).int()
        elif mode == "denorm":
            return data.float() / SIZE_MAX

    raise ValueError(f"Unsupported type or mode: type={type}, mode={mode}")


def decode_text_to_scene(text):
    def to_numpy(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    # 匹配每个 <imageX>...</imageX> 块
    image_blocks = re.findall(r'<(image\d+)>(.*?)</\1>', text)

    result = {}

    for img_name, content in image_blocks:
        # 每个物体是以 类别名+若干属性组成，例如 bowl<loc0675><loc0758><loc0799>...;
        # 匹配 类名，提取所有属性对
        object_pattern = r'([^;<]+)((?:<\w+\d+>)+);'  # 类名及后面的属性，如 bowl<loc0675><loc0758>...;
        objects = re.findall(object_pattern, content)

        objects_list = []
        for class_name, attr_str in objects:
            # 分别匹配属性
            # loc, 支持2或4
            loc_vals = [int(m) for m in re.findall(r'<loc(\d+)>', attr_str)]
            loc = quantize_value(torch.tensor(loc_vals), "loc", "denorm") if len(loc_vals) in [2, 4] else None

            # trans
            trans_vals = [int(m) for m in re.findall(r'<trans(\d+)>', attr_str)]
            trans = quantize_value(torch.tensor(trans_vals), "trans", "denorm") if len(trans_vals) == 3 else None

            # rot
            rot_vals = [int(m) for m in re.findall(r'<rot(\d+)>', attr_str)]
            rot = quantize_value(torch.tensor(rot_vals), "rot", "denorm") if len(rot_vals) == 3 else None

            # size
            size_vals = [int(m) for m in re.findall(r'<size(\d+)>', attr_str)]
            size = quantize_value(torch.tensor(size_vals), "size", "denorm") if len(size_vals) == 3 else None

            # 分别匹配属性
            # locs = [quantize_value(m, "loc", "denorm") for m in re.findall(r'<loc(\d+)>', attr_str)]
            # trans = [quantize_value(m, "trans", "denorm") for m in re.findall(r'<trans(\d+)>', attr_str)]
            # rot = [quantize_value(m, "rot", "denorm") for m in re.findall(r'<rot(\d+)>', attr_str)]
            # sizes = [quantize_value(m, "size", "denorm") for m in re.findall(r'<size(\d+)>', attr_str)]

            obj_dict = {
                'class': class_name.strip(), # string
                'loc': to_numpy(loc), # np.array (4,)
                'trans': to_numpy(trans), # np.array (3,)
                'rot': to_numpy(rot), # np.array (3,) euler xyz
                'size': to_numpy(size)# np.array (3,)
            }
            objects_list.append(obj_dict)

        result[img_name] = objects_list

    return result

def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.asin(sinp.clamp(-1, 1))

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    euler_xyz = torch.stack([roll, pitch, yaw])

    return euler_xyz

def encode_scene_to_text(class_name, multi_images, multi_2ds, multi_poses, multi_sizes):
    text_label = ''
    for view_i, (image, list_2d, list_pose, list_size) in enumerate(
            zip(multi_images, multi_2ds, multi_poses, multi_sizes)):
        _, h, w = image.shape
        text_label += f"<image{view_i}>"
        for obj_i, (loc, pose, size) in enumerate(zip(list_2d, list_pose, list_size)):
            def box_to_norm(loc, w, h):
                x1 = (loc[0].float() / w * 1023).clamp(0, 1023).int()
                y1 = (loc[1].float() / h * 1023).clamp(0, 1023).int()
                x2 = (loc[2].float() / w * 1023).clamp(0, 1023).int()
                y2 = (loc[3].float() / h * 1023).clamp(0, 1023).int()
                return x1, y1, x2, y2

            x1, y1, x2, y2 = box_to_norm(loc, w, h)

            def euler_to_norm(euler_xyz):
                mapped = (euler_xyz + torch.pi) / (2 * torch.pi) * 255
                mapped = torch.clamp(mapped, 0, 255).int()
                return mapped

            euler_xyz = quaternion_to_euler(pose[:4])  # shape: (3,)
            rx, ry, rz = euler_to_norm(euler_xyz)

            def translation_to_norm(trans):
                return ((trans + 1.) / 2. * 2047).clamp(0, 2047).int()

            tx, ty, tz = translation_to_norm(pose[4:7])

            def size_to_norm(size):
                return torch.clamp(size * 1023, 0, 1023).int()

            sx, sy, sz = size_to_norm(size)
            obj_2d_string = f"{class_name}<loc{x1:04d}><loc{y1:04d}><loc{x2:04d}><loc{y2:04d}>"
            obj_pose_string = f"<rot{rx:03d}><rot{ry:03d}><rot{rz:03d}><trans{tx:04d}><trans{ty:04d}><trans{tz:04d}>"
            obj_size_string = f"<size{sx:04d}><size{sy:04d}><size{sz:04d}>;"
            text_label += f"{obj_2d_string}{obj_pose_string}{obj_size_string}"
        text_label += f"</image{view_i}>"

    return text_label

def broadcast_class_names(multi_2ds, attributes):
    """将单个/列表形式的类别名广播为与 multi_2ds 嵌套结构匹配的二维列表。"""
    # 支持二维嵌套list直接返回
    if isinstance(attributes, list) and attributes and isinstance(attributes[0], list):
        return attributes
    repeat_names = []
    for objs_in_view in multi_2ds:
        names_in_view = []
        for _ in objs_in_view:  # 实例数不管，均取第一个类名
            if isinstance(attributes, list):
                name = attributes[0]
            else:
                name = attributes
            names_in_view.append(name)
        repeat_names.append(names_in_view)
    return repeat_names

def encode_scene_to_text_with_tokenizer(class_names, multi_images, multi_2ds, multi_poses, multi_sizes, tokenizer, near2far=True):
    '''
    将3D场景标注信息转换为字符串，适用于分割/检测/pose等任务。
    输入:
       class_names: str, list[str], list[list[str]]，物体类别，与multi_2ds匹配
       multi_images: list[Tensor]，每个视图图片(shape: 3×H×W)
       multi_2ds: list[list]，每个视图的物体2D信息，None表示无物体
       multi_poses: list[list[Tensor]]，每个视图的物体7D位姿 [q0, q1, q2, q3, tx, ty, tz]
       multi_sizes: list[list[Tensor]]，每个视图的物体尺寸
       tokenizer: 实例，包含encode方法（如encode_loc_uniform等）
    输出:
       text_label: str，对所有视图和物体编码后的字符串
    '''
    class_names = broadcast_class_names(multi_2ds, class_names)
    # import pdb;pdb.set_trace()

    text_label = ''
    for view_i, (list_name, image, list_2d, list_pose, list_size) in enumerate(
            zip(class_names, multi_images, multi_2ds, multi_poses, multi_sizes)):
        _, h, w = image.shape
        text_label += f"<image{view_i}>"

        # near to far 排序
        if near2far:
            objs = list(zip(list_name, list_2d, list_pose, list_size)) # pose: [qw, qx, qy, qz, x, y, z]
            objs_sorted = sorted(objs, key=lambda x: x[2][6])  # x[2] 是 pose，x[2][6] 是 z

            list_name = [x[0] for x in objs_sorted]
            list_2d = [x[1] for x in objs_sorted]
            list_pose = [x[2] for x in objs_sorted]
            list_size = [x[3] for x in objs_sorted]

        for obj_i, (name, loc, pose, size) in enumerate(zip(list_name, list_2d, list_pose, list_size)):
            if loc is None and pose is None and size is None:
                continue
            # grasp pose可能为0
            if torch.all(pose == 0):
                continue

            # loc
            obj_2d_string = ""
            if loc is not None:
                if tokenizer.USE_BOX_FORMAT:
                    obj_2d_string = tokenizer.encode_box(loc, w, h)
                else:
                    loc_tokens, _ = tokenizer.encode_loc_uniform(loc, w, h)
                    obj_2d_string = ''.join([f"{tok}" for tok in loc_tokens])

            # rot (quaternion->euler->token)
            obj_trans_string = ""
            if pose is not None:
                euler_xyz = quaternion_to_euler(pose[:4])  # shape: (3,)
                rot_tokens, _ = tokenizer.encode_rxyz_uniform(euler_xyz)
                obj_rot_string = ''.join([f"{tok}" for tok in rot_tokens])

                # trans分别处理 x, y 和 z
                tx_token, _ = tokenizer.encode_txy(pose[4])
                ty_token, _ = tokenizer.encode_txy(pose[5])
                tz_token, _ = tokenizer.encode_tz(pose[6])
                obj_trans_string = f"{tx_token}{ty_token}{tz_token}"

            # size
            obj_size_string = ""
            if size is not None:
                size_tokens, _ = tokenizer.encode_sxyz(size)
                obj_size_string = ''.join([f"{tok}" for tok in size_tokens])

            obj_str = f"{name}{obj_2d_string}{obj_trans_string}{obj_size_string}{obj_rot_string};"
            text_label += obj_str
        text_label += f"</image{view_i}>"

    return text_label

def encode_scene_to_text_with_tokenizer_ablation(class_names, multi_images, multi_2ds, multi_poses, multi_sizes, tokenizer, near2far=True):
    '''
    将3D场景标注信息转换为字符串，适用于分割/检测/pose等任务。
    输入:
       class_names: str, list[str], list[list[str]]，物体类别，与multi_2ds匹配
       multi_images: list[Tensor]，每个视图图片(shape: 3×H×W)
       multi_2ds: list[list]，每个视图的物体2D信息，None表示无物体
       multi_poses: list[list[Tensor]]，每个视图的物体7D位姿 [q0, q1, q2, q3, tx, ty, tz]
       multi_sizes: list[list[Tensor]]，每个视图的物体尺寸
       tokenizer: 实例，包含encode方法（如encode_loc_uniform等）
    输出:
       text_label: str，对所有视图和物体编码后的字符串
    '''
    class_names = broadcast_class_names(multi_2ds, class_names)
    # import pdb;pdb.set_trace()

    text_label = ''
    for view_i, (list_name, list_2d, list_pose, list_size) in enumerate(
            zip(class_names, multi_2ds, multi_poses, multi_sizes)):

        # near to far 排序
        if near2far:
            objs = list(zip(list_name, list_2d, list_pose, list_size)) # pose: [qw, qx, qy, qz, x, y, z]
            objs_sorted = sorted(objs, key=lambda x: x[2][6])  # x[2] 是 pose，x[2][6] 是 z

            list_name = [x[0] for x in objs_sorted]
            list_2d = [x[1] for x in objs_sorted]
            list_pose = [x[2] for x in objs_sorted]
            list_size = [x[3] for x in objs_sorted]

        for obj_i, (name, loc, pose, size) in enumerate(zip(list_name, list_2d, list_pose, list_size)):
            print(list_pose.shape)
            if loc is None and pose is None and size is None:
                continue
            # grasp pose可能为0
            if torch.all(pose == 0):
                continue

            # rot (quaternion->euler->token)
            obj_trans_string = ""
            if pose is not None:
                euler_xyz = quaternion_to_euler(pose[:4])  # shape: (3,)
                rot_tokens, _ = tokenizer.encode_rxyz_uniform(euler_xyz)
                obj_rot_string = ''.join([f"{tok}" for tok in rot_tokens])

                # trans分别处理 x, y 和 z
                tx_token, _ = tokenizer.encode_txy(pose[4])
                ty_token, _ = tokenizer.encode_txy(pose[5])
                tz_token, _ = tokenizer.encode_tz(pose[6])
                obj_trans_string = f"{tx_token}{ty_token}{tz_token}"

            obj_str = f"{name}{obj_trans_string}{obj_rot_string};"
            text_label += obj_str

    return text_label

def decode_text_to_scene_with_tokenizer(text, tokenizer):
    """
    :param text: 原始文本
    :param tokenizer: 你的Tokenizer类实例，提供decode_txy等方法
    :return: 结构化解析结果
    """
    try:
        # 匹配每个 <imageX>...</imageX> 块
        image_blocks = regex.findall(r'<(image\d+)>(.*?)</\1>', text, timeout=1000)
    except regex.TimeoutError:
        print(text)

    result = {}

    for img_name, content in image_blocks:
        try:
            # 匹配 类名 + 属性串（兼容 <box>...</box> 和 <loc>... 两种格式）
            object_pattern = r'([^;<]+?)(<box>\([^)]+\)(?:,\([^)]+\))*</box>(?:<\w+\d+>)*|(?:<\w+\d+>)+);'
            objects = regex.findall(object_pattern, content, timeout=1000)
        except regex.TimeoutError:
            print(text)
            print(content)

        objects_list = []
        for class_name, attr_str in objects:
            class_name = class_name.strip()  # 去掉前导空格
            # loc: 优先检测 <box> 格式，否则回退到 <loc> 格式
            loc = None
            box_match = re.search(r'<box>(.*?)</box>', attr_str)
            if box_match:
                loc = tokenizer.decode_box(box_match.group(0))
            else:
                pattern = r'<loc(\d+)>'
                loc_tokens = re.findall(pattern, attr_str)     # 取出来的都是str
                if len(loc_tokens) in [2, 4]:
                    loc = tokenizer.decode_loc_uniform(loc_tokens)

            # trans: txy和tz
            # pattern = r'<trans(\d+)>'
            pattern = r'<trans[A-Za-z]+\d+>'
            trans_tokens = re.findall(pattern, attr_str)
            trans = None
            # if len(trans_tokens) == 3:
            if (len(trans_tokens) == 3 and
                    trans_tokens[0].startswith('<transXY') and
                    trans_tokens[1].startswith('<transXY') and
                    trans_tokens[2].startswith('<transZ')
            ):
                # # 假定前两个用于txy，第三个用于tz
                txy_tokens = trans_tokens[:2]
                tz_token = [trans_tokens[2]] # 这样保证token数量和decode函数兼容
                txy_val, _ = tokenizer.decode_txy(txy_tokens)
                tz_val, _ = tokenizer.decode_tz(tz_token)
                # 拼接转换为np数组
                trans = np.array([*txy_val, tz_val])

            # rot:
            pattern = r'<rot(\d+)>'
            rot_tokens = re.findall(pattern, attr_str)
            rot = None
            if len(rot_tokens) == 3:
                # rot直接用decode
                rot_val = tokenizer.decode_rxyz_uniform(rot_tokens)
                rot = np.array(rot_val)


            # size
            # pattern = r'<size(\d+)>'
            pattern = r'<size\d+>'
            size_tokens = re.findall(pattern, attr_str)
            size = None
            if len(size_tokens) == 3:
                size_val, _ = tokenizer.decode_sxyz(size_tokens)
                size = np.array(size_val)

            obj_dict = {
                'class': class_name.strip(),
                'loc': loc,
                'trans': trans,
                'rot': rot,
                'size': size
            }
            objects_list.append(obj_dict)

        result[img_name] = objects_list

    return result


# ============================================================
# 兼容性别名 (Backward-compatible aliases)
# 旧函数名仍可用，建议在新代码中使用上方的新名字。
# ============================================================
norm_process = quantize_value
text_to_class_attr_dict = decode_text_to_scene
map_3d_label_to_string = encode_scene_to_text
repeat_attribute_to_match = broadcast_class_names
map_3d_label_to_string_tokenizer = encode_scene_to_text_with_tokenizer
map_3d_label_to_string_tokenizer_ablation = encode_scene_to_text_with_tokenizer_ablation
text_to_class_attr_dict_tokenizer = decode_text_to_scene_with_tokenizer