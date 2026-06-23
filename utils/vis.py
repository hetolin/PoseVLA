import copy
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
import seaborn as sns
import numpy as np
import math
import re
import cv2
import torch
from io import BytesIO
from PIL import Image
from scipy.spatial.transform import Rotation
# pip install adjustText
from adjustText import adjust_text


def visualize_attention_mask(attention_mask, batch_index=0, head_index=0):
    """
    可视化 attention mask（debug 用，例如 modeling_pi0 里 att_2d_masks）。

    Args:
        attention_mask: (bs, q_len, k_len) 或 (bs, heads, q_len, k_len)
        batch_index: 批次索引
        head_index: 多头时的头索引（仅 4D 输入生效）
    """
    if len(attention_mask.shape) == 3:
        mask_to_visualize = attention_mask[batch_index]              # (q_len, k_len)
    elif len(attention_mask.shape) == 4:
        mask_to_visualize = attention_mask[batch_index, head_index]  # (q_len, k_len)
    else:
        raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

    plt.figure(figsize=(32, 32))
    sns.heatmap(
        mask_to_visualize.float().cpu().detach().numpy(),
        cmap='viridis', cbar=True, linecolor='black', square=False,
    )
    plt.title(f'Attention Mask - Batch {batch_index}, Head {head_index}')
    plt.xlabel('Key/Value Length')
    plt.ylabel('Query Length')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.savefig('attention_mask.png')
    plt.close()


# def vis_atten_map(att_output, images, wandb=False):
#     # images [[b,c,h,w],[b,c,h,w],[b,c,h,w]]
#     # att_output [[b, num_head, q_len, k_len]]
#     ########################### check attention #############################
#
#     images_vis = images[:]
#     for i, img in enumerate(images_vis):
#         # (-1,1) -> (0, 1)
#         images_vis[i] = (images_vis[i]*0.5)+0.5
#         # chw 2 hwc
#         images_vis[i] = images_vis[i].permute(0,2,3,1).float().detach().cpu().numpy()
#
#     def overlay(rgb, heatmap):
#         heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_NEAREST)
#         heatmap = plt.cm.jet(heatmap)[:, :, :3]  # 只取 RGB 通道
#         heatmap = (heatmap * 255).astype(np.uint8)  # 转换为 uint8 类型
#         rgb = cv2.resize(rgb, (224, 224))
#         if np.max(rgb)<2:
#             rgb *= 255.
#         rgb = rgb.astype(np.uint8)
#
#         overlay = cv2.addWeighted(rgb, 0.5, heatmap, 0.5, 0)
#         return overlay
#
#     bs = 0
#     num_views = len(images_vis)
#     num_layers = len(att_output)
#     num_heads = att_output[bs].shape[1]
#     patch_size = 16
#     num_patches = patch_size * patch_size
#     for layer in range(1, num_layers):
#         # map = output_action["attentions"][layer].view(1, 1, 51 * 8, -1)
#         # plt.imshow(map[0, 0, :, :].float().cpu().numpy(), cmap='viridis')
#         # for i in range(8):
#         #     plt.subplot(8, 1, i + 1)
#         #     map = output_action["attentions"][layer][0, i, :, :].float().cpu().numpy()
#         #     plt.imshow(map, cmap='viridis')
#         # plt.show()
#
#         fig, axes = plt.subplots(num_views, num_heads + 1, figsize=(4 * (num_heads + 1), 4 * 3))
#         for view in range(num_views):
#             for i in range(num_heads + 1):
#                 ax = axes[view, i]
#                 if i == num_heads:
#                     map = images_vis[view][bs]
#                 else:
#                     map = att_output[layer][bs, i, 1, :num_patches].reshape(patch_size,
#                                                                             patch_size).float().detach().cpu().numpy()
#                     map = (map - np.min(map)) / (np.max(map) - np.min(map))
#                     map = overlay(images_vis[view][bs], map)
#                 ax.imshow(map)
#                 ax.set_title(f'View{view + 1} Head Attention{i + 1}' if i < num_heads else f'View{view + 1} ')
#                 ax.axis('off')
#
#         plt.show()
#         if wandb:
#             return fig
#         # 保存图片，建议文件命名带上layer序号
#         plt.tight_layout()
#         save_dir = "./attention_maps"
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(f'{save_dir}/attention_layer_{layer}.png', dpi=200)  # 可改文件夹，如'./att_plot/attention_layer_{layer+19}.png'
#         plt.close()  # 不需要show，否则弹出窗口；close可以释放内存


def vis_atten_map(att_output, images, wandb=False):
    def overlay(rgb, heatmap):
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_NEAREST)
        heatmap = plt.cm.jet(heatmap)[:, :, :3]  # 只取 RGB 通道
        heatmap = (heatmap * 255).astype(np.uint8)  # 转换为 uint8 类型
        rgb = cv2.resize(rgb, (224, 224))
        if np.max(rgb)<2:
            rgb = rgb.astype(np.float32) * 255.
        rgb = rgb.astype(np.uint8)

        overlay = cv2.addWeighted(rgb, 0.5, heatmap, 0.5, 0)
        return overlay

    def concat_layers_imgs_auto(layer_figs, n_cols=6):  # 默认每行6个
        layer_num = len(layer_figs)
        n_rows = math.ceil(layer_num / n_cols)

        # 假设所有图片大小一致
        img_width, img_height = layer_figs[0].size

        composite = Image.new('RGB', (n_cols * img_width, n_rows * img_height),(255, 255, 255))

        for idx, im in enumerate(layer_figs):
            row = idx // n_cols
            col = idx % n_cols
            x = col * img_width
            y = row * img_height
            composite.paste(im, (x, y))

        return composite

    # images: list of (bs*history_len, c, h, w)，list 长度 = 相机数
    # att_output [[b, num_head, q_len, k_len]]，k_len = num_patches * 相机数（不含 history）
    # num_views = 相机数，不是 相机数 × history_len
    num_views = len(images)
    # 每个相机只取第一帧用于可视化（最新帧）
    images_vis = []
    for img in images:
        if isinstance(img, torch.Tensor) and img.dim() == 4:
            images_vis.append(convert_to_opencv_format(img[0]))  # 取第一帧 (c, h, w)
        else:
            images_vis.append(convert_to_opencv_format(img))

    # calculate mean attention
    for i, att in enumerate(att_output):
        att_output[i] = torch.mean(att, dim=1, keepdim=True)

    bs = 0
    num_layers = len(att_output)
    num_heads = att_output[bs].shape[1]
    # 动态推算 patch_size：att_output 的 key 维度已经是纯 image patch（连续排列）
    total_img_patches = att_output[0].shape[-1]
    num_patches = total_img_patches // num_views
    patch_size = int(num_patches ** 0.5)

    layer_figs = []
    for layer in range(num_layers):
        fig, axes = plt.subplots(num_views, num_heads + 1, figsize=(2 * (num_heads + 1), 2 * num_views))
        for view in range(num_views):
            for i in range(num_heads + 1):

                ax = axes[view, i]
                if i == num_heads:
                    map_img = images_vis[view]
                else:
                    start = num_patches * view
                    end = num_patches * (view + 1)
                    map = att_output[layer][bs, i, 1, start:end].reshape(
                        patch_size, patch_size).float().detach().cpu().numpy()
                    map = (map - np.min(map)) / (np.max(map) - np.min(map) + 1e-6)
                    map_img = overlay(images_vis[view], map)
                ax.imshow(map_img)
                ax.set_title(f'Layer {layer + 1} Head {i + 1}' if i < num_heads else f'RGB{view + 1}')
                ax.axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_pil = Image.open(buf)
        layer_figs.append(img_pil)
        plt.close(fig)  # 释放内存

    composite = concat_layers_imgs_auto(layer_figs, n_cols=6)

    if wandb:
        return composite

    save_dir = "./attention_maps"
    os.makedirs(save_dir, exist_ok=True)
    composite.save(f"{save_dir}/attn_map.png")   # 最简单
    plt.close()

def plot_all_joints(action, obs, save_path='all_joints.png', wandb=False):
    N_t, joint_dim = action.shape
    # 自动决定子图的行列数（比如每行4个）
    ncols = min(4, joint_dim)
    nrows = math.ceil(joint_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False, sharex=True, sharey=True)

    for j in range(joint_dim):
        row, col = divmod(j, ncols)
        ax = axes[row, col]
        ax.plot(action[:, j], label='GT', linewidth=2, marker='o')
        ax.plot(obs[:, j], label='Observation', linestyle='--', linewidth=2, marker='o')
        ax.set_title(f'Joint {j}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')
        ax.legend()

    # 去掉多余的空子图
    for j in range(joint_dim, nrows * ncols):
        row, col = divmod(j, ncols)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)

    if wandb:
        return img_pil

    img_pil.save(save_path)
    print(f"Saved all joint trajectories to {save_path}")

# 用法示例
# gt = np.random.randn(100, 6)
# obs = np.random.randn(100, 6)
# plot_all_joints(gt, obs, save_path='all_joints.png')

def convert_to_opencv_format(image):
    """
    将图像转换为适合OpenCV处理的标准格式

    参数:
        image: 输入图像，可以是numpy数组或PyTorch张量

    返回:
        numpy数组格式的图像，具有以下特性:
        - HWC (高度,宽度,通道) 格式
        - uint8数据类型，像素值范围0-255
        - 连续的内存布局
        - 3通道BGR格式
    """
    # 确保图像是 numpy 数组
    if isinstance(image, torch.Tensor):
        # 如果是 PyTorch 张量，转换为 numpy 数组
        image_np = image.detach().clone()  # 创建副本以避免修改原始张量

        if image_np.dim() == 4:  # BCHW 格式
            image_np = image_np.squeeze(0)  # 移除批次维度

        # 将 CHW 转换为 HWC (适用于 OpenCV)
        if image_np.dim() == 3 and image_np.shape[0] in [1, 3, 4]:
            image_np = image_np.permute(1, 2, 0)

        # 转换为 numpy 数组
        image_np = image_np.float().cpu().numpy()

        # 自适应值范围：如果存在负值，说明是 [-1,1]，先转为 [0,1]
        if image_np.min() < 0:
            image_np = (image_np + 1.0) / 2.0

        # 标准化到 0-255 范围
        image_np = np.clip(image_np, 0, 1)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    else:
        image_np = image

    # 确保数组是连续的，提高OpenCV处理效率
    image_np = np.ascontiguousarray(image_np)

    # 确保图像是 3 通道
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 1:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    return image_np

import random
def get_history_indices(
    step_id: int,
    history_size: int,
    interval: int,
    random_sample: bool = True,
    stride_jitter: float = 0.15,
    phase_jitter: bool = True,
):
    """
    获取历史帧的索引列表（全局 stride + 全局 phase 策略）。

    设计原则：
    - 同一个 sample 内的所有历史锚点共享同一个 stride 和同一个 phase，
      保证相邻帧之间的视觉差分幅度在 sample 内部是稳定的。
    - 训练时允许对 stride 做温和抖动 (stride_jitter)，让模型对推理时帧率
      的轻微变化更鲁棒；不要抖动过大，以免人为制造 OOD。
    - 训练时允许对 phase 做抖动（整体左移 0..stride-1 帧），等价于在
      episode 内"随机选一个起始时刻"，而不是"每帧各自乱抽"。
    - 边界处（episode 开头，历史帧不够）用 clamp 到 [0, step_id]，
      语义上相当于用最早那一帧做 left-padding，不会产生多个重复 0 帧
      之外的副作用。

    Args:
        step_id (int): 当前帧索引（最新一帧）。
        history_size (int): 包含 current 在内的历史帧总数（返回会去掉 current）。
        interval (int): 基准采样步长。
        random_sample (bool): 训练时 True，推理时 False（严格等间隔、zero phase）。
        stride_jitter (float): stride 的相对抖动幅度 (e.g. 0.15 表示 ±15%)。
                               仅在 random_sample=True 时生效。
        phase_jitter (bool): 是否对整段锚点整体随机左移；仅在 random_sample=True 时生效。

    Returns:
        List[int]: 历史帧索引列表（不含 current），长度 = history_size - 1。
    """
    # -------- 1. 确定本次采样使用的全局 stride --------
    if random_sample and stride_jitter > 0:
        low = max(1, int(round(interval * (1.0 - stride_jitter))))
        high = max(low, int(round(interval * (1.0 + stride_jitter))))
        stride = random.randint(low, high)
    else:
        stride = max(1, int(interval))

    # -------- 2. 确定本次采样使用的全局 phase --------
    # phase 表示"整段锚点整体向过去平移多少帧"。
    # 训练时允许 0..stride-1 的随机平移，让模型见到不同的时间对齐。
    # 这样能保留"时间概念"（stride 稳定、相邻帧差分稳定），
    # 同时又不会让每帧独立 randint 制造的差分噪声。
    if random_sample and phase_jitter and stride > 1:
        phase = random.randint(0, stride - 1)
    else:
        phase = 0

    # -------- 3. 按照统一 stride / phase 生成锚点 --------
    # 最新锚点 = step_id - phase；
    # 其他锚点依次向过去按 stride 步长后退。
    # 边界处用 clamp 到 [0, step_id]：相当于用 episode 最早帧做 left-padding。
    indices = []
    for i in range(history_size):
        offset = (history_size - 1 - i) * stride + phase
        idx = step_id - offset
        idx = max(0, min(idx, step_id))  # clamp，防越界
        indices.append(idx)

    # 最后一帧强制对齐到 current timestep（即使 phase>0 也必须以 current 结尾）
    indices[-1] = step_id

    # 去除最后一帧 current timestep，只返回历史部分
    return indices[:-1]

def build_K_from_intrinsics(intrinsics):
    # 转成numpy数组
    if isinstance(intrinsics, torch.Tensor):
        arr = intrinsics.float().detach().cpu().numpy()
    elif isinstance(intrinsics, np.ndarray):
        arr = intrinsics
    elif isinstance(intrinsics, (list, tuple)):
        arr = np.array(intrinsics)
    else:
        raise TypeError(f"Unsupported type: {type(intrinsics)}")

    # 如果本身就是3x3矩阵
    if arr.shape == (3, 3):
        K = arr.astype(np.float32)
    else:
        if arr.shape[0] < 4:
            raise ValueError("Intrinsics must contain at least 4 elements (fx, fy, cx, cy)")
        fx, fy, cx, cy = arr[:4]
        K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape(3, 3)
    return K

def euler_to_matrix(rot, degrees=False):
    return Rotation.from_euler('xyz', rot, degrees=degrees).as_matrix()

def project_point(trans, K):
    pt3d = np.array(trans, dtype=np.float32).reshape(3, 1)
    x = K @ pt3d
    x /= x[2]
    return float(x[0][0]), float(x[1][0])

# def draw_bbox_matplot(ax, loc, img_hw, color='#1976D2', label=None):
#     h, w = img_hw
#     if len(loc) == 2:  # 点
#         x = loc[0] * w
#         y = loc[1] * h
#         ax.scatter(x, y, s=70, c=color, marker='o', edgecolors='white', linewidths=1.8, zorder=10)
#         if label:
#             ax.text(x+8, y-8, label, fontsize=12, color='white',
#                     ha='left', va='bottom',
#                     bbox=dict(boxstyle='round,pad=0.19', fc=color, ec='none', alpha=0.85), zorder=12)
#     elif len(loc) == 4:  # 框
#         xmin, ymin, xmax, ymax = loc
#         xmin, xmax = xmin*w, xmax*w
#         ymin, ymax = ymin*h, ymax*h
#         rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2.3,
#                              edgecolor=color, facecolor='none', zorder=9, alpha=0.72)
#         ax.add_patch(rect)
#         if label:
#             ax.text(xmin+8, ymin+11, label, fontsize=12, color='white', ha='left', va='top',
#                     bbox=dict(boxstyle='round,pad=0.18', fc=color, ec='none', alpha=0.7), zorder=11)
#     # else: pass

def draw_bbox_matplot(ax, locs_labels, img_hw, color='#1976D2'):
    h, w = img_hw
    text_list = []
    for i, (loc, label) in enumerate(locs_labels):

        # ---- 点: loc 长度2 ----
        if len(loc) == 2:
            point_color = '#7B1FA2'
            x = loc[0] * w
            y = loc[1] * h
            ax.scatter(x, y, s=40, c=point_color, marker='o', edgecolors='white', linewidths=1.8, zorder=10)
            if label:
                if label and ("trajectory" not in label or i in (0, len(locs_labels)-1)):
                    t = ax.text(x, y, label, fontsize=12, color='white',
                                ha='left', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', fc=point_color, ec='none', alpha=0.85), zorder=12)
                    text_list.append(t)
        # ---- 框: loc 长度4 ----
        elif len(loc) == 4:
            xmin, ymin, xmax, ymax = loc
            xmin, xmax = xmin*w, xmax*w
            ymin, ymax = ymin*h, ymax*h
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2.3,
                                 edgecolor=color, facecolor='none', zorder=9, alpha=0.72)
            ax.add_patch(rect)
            if label:
                label_offset = 6  # 缝隙可调整
                ax.text(xmin, ymin - label_offset, label, fontsize=12, color='white', ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.18', fc=color, ec='none', alpha=0.7), zorder=11)
        # ---- 其他情况: 按点画 ----
        else:
            point_color = '#7B1FA2'
            # 假设loc里面有两个元素（否则你需设计容错）
            # 如果长度不是2，需要选前两个分量作为点坐标
            # 你可以用更多的自定义策略，比如取前两个分量
            try:
                x = loc[0] * w
                y = loc[1] * h
            except Exception as e:
                continue  # 跳过错误数据
            ax.scatter(x, y, s=70, c=point_color, marker='o', edgecolors='white', linewidths=1.8, zorder=10)

            if label:
                t = ax.text(x, y, label, fontsize=12, color='white', ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', fc=point_color, ec='none', alpha=0.85), zorder=12)
                text_list.append(t)
    # 只有点的label自动调整
    if text_list:
        adjust_text(
            text_list,
            ax=ax,
            force_arrow=True,
            arrowprops=dict(arrowstyle='-', color='#7B1FA2')
        )

def draw_axes_matplot(ax, trans, rot, K, length=0.02, colors=['#FF5252','#76FF03','#2979FF'], alpha=0.92):
    center_2d = project_point(trans, K)
    R = euler_to_matrix(rot)
    axes_dirs = np.eye(3)*length
    for i in range(3):
        end_3d = np.array(trans) + R @ axes_dirs[i]
        end_2d = project_point(end_3d, K)
        ax.plot([center_2d[0], end_2d[0]], [center_2d[1], end_2d[1]], colors[i], lw=3.0, alpha=alpha, zorder=19)
    ax.scatter([center_2d[0]], [center_2d[1]], c='w', s=21, lw=0.7, alpha=alpha, zorder=21)

def draw_3d_bbox_matplot(ax, trans, rot, size, K, color='#8E24AA', linewidth=2, fill=True, fill_alpha=0.5, ground_dir=np.array([0,-1,0])):
    # 获取立方体的长宽高
    l, w, h = size

    # 构造8个立方体角点（局部坐标系原点在立方体中心）
    corners_local = np.array([[l / 2, w / 2, h / 2],
                              [l / 2, -w / 2, h / 2],
                              [-l / 2, -w / 2, h / 2],
                              [-l / 2, w / 2, h / 2],
                              [l / 2, w / 2, -h / 2],
                              [l / 2, -w / 2, -h / 2],
                              [-l / 2, -w / 2, -h / 2],
                              [-l / 2, w / 2, -h / 2]])

    # 旋转+平移，把角点从局部坐标系变换到世界坐标系
    R = euler_to_matrix(rot)  # 欧拉角转旋转矩阵
    world_corners = np.dot(corners_local, R.T) + trans

    # 将世界坐标系下的八个角点投影到像素坐标
    pts_2d = np.array([project_point(pt, K) for pt in world_corners])

    # 立方体的6个面，每面4个点
    cube_faces = [
        [0,1,2,3],  # 顶面(+z)
        [4,5,6,7],  # 底面(-z)
        [0,4,5,1],  # 侧面(+x)
        [1,5,6,2],  # 侧面(-y)
        [2,6,7,3],  # 侧面(-x)
        [3,7,4,0],  # 侧面(+y)
    ]

    # 找和地面方向ground_dir最接近的那个面
    max_dot = -np.inf
    best_face_idx = None
    ground_dir = np.asarray(ground_dir, dtype=np.float32)
    ground_dir = ground_dir / np.linalg.norm(ground_dir)
    for fi, face in enumerate(cube_faces):
        # 计算面法线
        p0, p1, p2 = [world_corners[i] for i in face[:3]]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normal = normal / (np.linalg.norm(normal) + 1e-7)
        dot = normal @ ground_dir  # 法线与地面的方向内积
        if dot > max_dot:  # 保留最接近地面的（点积最大）
            max_dot = dot
            best_face_idx = fi

    # 拿到要填充的面的4个角点的2D投影坐标
    base_idx = cube_faces[best_face_idx]
    base_pts2d = pts_2d[base_idx]

    # 填充底面
    if fill:
        # 用Polygon加入面到ax
        poly = Polygon(base_pts2d, closed=True, facecolor=color, edgecolor=None, alpha=fill_alpha, zorder=8)
        ax.add_patch(poly)

    # 画立方体的所有边
    edges = [
        [0,1],[1,2],[2,3],[3,0],   # 顶面边
        [4,5],[5,6],[6,7],[7,4],   # 底面边
        [0,4],[1,5],[2,6],[3,7],   # 竖直边
    ]
    for a,b in edges:
        ax.plot([pts_2d[a,0], pts_2d[b,0]], [pts_2d[a,1], pts_2d[b,1]],
                color, lw=linewidth, alpha=0.9, zorder=9)

    # 画八个角点作为散点（方便观察结构）
    ax.scatter(pts_2d[:,0], pts_2d[:,1],
               c=color, s=20, lw=0.8, edgecolors='white', zorder=11, alpha=0.8)


def draw_gripper_2d_with_arm(ax, trans, rot, size, K,
                             arm_len=0.08, arm_thick=0.019,
                             color='dodgerblue', arm_color='dimgray',
                             linewidth=2, fill=True, fill_alpha=0.5):
    """
    在matplotlib ax上画一个抓手（U形指+臂），并投影到2D。
    ax      : matplotlib 2D轴对象
    trans   : (3,) np.array, gripper在世界坐标中的xyz位置
    rot     : (3,)欧拉角(xyz) 或 (3,3)旋转矩阵
    size    : [width_x, thickness_y, finger_length_z]，三个抓手参数
    K       : (3,3) 投影矩阵（内参矩阵）
    arm_len : float, 臂长度（世界单位m）
    arm_thick: float, 臂的粗细（方形截面边长，世界单位m）
    color   : gripper颜色
    arm_color: 臂颜色
    linewidth: 线宽
    fill, fill_alpha: 是否填充抓手/臂面及透明度
    返回所有点投影后的(x, y)像素坐标（np.array，shape=(总点数, 2)）
    坐标系约定：gripper坐标，front=z, right=x, down=y
    """
    width, thick, finger_len = size

    # ========== 1. 构造本地坐标系下的所有关键点 ==========
    # ---- 基座 ----
    base = np.array([
        [ width/2,  thick/2, 0],   # 右上
        [ width/2, -thick/2, 0],   # 右下
        [-width/2, -thick/2, 0],   # 左下
        [-width/2,  thick/2, 0],   # 左上
    ])

    # ---- 左抓指 ----
    left_finger = np.array([
        [-width/2,  thick/2, 0],                    # 左指根 上
        [-width/2,  thick/2, finger_len],           # 左指尖 上
        [-width/2, -thick/2, finger_len],           # 左指尖 下
        [-width/2, -thick/2, 0],                    # 左指根 下
    ])
    # ---- 右抓指 ----
    right_finger = np.array([
        [ width/2,  thick/2, 0],                    # 右指根 上
        [ width/2,  thick/2, finger_len],           # 右指尖 上
        [ width/2, -thick/2, finger_len],           # 右指尖 下
        [ width/2, -thick/2, 0],                    # 右指根 下
    ])

    # ---- 臂（方形杆，底面在gripper base中心，顶面在 -z 方向 arm_len 处） ----
    arm = np.array([
        [ arm_thick/2,  arm_thick/2,     0],              # 底面右上
        [ arm_thick/2, -arm_thick/2,     0],              # 底面右下
        [-arm_thick/2, -arm_thick/2,     0],              # 底面左下
        [-arm_thick/2,  arm_thick/2,     0],              # 底面左上

        [ arm_thick/2,  arm_thick/2, -arm_len],           # 顶面右上
        [ arm_thick/2, -arm_thick/2, -arm_len],           # 顶面右下
        [-arm_thick/2, -arm_thick/2, -arm_len],           # 顶面左下
        [-arm_thick/2,  arm_thick/2, -arm_len],           # 顶面左上
    ])

    # 所有点拼起来
    all_points = np.vstack([base, left_finger, right_finger, arm])   # shape=(20,3)

    # ========== 2. 旋转+平移 → 世界坐标 ==========
    if rot.shape == (3,):
        # 使用欧拉角，xyz方向（rad），你可以改用你的euler_to_matrix
        from scipy.spatial.transform import Rotation as R
        Rmat = R.from_euler('xyz', rot, degrees=False).as_matrix()
    else:
        Rmat = rot
    world_points = np.dot(all_points, Rmat.T) + trans    # shape=(20,3)

    # ========== 3. 投影到2D像素坐标 ==========
    def project_point(pt, K):
        pt_cam = pt   # 假设世界-相机不变换
        uvw = K @ pt_cam
        return uvw[:2] / uvw[2]
    pts_2d = np.array([project_point(p, K) for p in world_points])

    # ========== 4. 可视化抓手与臂 ==========
    # ---- (a) 基座 ----
    base2d = pts_2d[0:4]
    poly_base = Polygon(base2d, closed=True,
                        facecolor=color if fill else 'none',
                        edgecolor=color, alpha=fill_alpha, lw=linewidth, zorder=5)
    ax.add_patch(poly_base)
    # ---- (b) 左右抓指 ----
    left2d = pts_2d[4:8]
    right2d = pts_2d[8:12]
    poly_left = Polygon(left2d, closed=True,
                        facecolor=color if fill else 'none',
                        edgecolor=color, alpha=fill_alpha, lw=linewidth, zorder=6)
    poly_right = Polygon(right2d, closed=True,
                         facecolor=color if fill else 'none',
                         edgecolor=color, alpha=fill_alpha, lw=linewidth, zorder=6)
    ax.add_patch(poly_left)
    ax.add_patch(poly_right)

    # ---- (c) 臂（两面、边线） ----
    arm2d = pts_2d[12:20]
    # 顶面（负z方向端点）
    poly_arm_top = Polygon(arm2d[4:8], closed=True,
                           facecolor=arm_color if fill else 'none',
                           edgecolor=arm_color, alpha=fill_alpha*0.7, lw=linewidth, zorder=3)
    # 底面（连接gripper base）
    poly_arm_base = Polygon(arm2d[0:4], closed=True,
                            facecolor=arm_color if fill else 'none',
                            edgecolor=arm_color, alpha=fill_alpha*0.6, lw=linewidth, zorder=3)
    ax.add_patch(poly_arm_top)
    ax.add_patch(poly_arm_base)
    # 臂的边线
    for i in range(4):
        j = (i+1)%4
        # 底面
        ax.plot([arm2d[i,0], arm2d[j,0]], [arm2d[i,1], arm2d[j,1]],
                 arm_color, lw=linewidth)
        # 顶面
        ax.plot([arm2d[i+4,0], arm2d[(j+4)%8,0]], [arm2d[i+4,1], arm2d[(j+4)%8,1]],
                 arm_color, lw=linewidth)
        # 四条侧边
        ax.plot([arm2d[i,0], arm2d[i+4,0]], [arm2d[i,1], arm2d[i+4,1]],
                 arm_color, lw=linewidth)
    # ---- 可视化所有关键点 ----
    ax.scatter(pts_2d[:,0], pts_2d[:,1], c=color, s=18, edgecolor='white', lw=0.6, zorder=9)

    # ---- 抓手基座和爪边线 ----
    # base边
    for i in range(4):
        j = (i+1)%4
        ax.plot([base2d[i,0], base2d[j,0]],
                [base2d[i,1], base2d[j,1]], color, lw=linewidth)
    # left/right finger
    for finger2d in [left2d, right2d]:
        for i in range(4):
            j = (i+1)%4
            ax.plot([finger2d[i,0], finger2d[j,0]],
                    [finger2d[i,1], finger2d[j,1]], color, lw=linewidth)
    # 连gripper base到两个指根
    for bi, fi in zip([0,3], [0,3]):   # base[0], left[0]; base[3], left[3]
        ax.plot([base2d[bi,0], left2d[fi,0]], [base2d[bi,1], left2d[fi,1]], color, lw=linewidth)
    for bi, fi in zip([1,2], [0,3]):   # base[1], right[0]; base[2], right[3]
        ax.plot([base2d[bi,0], right2d[fi,0]], [base2d[bi,1], right2d[fi,1]], color, lw=linewidth)

    return pts_2d


def visualize_2d_3d_all(images, gt_res, pred_res, intrinsics=None, instruction='', save_path='output_point.png', vis_2d=True, vis_3d=True, wandb=False):
    '''
    images: dict, {img_name: img (np.uint8, RGB, shape=(H,W,3)/(torch.Tensor, RGB, shape=3,H,W))}
    gt_res/pred_res: dict, {img_name: [obj, ...]}, 每个obj: loc/trans/rot/size/class
    intrinsics: dict, {img_name: intr (matrix or [fx,fy,cx,cy])}
    '''
    img_names = list(images.keys())
    N = len(img_names)
    fig, axs = plt.subplots(2, N, figsize=(4 * N, 8))
    for row, mode in enumerate(['gt','pred']):
        obj_dict = gt_res if mode=='gt' else pred_res
        for col, img_name in enumerate(img_names):
            img = convert_to_opencv_format(images[img_name])
            if img.shape[-1]==3 and img.dtype==np.uint8:
                img_rgb = img.copy()
            else:
                img_rgb = img
            h, w = img_rgb.shape[:2]

            if intrinsics[img_name] is not None:
                K = build_K_from_intrinsics(intrinsics[img_name])

            ax = axs[row, col] if N > 1 else axs[row]
            ax.imshow(img_rgb)
            objs = obj_dict.get(img_name, [])
            # -------叠加画所有特征
            # for obj in objs:
            #     if obj is None or 'loc' not in obj: continue
            #     label = obj.get('class', None)
            #     draw_bbox_matplot(ax, obj['loc'], (h,w), color='#1976D2', label=label)
            loc_label_list = [(obj['loc'], obj.get('class', None)) for obj in objs
                              if obj is not None and obj['loc'] is not None]
            if vis_2d:
                draw_bbox_matplot(ax, loc_label_list, (h, w), color='#1976D2')

            if vis_3d:
                for idx, obj in enumerate(objs):
                    if obj is None: continue
                    if obj['trans'] is not None and obj['rot'] is not None:
                        # if len(objs) > 1:
                        #     alpha = 0.25 + 0.75 * (1 - idx / (len(objs) - 1))
                        # else:
                        #     alpha = 1.0
                        alpha = 1.0
                        #TODO: length=obj["size"].max() * 0.5
                        draw_axes_matplot(ax, obj['trans'], obj['rot'], K, length=0.1, alpha=alpha)
                for obj in objs:
                    if obj is None: continue
                    if obj['trans'] is not None and obj['rot'] is not None and obj['size'] is not None:
                        if "grasp" in instruction:
                            draw_gripper_2d_with_arm(ax, obj['trans'], obj['rot'], obj['size'], K)
                        else:
                            draw_3d_bbox_matplot(ax, obj['trans'], obj['rot'], obj['size'], K)
            ax.set_title(f"{img_name} - {mode.upper()}", fontsize=14, backgroundcolor='#F3E5F5' if mode=='gt' else '#E1F5FE', color='#222', pad=10)
            ax.axis('off')
            ax.set_xlim(0, img.shape[1] - 1)
            ax.set_ylim(img.shape[0] - 1, 0)

    # 大标题
    fig.suptitle(instruction, fontsize=18, color='white', backgroundcolor='#7B1FA2', y=0.99)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)

    if wandb:
        return img_pil

    img_pil.save(save_path)

def visualize_traj(images, gt_res, pred_res, intrinsics=None, instruction='', save_path='output_point.png', wandb=False):
    '''
    images: dict, {img_name: img (np.uint8, RGB, shape=(H,W,3)/(torch.Tensor, RGB, shape=3,H,W))}
    gt_res/pred_res: dict, {obj: [...]}, 每个obj: loc/trans/rot/size/class
    intrinsics: dict, {img_name: intr (matrix or [fx,fy,cx,cy])}
    '''
    img_names = list(images.keys())
    N = len(img_names)
    fig, axs = plt.subplots(2, N, figsize=(4 * N, 8))
    for row, mode in enumerate(['gt','pred']):
        obj_dict = gt_res if mode=='gt' else pred_res
        for col, img_name in enumerate(img_names):
            img = convert_to_opencv_format(images[img_name])
            if img.shape[-1]==3 and img.dtype==np.uint8:
                img_rgb = img.copy()
            else:
                img_rgb = img
            h, w = img_rgb.shape[:2]

            if intrinsics[img_name] is not None:
                K = build_K_from_intrinsics(intrinsics[img_name])

            ax = axs[row, col] if N > 1 else axs[row]
            ax.imshow(img_rgb)
            objs = obj_dict.get(img_name, [])
            # -------叠加画所有特征
            for idx, obj in enumerate(objs):
                if obj is None: continue
                if obj['trans'] is not None and obj['rot'] is not None:
                    if len(objs) > 1:
                        alpha = 0.25 + 0.75 * (1 - idx / (len(objs) - 1))
                    else:
                        alpha = 1.0
                    draw_axes_matplot(ax, obj['trans'], obj['rot'], K, length=0.1, alpha=alpha)

                    #TODO: length=obj["size"].max() * 0.5
                    # draw_axes_matplot(ax, obj['trans'], obj['rot'], K, length=0.1)
            ax.set_title(f"{img_name} - {mode.upper()}", fontsize=14, backgroundcolor='#F3E5F5' if mode=='gt' else '#E1F5FE', color='#222', pad=10)
            ax.axis('off')

    # 大标题
    fig.suptitle(instruction, fontsize=18, color='white', backgroundcolor='#7B1FA2', y=0.99)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)

    if wandb:
        return img_pil

    img_pil.save(save_path)

def visualize_views(
    image_dict,    # {name: tensor}, shape (B, History, C, H, W), C=3
    depth_dict,    # {name: tensor}, shape (B, History, 2, H, W)，通常第0通道是depth
    ray_dict,      # {name: tensor}, shape (B, History, 3, H, W)
    mask_dict=None,   # 可选, {name: tensor}, shape (B, History, H, W) or (H, W)
    figsize=(12, 6),
    show_channel_labels=True,
    save_path="output_depth_ray.png"
):
    """
    一行可视化每个view的rgb、depth、rays(normal map)，对比显示。
    """
    names = list(image_dict.keys())
    num_views = len(names)
    num_types = 3   # rgb, depth, ray
    fig, axs = plt.subplots(num_views, num_types, figsize=(figsize[0], figsize[1]*num_views))

    # 如果只有1个view，axis shape修正
    if num_views == 1:
        axs = axs[None, :]
    elif num_types == 1:
        axs = axs[:, None]

    for i, name in enumerate(names):
        # RGB 图像
        img = image_dict[name]  # (C,H,W)
        img = img.permute(1, 2, 0).float().cpu().numpy()        # (H,W,C)
        img = img.clip(0, 1)                            # 0-1

        axs[i,0].imshow(img)
        axs[i,0].set_title(f"{name}\nRGB" if show_channel_labels else name)
        axs[i,0].axis('off')

        # Depth (通常选第0通道，或你用 [batch_idx, history_idx, 0] )
        depth_map = depth_dict[name][0].float().cpu().numpy()  # (H,W)
        # 显示为灰度图
        axs[i,1].imshow(depth_map, cmap='plasma')  # 或 'viridis', 'magma'
        axs[i,1].set_title(f"Depth", fontsize=10)
        axs[i,1].axis('off')

        # Rays (normal map风格)
        rays = ray_dict[name]                             # (3,H,W)
        rays = rays.permute(1,2,0)                        # (H,W,3)
        norm = torch.linalg.norm(rays, dim=2, keepdim=True) + 1e-6
        rays_rgb = (rays / norm + 1) / 2                  # 映射到[0,1]
        rays_rgb = rays_rgb.clamp(0,1).float().cpu().numpy()

        # mask
        if mask_dict is not None and name in mask_dict:
            mask = mask_dict[name]
            if mask.ndim == 4:
                mask_show = mask
            else:
                mask_show = mask
            rays_rgb[mask_show.cpu().numpy()==0] = 0
            img[mask_show.cpu().numpy()==0] = 0
            depth_map[mask_show.cpu().numpy()==0] = 0

        axs[i,2].imshow(rays_rgb)
        axs[i,2].set_title(f"Ray(Normal Map)", fontsize=10)
        axs[i,2].axis('off')

    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)
    #
    # if wandb:
    #     return img_pil
    img_pil.save(save_path)

