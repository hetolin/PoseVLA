from posevla.modeling_posevla import PoseVLAPolicy, PoseVLAConfig
from data.collators import CollatorForDetectionDataset
from data.ds_train.detection.dataset_omni3d import Omni3DConsumerDataset
import hydra
from omegaconf import DictConfig
from posevla.modeling_posevla import bin_tokenizer
from utils.mapping_token import decode_text_to_scene_with_tokenizer
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pickle

"""
评估 DetAny3D 在 pi0 数据集上的推理结果

参考 GenPose2 的评估流程,计算以下指标:
- 3D IoU (mean, acc@0.25, acc@0.5, acc@0.75)
- Pose Error (rotation error in degrees, translation error in cm)
- Pose Accuracy (5°2cm, 5°5cm, 10°2cm, 10°5cm)
- Depth Error
- Size Error

使用方法:
    python evaluate_pi0_results.py \
        --pred_json exps/inference_pi0/1112-072536/pi0_pi0_output_results.json \
        --config_path detect_anything/configs/inference_pi0_gt_prompt.yaml
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict


def quaternion_to_rotation_matrix(quat_wxyz):
    """
    将四元数转换为旋转矩阵

    参数:
        quat_wxyz: [w, x, y, z] 四元数
    返回:
        3x3 旋转矩阵
    """
    w, x, y, z = quat_wxyz

    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

    return R

def compute_rotation_error_degrees(R_pred, R_gt):
    """
    计算两个旋转矩阵之间的角度误差（度）

    参数:
        R_pred: 3x3 预测旋转矩阵
        R_gt: 3x3 GT 旋转矩阵
    返回:
        角度误差（度）
    """
    # R_error = R_pred @ R_gt^T
    R_error = R_pred @ R_gt.T

    # 旋转角度 = arccos((trace(R_error) - 1) / 2)
    trace = np.trace(R_error)
    # 确保 trace 在有效范围内 [-1, 3]
    trace = np.clip(trace, -1, 3)
    angle = np.arccos((trace - 1) / 2)

    return np.degrees(angle)


def compute_translation_error_cm(t_pred, t_gt):
    """
    计算平移误差（厘米）

    参数:
        t_pred: [x, y, z] 预测平移向量（米）
        t_gt: [x, y, z] GT 平移向量（米）
    返回:
        欧氏距离（厘米）
    """
    return np.linalg.norm(t_pred - t_gt) * 100  # 转换为厘米


def compute_3d_iou(corners_pred, corners_gt):
    """
    计算两个 3D 边界框的 IoU

    参数:
        corners_pred: [8, 3] 预测的 8 个角点坐标
        corners_gt: [8, 3] GT 的 8 个角点坐标
    返回:
        3D IoU 值
    """
    # 简化实现：计算轴对齐边界框的 IoU
    # 对于更精确的 3D IoU,需要使用 oriented bounding box 的交集体积计算

    # 计算 AABB (Axis-Aligned Bounding Box)
    min_pred = corners_pred.min(axis=0)
    max_pred = corners_pred.max(axis=0)
    min_gt = corners_gt.min(axis=0)
    max_gt = corners_gt.max(axis=0)

    # 计算交集
    inter_min = np.maximum(min_pred, min_gt)
    inter_max = np.minimum(max_pred, max_gt)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_volume = np.prod(inter_dims)

    # 计算并集
    pred_volume = np.prod(max_pred - min_pred)
    gt_volume = np.prod(max_gt - min_gt)
    union_volume = pred_volume + gt_volume - inter_volume

    if union_volume == 0:
        return 0.0

    return inter_volume / union_volume


def corners_from_pose_and_size(center, rotation_matrix, dimensions):
    """
    根据中心、旋转矩阵和尺寸计算 3D 边界框的 8 个角点

    参数:
        center: [x, y, z] 中心坐标
        rotation_matrix: [3, 3] 旋转矩阵
        dimensions: [w, h, l] 尺寸
    返回:
        corners: [8, 3] 角点坐标
    """
    w, h, l = dimensions

    # 在物体坐标系下的 8 个角点（中心在原点）
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    z_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])

    corners_obj = np.vstack([x_corners, y_corners, z_corners])  # [3, 8]

    # 旋转到相机坐标系
    corners_cam = rotation_matrix @ corners_obj  # [3, 8]

    # 平移到中心位置
    corners_cam = corners_cam.T + center  # [8, 3]

    return corners_cam


def evaluate_sample(pred_sample: dict, gt_sample: dict, camera_idx: int = 0):
    """
    评估单个样本的预测结果, 返回每个预测对GT的配对信息, 以及每个GT的指标
    """
    # GT部分
    gt_poses = gt_sample['pose'][0][camera_idx]
    gt_bbox_sizes = gt_sample['size'][0][camera_idx]
    gt_class_labels = gt_sample['class_label'][0][camera_idx]
    num_gt = len(gt_class_labels)

    img_idx = f'image{camera_idx}'
    pred_list = pred_sample.get(img_idx, [])
    num_pred = len(pred_list)

    if num_gt == 0:
        return None
    # 构建GT
    gt_data = []
    for i in range(num_gt):
        pose = gt_poses[i]
        quat_wxyz = pose[:4].float().cpu().numpy()
        translation = pose[4:7].cpu().numpy()
        bbox_size = gt_bbox_sizes[i].cpu().numpy()
        R_gt = quaternion_to_rotation_matrix(quat_wxyz)
        corners_gt = corners_from_pose_and_size(translation, R_gt, bbox_size)
        gt_data.append({'translation': translation, 'rotation': R_gt, 'size': bbox_size,
                        'corners': corners_gt, 'class_label': gt_class_labels[i]})

    # 构建所有预测（score=1.0 固定）
    pred_data = []
    for pred in pred_list:
        if pred['trans'] is None or pred['rot'] is None or pred['size'] is None:
            continue
        center_cam = np.array(pred['trans'])
        pose_matrix = R.from_euler('xyz', pred["rot"]).as_matrix()
        dimensions = np.array(pred['size'])
        bbox3D = corners_from_pose_and_size(center_cam, pose_matrix, dimensions)
        pred_data.append({
            'translation': center_cam,
            'rotation': pose_matrix,
            'size': dimensions,
            'corners': bbox3D,
            'score': 1.0
        })
    num_pred = len(pred_data)

    # 匹配配对
    pred_iou = np.zeros(num_pred)          # 每个预测和其分派GT的IoU
    pred_gt_idx = np.full(num_pred, -1)    # -1表示未分配
    pred_matched = np.zeros(num_pred, dtype=bool)

    iou_matrix = np.zeros((num_pred, num_gt))
    for i, pred in enumerate(pred_data):
        for j, gt in enumerate(gt_data):
            iou_matrix[i, j] = compute_3d_iou(pred['corners'], gt['corners'])

    # 贪心配对,每个GT/预测只能被分一次
    matched_pred = set()
    matched_gt = set()
    iou_pairs = []
    for i in range(num_pred):
        for j in range(num_gt):
            iou_pairs.append((iou_matrix[i, j], i, j))
    iou_pairs.sort(reverse=True)  # 按IoU降序

    for iou_val, pred_idx, gt_idx in iou_pairs:
        if pred_idx not in matched_pred and gt_idx not in matched_gt and iou_val > 0:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            pred_iou[pred_idx] = iou_val
            pred_gt_idx[pred_idx] = gt_idx
            pred_matched[pred_idx] = True

    # 未分配预测最大iou是多少
    for i in range(num_pred):
        if not pred_matched[i]:
            pred_iou[i] = iou_matrix[i].max() if num_gt > 0 else 0

    # 统计每个GT有没有分到预测
    gt_matched = np.zeros(num_gt, dtype=bool)
    for gt_idx in matched_gt:
        gt_matched[gt_idx] = True

    # 统计每个GT的误差信息
    gt_metrics = {
        'iou_3d': np.zeros(num_gt),
        'rotation_error': np.full(num_gt, np.inf),
        'translation_error': np.full(num_gt, np.inf),
        'depth_error': np.full(num_gt, np.inf),
        'size_error': np.full(num_gt, np.inf),
        'matched': gt_matched
    }
    for pred_idx, gt_idx in zip(np.where(pred_matched)[0], pred_gt_idx[pred_matched]):
        pred = pred_data[pred_idx]
        gt = gt_data[gt_idx]

        gt_metrics['iou_3d'][gt_idx] = pred_iou[pred_idx]

        gt_metrics['rotation_error'][gt_idx] = compute_rotation_error_degrees(
            pred['rotation'], gt['rotation']
        )
        gt_metrics['translation_error'][gt_idx] = compute_translation_error_cm(
            pred['translation'], gt['translation']
        )
        gt_metrics['depth_error'][gt_idx] = abs(
            pred['translation'][2] - gt['translation'][2]) * 100
        gt_metrics['size_error'][gt_idx] = np.mean(np.abs(
            pred['size'] - gt['size'])) * 100

    return {
        'pred_scores': np.array([pred['score'] for pred in pred_data]),
        'pred_iou': pred_iou,
        'pred_matched': pred_matched,
        'pred_gt_idx': pred_gt_idx,
        'total_gt': num_gt,
        'gt_metrics': gt_metrics
    }

# def compute_pr_ap(pred_scores, pred_iou, pred_matched, total_gt, iou_threshold, sort_by='iou'):
#     """可以指定按score或iou排序，score为主，iou仅用于分析"""
#     if sort_by == 'score':
#         idx_sorted = np.argsort(-pred_scores)
#     elif sort_by == 'iou':
#         idx_sorted = np.argsort(-pred_iou)
#     else:
#         raise ValueError("sort_by should be 'score' or 'iou'")
#     pred_scores = pred_scores[idx_sorted]
#     pred_iou = pred_iou[idx_sorted]
#     pred_matched = pred_matched[idx_sorted]
#
#     is_tp = (pred_iou >= iou_threshold) & pred_matched
#     tp = np.cumsum(is_tp)
#     fp = np.cumsum(~is_tp)
#     recalls = tp / (total_gt if total_gt > 0 else 1)
#     precisions = tp / (tp + fp + 1e-8)
#
#     ap = 0.
#     for t in np.linspace(0, 1, 101):
#         prec = precisions[recalls >= t].max() if np.any(recalls >= t) else 0
#         ap += prec / 101
#     return recalls, precisions, ap

# def compute_pr_ap(pred_scores, pred_iou, pred_matched, total_gt, iou_threshold):
#     # 选择所有预测，无需排序
#     is_tp = (pred_iou >= iou_threshold) & pred_matched
#     tp = np.sum(is_tp)
#     fp = len(pred_scores) - tp
#     recall = tp / (total_gt if total_gt > 0 else 1)
#     precision = tp / (tp + fp + 1e-8)
#     ap = precision  # 只有一个点，AP=precision
#     # 返回的recalls、precisions可以也是单点
#     return np.array([recall]), np.array([precision]), ap

def compute_pr_ap(pred_scores, pred_iou, pred_matched, total_gt, iou_threshold, use_101_point=True):
    """
    兼容两种情况的AP计算。
    :param pred_scores: 预测框置信分数，(N, )
    :param pred_iou: 每个预测框与最佳GT的IoU，(N, )
    :param pred_matched: 每个预测框是否与GT成功匹配（按唯一性），(N, )
    :param total_gt: 所有关GT框数（单个类别统计）
    :param iou_threshold: IoU阈值
    :param use_101_point: 是否用VOC/COCO插值法，否则返回细分点AP
    :return: recalls, precisions, ap
    """
    if len(pred_scores) == 0:
        return np.array([0]), np.array([0]), 0.0

    # TODO: 先判断score是否全为1
    # if np.allclose(pred_scores, 1):
    #     # 无需排序，PR曲线只有1点
    #     is_tp = (pred_iou >= iou_threshold) & pred_matched
    #     tp = np.sum(is_tp)
    #     fp = len(pred_scores) - tp
    #     recall = tp / (total_gt if total_gt > 0 else 1)
    #     precision = tp / (tp + fp + 1e-8)
    #     ap = precision
    #     return np.array([recall]), np.array([precision]), ap

    # 有score，先按score降序排序
    # 当score全为1时按score排序无意义，改为按IoU排序，使TP排在FP前面，AP值稳定可复现
    # order = np.argsort(-pred_scores)
    order = np.argsort(-pred_iou)
    pred_scores_sorted = pred_scores[order]
    pred_iou_sorted = pred_iou[order]
    pred_matched_sorted = pred_matched[order]
    is_tp = (pred_iou_sorted >= iou_threshold) & pred_matched_sorted

    # 累加TP、FP
    tps = is_tp.astype(np.float32)
    fps = (~is_tp).astype(np.float32)
    cumul_tp = np.cumsum(tps)
    cumul_fp = np.cumsum(fps)

    recalls = cumul_tp / (total_gt if total_gt > 0 else 1)
    precisions = cumul_tp / (cumul_tp + cumul_fp + 1e-8)

    # 标准插值AP（101点法）
    if use_101_point:
        ap = 0.0
        recall_points = np.linspace(0, 1, 101)
        precisions_interp = []
        for r in recall_points:
            # 对于Recall >= r的点求对应Precision最大值
            inds = np.where(recalls >= r)[0]
            if len(inds) > 0:
                precisions_interp.append(np.max(precisions[inds]))
            else:
                precisions_interp.append(0.0)
        ap = np.mean(precisions_interp)
        return recall_points, np.array(precisions_interp), ap
    else:
        # 直接积分
        ap = voc_ap(recalls, precisions)
        return recalls, precisions, ap

def voc_ap(recalls, precisions):
    # VOC-2007 11-point法，或直接积分
    # 这里只用更常见的阶梯积分
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    inds = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[inds+1] - recalls[inds]) * precisions[inds+1])
    return ap

def compute_metrics_summary(all_results, save_dir, plot_pr=True):
    all_pred_scores = []
    all_pred_ious = []
    all_pred_matched = []
    total_gt = 0
    gt_metrics_list = []

    for result in all_results:
        if result is None:
            continue
        all_pred_scores.append(result['pred_scores'])
        # all_pred_scores.append(np.ones_like(result['pred_iou']))  # 每个预测score固定为1
        all_pred_ious.append(result['pred_iou'])
        all_pred_matched.append(result['pred_matched'])
        total_gt += result['total_gt']
        gt_metrics_list.append(result['gt_metrics'])

    if total_gt == 0 or len(all_pred_scores) == 0:
        print("Warning: No valid prediction or GT!")
        return {}

    # 全部预测数据
    all_pred_scores = np.concatenate(all_pred_scores)
    all_pred_ious = np.concatenate(all_pred_ious)
    all_pred_matched = np.concatenate(all_pred_matched)

    os.makedirs(save_dir, exist_ok=True)
    # PR/AP
    iou_thresholds = np.arange(0.05, 0.51, 0.05)
    ap_dict = {}
    pr_curves = {}
    plt.figure()
    for th in iou_thresholds:
        recalls, precisions, ap = compute_pr_ap(
            all_pred_scores, all_pred_ious, all_pred_matched, total_gt, th
        )
        ap_dict[f"AP@{th:.2f}"] = ap
        pr_curves[f"PR@{th:.2f}"] = (recalls, precisions)
        if plot_pr:
            plt.plot(recalls, precisions, label=f"IoU>{th:.2f}, AP={ap:.4f}")
    if plot_pr:
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curves under different IoU thresholds')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(save_dir, 'pr_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # AP vs IoU曲线
        plt.figure()
        plt.plot(iou_thresholds, [ap_dict[f"AP@{th:.2f}"] for th in iou_thresholds], marker='o')
        plt.xlabel("IoU Threshold")
        plt.ylabel("AP")
        plt.title("AP at Different IoU Thresholds")
        plt.grid(True)
        save_path = os.path.join(save_dir, 'ap_vs_iou.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # 精度统计按GT角度
    all_iou = np.concatenate([m['iou_3d'] for m in gt_metrics_list])
    all_rot_error = np.concatenate([m['rotation_error'] for m in gt_metrics_list])
    all_trans_error = np.concatenate([m['translation_error'] for m in gt_metrics_list])
    all_depth_error = np.concatenate([m['depth_error'] for m in gt_metrics_list])
    all_size_error = np.concatenate([m['size_error'] for m in gt_metrics_list])
    all_gt_matched = np.concatenate([m['matched'] for m in gt_metrics_list])
    total_matched = all_gt_matched.sum()

    # 只统计匹配的GT
    matched_iou = all_iou[all_gt_matched]
    matched_rot_error = all_rot_error[all_gt_matched]
    matched_trans_error = all_trans_error[all_gt_matched]
    matched_depth_error = all_depth_error[all_gt_matched]
    matched_size_error = all_size_error[all_gt_matched]

    metrics = {
        'recall': total_matched / total_gt if total_gt > 0 else 0.0,
        'total_gt': total_gt,
        'total_matched': total_matched,
        'iou_mean': np.mean(matched_iou),
        'iou_median': np.median(matched_iou),
        'iou_acc_25': np.mean(matched_iou >= 0.25),
        'iou_acc_50': np.mean(matched_iou >= 0.50),
        'iou_acc_75': np.mean(matched_iou >= 0.75),
        'rot_error_mean': np.mean(matched_rot_error),
        'rot_error_median': np.median(matched_rot_error),
        'trans_error_mean': np.mean(matched_trans_error),
        'trans_error_median': np.median(matched_trans_error),
        'depth_error_mean': np.mean(matched_depth_error),
        'depth_error_median': np.median(matched_depth_error),
        'size_error_mean': np.mean(matched_size_error),
        'size_error_median': np.median(matched_size_error),
        'pose_acc_5deg_2cm': np.mean((matched_rot_error < 5) & (matched_trans_error < 2)),
        'pose_acc_5deg_5cm': np.mean((matched_rot_error < 5) & (matched_trans_error < 5)),
        'pose_acc_10deg_2cm': np.mean((matched_rot_error < 10) & (matched_trans_error < 2)),
        'pose_acc_10deg_5cm': np.mean((matched_rot_error < 10) & (matched_trans_error < 5)),
        'pose_acc_15deg_10cm': np.mean((matched_rot_error < 15) & (matched_trans_error < 10)),
    }
    metrics.update(ap_dict)
    metrics['mean_AP'] = np.mean(list(ap_dict.values()))
    metrics['pr_curves'] = pr_curves
    return metrics

# def evaluate_sample(pred_sample: List[Dict], gt_sample: Dict, camera_idx: int = 0):
#     """
#     评估单个样本的预测结果
#
#     参数:
#         pred_list: 该样本的所有预测结果列表
#         gt_sample: pi0 数据集的 GT 样本
#         camera_idx: 相机索引（默认 0 = top_head）
#
#     返回:
#         metrics: 包含各种指标的字典
#     """
#     # 提取 GT 数据
#     # gt_poses = gt_sample['camera_pose'][camera_idx]  # [N, 7] (quat_wxyz + translation)
#     # gt_bbox_sizes = gt_sample['camera_bbox_side_len'][camera_idx]  # [N, 3]
#     # gt_class_labels = gt_sample['camera_class_labels'][camera_idx]  # List[int]
#     # gt_bboxes_2d = gt_sample['bbox_2d'][camera_idx]  # List[Tensor[4]]
#
#     gt_poses = gt_sample['pose'][0][camera_idx]  # [N, 7] (quat_wxyz + translation)
#     gt_bbox_sizes = gt_sample['size'][0][camera_idx]  # [N, 3]
#     gt_class_labels = gt_sample['class_label'][0][camera_idx]  # List[int]
#     gt_bboxes_2d = gt_sample['bbox_2d'][0][camera_idx]  # List[Tensor[4]]
#
#     num_gt = len(gt_class_labels)
#
#     # 判断imagen是否检测到目标物体
#     img_idx = "image{}".format(camera_idx)
#     if img_idx in pred_sample.keys():
#         pred_list = pred_sample[img_idx]
#     else:
#         pred_list = []
#
#     num_pred = len(pred_list)
#
#     if num_gt == 0:
#         return None
#
#     # 构建 GT 数据结构
#     gt_data = []
#     for i in range(num_gt):
#         pose = gt_poses[i]
#         quat_wxyz = pose[:4].numpy()
#         translation = pose[4:7].numpy()
#         bbox_size = gt_bbox_sizes[i].numpy()
#
#         # 四元数转旋转矩阵
#         R_gt = quaternion_to_rotation_matrix(quat_wxyz)
#
#         # 计算角点
#         corners_gt = corners_from_pose_and_size(translation, R_gt, bbox_size)
#
#         gt_data.append({
#             'translation': translation,
#             'rotation': R_gt,
#             'size': bbox_size,
#             'corners': corners_gt,
#             'class_label': gt_class_labels[i]
#         })
#
#     # 如果没有预测结果,返回全零指标
#     if num_pred == 0:
#         return {
#             'iou_3d': np.zeros(num_gt),
#             'rotation_error': np.full(num_gt, np.inf),
#             'translation_error': np.full(num_gt, np.inf),
#             'depth_error': np.full(num_gt, np.inf),
#             'size_error': np.full(num_gt, np.inf),
#             'matched': np.zeros(num_gt, dtype=bool)
#         }
#
#     # 构建预测数据结构
#     pred_data = []
#     for pred in pred_list:
#         # 从 JSON 提取数据
#
#         if pred['trans'] is None or pred['rot'] is None or pred['size'] is None:
#             continue
#         center_cam = np.array(pred['trans'])
#         pose_matrix = R.from_euler('xyz', pred["rot"]).as_matrix()  # [3, 3] 旋转矩阵
#         dimensions = np.array(pred['size'])  # [w, h, l]
#         bbox3D = corners_from_pose_and_size(center_cam, pose_matrix, dimensions)  # [8, 3] 角点
#
#         pred_data.append({
#             'translation': center_cam,
#             'rotation': pose_matrix,
#             'size': dimensions,
#             'corners': bbox3D,
#             'score': 1.0 #pred['score']
#         })
#
#     # 计算所有 pred-gt 配对的指标
#     iou_matrix = np.zeros((num_pred, num_gt))
#     rot_error_matrix = np.zeros((num_pred, num_gt))
#     trans_error_matrix = np.zeros((num_pred, num_gt))
#
#     for i, pred in enumerate(pred_data):
#         for j, gt in enumerate(gt_data):
#             # 3D IoU
#             iou_matrix[i, j] = compute_3d_iou(pred['corners'], gt['corners'])
#
#             # 旋转误差
#             rot_error_matrix[i, j] = compute_rotation_error_degrees(
#                 pred['rotation'], gt['rotation']
#             )
#
#             # 平移误差
#             trans_error_matrix[i, j] = compute_translation_error_cm(
#                 pred['translation'], gt['translation']
#             )
#
#     # 使用贪心匹配（基于 IoU）
#     matched_pred = set()
#     matched_gt = set()
#     matches = []
#
#     # 按 IoU 从大到小排序
#     iou_pairs = []
#     for i in range(num_pred):
#         for j in range(num_gt):
#             iou_pairs.append((iou_matrix[i, j], i, j))
#     iou_pairs.sort(reverse=True)
#
#     for iou_val, pred_idx, gt_idx in iou_pairs:
#         if pred_idx not in matched_pred and gt_idx not in matched_gt:
#             if iou_val > 0:  # 只匹配有正 IoU 的
#                 matches.append((pred_idx, gt_idx))
#                 matched_pred.add(pred_idx)
#                 matched_gt.add(gt_idx)
#
#     # 构建每个 GT 的指标
#     iou_3d = np.zeros(num_gt)
#     rotation_error = np.full(num_gt, np.inf)
#     translation_error = np.full(num_gt, np.inf)
#     depth_error = np.full(num_gt, np.inf)
#     size_error = np.full(num_gt, np.inf)
#     matched = np.zeros(num_gt, dtype=bool)
#
#     for pred_idx, gt_idx in matches:
#         iou_3d[gt_idx] = iou_matrix[pred_idx, gt_idx]
#         rotation_error[gt_idx] = rot_error_matrix[pred_idx, gt_idx]
#         translation_error[gt_idx] = trans_error_matrix[pred_idx, gt_idx]
#
#         # 深度误差（Z 轴）
#         depth_error[gt_idx] = abs(
#             pred_data[pred_idx]['translation'][2] -
#             gt_data[gt_idx]['translation'][2]
#         ) * 100  # 转换为厘米
#
#         # 尺寸误差（L1 距离）
#         size_error[gt_idx] = np.mean(np.abs(
#             pred_data[pred_idx]['size'] - gt_data[gt_idx]['size']
#         )) * 100  # 转换为厘米
#
#         matched[gt_idx] = True
#
#     return {
#         'iou_3d': iou_3d,
#         'rotation_error': rotation_error,
#         'translation_error': translation_error,
#         'depth_error': depth_error,
#         'size_error': size_error,
#         'matched': matched
#     }
#
#
# def compute_pr_ap(ious, total_gt, iou_threshold):
#     ious = np.array(ious)
#
#     # TP: IoU > iou_threshold, FP: 其他
#     is_tp = ious >= iou_threshold
#     tp = np.cumsum(is_tp)
#     fp = np.cumsum(~is_tp)
#     # num_gt = len(is_tp)  # 或根据你的实际GT数量
#     num_gt = total_gt  # 或根据你的实际GT数量
#
#     recalls = tp / (num_gt if num_gt > 0 else 1)
#     precisions = tp / (tp + fp + 1e-8)
#     # AP 计算类似COCO标准（在每个recall点取最大precision均值）
#     ap = 0.
#     for t in np.linspace(0, 1, 101):
#         prec = precisions[recalls >= t].max() if np.any(recalls >= t) else 0
#         ap += prec / 101
#     return recalls, precisions, ap
#
# def compute_metrics_summary(all_results: List[Dict], save_dir, plot_pr=True) -> Dict:
#     # 收集所有有效的指标值
#     all_iou = []
#     all_rot_error = []
#     all_trans_error = []
#     all_depth_error = []
#     all_size_error = []
#     total_gt = 0
#     total_matched = 0
#
#     # # AP部分需要收集所有预测的scores和iou_3d
#     # all_scores = []
#     all_ious = []
#
#     for result in all_results:
#         if result is None:
#             continue
#
#         matched = result['matched']
#         total_gt += len(matched)
#         total_matched += matched.sum()
#
#         for i in range(len(matched)):
#             if matched[i]:
#                 all_iou.append(result['iou_3d'][i])
#                 all_rot_error.append(result['rotation_error'][i])
#                 all_trans_error.append(result['translation_error'][i])
#                 all_depth_error.append(result['depth_error'][i])
#                 all_size_error.append(result['size_error'][i])
#             # # AP & PR 曲线统计用
#             # # 假设每个sample有 scores、iou_3d字段，并且一一对应
#             # # all_scores.append(result['scores'][i])
#             all_ious.append(result['iou_3d'][i])
#
#     if len(all_iou) == 0:
#         print("警告: 没有匹配的预测结果!")
#         return {}
#
#     all_iou = np.array(all_iou)
#     all_rot_error = np.array(all_rot_error)
#     all_trans_error = np.array(all_trans_error)
#     all_depth_error = np.array(all_depth_error)
#     all_size_error = np.array(all_size_error)
#
#     # 计算召回率
#     recall = total_matched / total_gt if total_gt > 0 else 0
#
#     # AP/PR部分
#     iou_thresholds = np.arange(0.05, 0.51, 0.05)
#     ap_dict = {}
#     pr_curves = {}
#     for th in iou_thresholds:
#         recalls, precisions, ap = compute_pr_ap(all_ious, total_gt, iou_threshold=th)
#         ap_dict[f"AP@{th:.2f}"] = ap
#         pr_curves[f"PR@{th:.2f}"] = (recalls, precisions)
#         if plot_pr:
#             plt.plot(recalls, precisions, label=f"IoU>{th:.2f}, AP={ap:.4f}")
#     if plot_pr:
#         # 原来的 pr 曲线图 ...
#         save_path = os.path.join(save_dir, 'pr_curves.png')
#         plt.figure()
#         for th in iou_thresholds:
#             recalls, precisions = pr_curves[f"PR@{th:.2f}"]
#             plt.plot(recalls, precisions, label=f"IoU>{th:.2f}")
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.title('PR curves under different IoU thresholds')
#         # plt.legend()
#         plt.grid(True)
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.show()
#
#         # 新增的 AP vs IoU 曲线图
#         save_path = os.path.join(save_dir, 'ap_vs_iou.png')
#         ap_list = [ap_dict[f"AP@{th:.2f}"] for th in iou_thresholds]
#         plt.figure()
#         plt.plot(iou_thresholds, ap_list, marker='o')
#         plt.xlabel("IoU Threshold")
#         plt.ylabel("AP")
#         plt.title("AP at Different IoU Thresholds")
#         plt.grid(True)
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.show()
#
#     metrics = {
#         # 原有指标
#         'recall': recall,
#         'total_gt': total_gt,
#         'total_matched': total_matched,
#         'iou_mean': all_iou.mean(),
#         'iou_median': np.median(all_iou),
#         'iou_acc_25': (all_iou >= 0.25).mean(),
#         'iou_acc_50': (all_iou >= 0.50).mean(),
#         'iou_acc_75': (all_iou >= 0.75).mean(),
#         'rot_error_mean': all_rot_error.mean(),
#         'rot_error_median': np.median(all_rot_error),
#         'trans_error_mean': all_trans_error.mean(),
#         'trans_error_median': np.median(all_trans_error),
#         'depth_error_mean': all_depth_error.mean(),
#         'depth_error_median': np.median(all_depth_error),
#         'size_error_mean': all_size_error.mean(),
#         'size_error_median': np.median(all_size_error),
#         'pose_acc_5deg_2cm': ((all_rot_error < 5) & (all_trans_error < 2)).mean(),
#         'pose_acc_5deg_5cm': ((all_rot_error < 5) & (all_trans_error < 5)).mean(),
#         'pose_acc_10deg_2cm': ((all_rot_error < 10) & (all_trans_error < 2)).mean(),
#         'pose_acc_10deg_5cm': ((all_rot_error < 10) & (all_trans_error < 5)).mean(),
#         'pose_acc_15deg_10cm': ((all_rot_error < 15) & (all_trans_error < 10)).mean(),
#     }
#     metrics.update(ap_dict)
#     metrics['mean_AP'] = np.mean(list(ap_dict.values()))
#     metrics['pr_curves'] = pr_curves  # 可选返回PR曲线数据
#
#     return metrics


# def compute_metrics_summary(all_results: List[Dict]) -> Dict:
#     """
#     汇总所有样本的指标
#
#     参数:
#         all_results: 所有样本的评估结果列表
#
#     返回:
#         汇总的指标字典
#     """
#     # 收集所有有效的指标值
#     all_iou = []
#     all_rot_error = []
#     all_trans_error = []
#     all_depth_error = []
#     all_size_error = []
#     total_gt = 0
#     total_matched = 0
#
#     for result in all_results:
#         if result is None:
#             continue
#
#         matched = result['matched']
#         total_gt += len(matched)
#         total_matched += matched.sum()
#
#         # 只统计匹配上的样本
#         for i in range(len(matched)):
#             if matched[i]:
#                 all_iou.append(result['iou_3d'][i])
#                 all_rot_error.append(result['rotation_error'][i])
#                 all_trans_error.append(result['translation_error'][i])
#                 all_depth_error.append(result['depth_error'][i])
#                 all_size_error.append(result['size_error'][i])
#
#     if len(all_iou) == 0:
#         print("警告: 没有匹配的预测结果!")
#         return {}
#
#     all_iou = np.array(all_iou)
#     all_rot_error = np.array(all_rot_error)
#     all_trans_error = np.array(all_trans_error)
#     all_depth_error = np.array(all_depth_error)
#     all_size_error = np.array(all_size_error)
#
#     # 计算召回率
#     recall = total_matched / total_gt if total_gt > 0 else 0
#
#     # 计算各种指标
#     metrics = {
#         # 召回率
#         'recall': recall,
#         'total_gt': total_gt,
#         'total_matched': total_matched,
#
#         # 3D IoU
#         'iou_mean': all_iou.mean(),
#         'iou_median': np.median(all_iou),
#         'iou_acc_25': (all_iou >= 0.25).mean(),
#         'iou_acc_50': (all_iou >= 0.50).mean(),
#         'iou_acc_75': (all_iou >= 0.75).mean(),
#
#         # 旋转误差（度）
#         'rot_error_mean': all_rot_error.mean(),
#         'rot_error_median': np.median(all_rot_error),
#
#         # 平移误差（厘米）
#         'trans_error_mean': all_trans_error.mean(),
#         'trans_error_median': np.median(all_trans_error),
#
#         # 深度误差（厘米）
#         'depth_error_mean': all_depth_error.mean(),
#         'depth_error_median': np.median(all_depth_error),
#
#         # 尺寸误差（厘米）
#         'size_error_mean': all_size_error.mean(),
#         'size_error_median': np.median(all_size_error),
#
#         # 姿态准确率（rotation deg, translation cm）
#         'pose_acc_5deg_2cm': ((all_rot_error < 5) & (all_trans_error < 2)).mean(),
#         'pose_acc_5deg_5cm': ((all_rot_error < 5) & (all_trans_error < 5)).mean(),
#         'pose_acc_10deg_2cm': ((all_rot_error < 10) & (all_trans_error < 2)).mean(),
#         'pose_acc_10deg_5cm': ((all_rot_error < 10) & (all_trans_error < 5)).mean(),
#         'pose_acc_15deg_10cm': ((all_rot_error < 15) & (all_trans_error < 10)).mean(),
#     }
#
#     return metrics


def print_metrics(metrics: Dict):
    """打印指标报告（自动支持多种AP指标）"""
    print("\n" + "="*60)
    print("评估结果报告")
    print("="*60)

    print(f"\n召回率:")
    print(f"  总 GT 数量: {metrics.get('total_gt', '-')}")
    print(f"  匹配数量: {metrics.get('total_matched', '-')}")
    print(f"  召回率: {metrics.get('recall', 0):.4f}")

    print(f"\n3D IoU:")
    print(f"  平均值: {metrics.get('iou_mean', 0):.4f}")
    print(f"  中位数: {metrics.get('iou_median', 0):.4f}")
    print(f"  IoU@0.25: {metrics.get('iou_acc_25', 0):.4f}")
    print(f"  IoU@0.50: {metrics.get('iou_acc_50', 0):.4f}")
    print(f"  IoU@0.75: {metrics.get('iou_acc_75', 0):.4f}")

    print(f"\n旋转误差 (度):")
    print(f"  平均值: {metrics.get('rot_error_mean', 0):.2f}°")
    print(f"  中位数: {metrics.get('rot_error_median', 0):.2f}°")

    print(f"\n平移误差 (厘米):")
    print(f"  平均值: {metrics.get('trans_error_mean', 0):.2f} cm")
    print(f"  中位数: {metrics.get('trans_error_median', 0):.2f} cm")

    print(f"\n深度误差 (厘米):")
    print(f"  平均值: {metrics.get('depth_error_mean', 0):.2f} cm")
    print(f"  中位数: {metrics.get('depth_error_median', 0):.2f} cm")

    print(f"\n尺寸误差 (厘米):")
    print(f"  平均值: {metrics.get('size_error_mean', 0):.2f} cm")
    print(f"  中位数: {metrics.get('size_error_median', 0):.2f} cm")

    print(f"\n姿态准确率:")
    print(f"  5°  2cm: {metrics.get('pose_acc_5deg_2cm', 0):.4f}")
    print(f"  5°  5cm: {metrics.get('pose_acc_5deg_5cm', 0):.4f}")
    print(f"  10° 2cm: {metrics.get('pose_acc_10deg_2cm', 0):.4f}")
    print(f"  10° 5cm: {metrics.get('pose_acc_10deg_5cm', 0):.4f}")
    print(f"  15° 10cm: {metrics.get('pose_acc_15deg_10cm', 0):.4f}")

    # 自动打印所有AP相关指标
    ap_keys = [k for k in metrics.keys() if k.startswith("AP@")]
    if ap_keys:
        print(f"\nAverage Precisions (AP) under不同IoU阈值:")
        for k in sorted(ap_keys):
            print(f"  {k}: {metrics[k]:.4f}")

        print(f"\nMean AP:")
        print(f"  mean_AP: {metrics.get('mean_AP', 0):.4f}")

    print("\n" + "="*60)


# 转换 numpy 类型为 Python 原生类型
def convert_to_native_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_native_types(item) for item in obj]   # <-----关键
    else:
        return obj

def save_all_results(all_results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"all_results 已保存到 {file_path}")

def load_all_results(file_path):
    with open(file_path, 'rb') as f:
        all_results = pickle.load(f)
    print(f"all_results 已从 {file_path} 加载")
    return all_results

def eval():
    save_dir = "./eval_results_objectron_weight"
    os.makedirs(save_dir, exist_ok=True)

    pkl_path = os.path.join(save_dir, "all_results.pkl")
    all_results = load_all_results(pkl_path)

    print(f"\n汇总指标...")
    metrics = compute_metrics_summary(all_results, save_dir)

    print_metrics(metrics)

    output_path = os.path.join(save_dir, 'evaluation_metrics.json')

    metrics_serializable = convert_to_native_types(metrics)

    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    print(f"\n指标已保存到: {output_path}")

@hydra.main(
        version_base=None,
        config_path="./config",
        config_name="base",
    )
def main(cfg: DictConfig) -> None:
    # ckpt_path = "ckpt/pi0_224_prior_bs7_gpu16_lr5e-5_decay1e-10/59999/model"
    # ckpt_path = "ckpt/pi0_224_prior_weighted_sample_bs7_gpu16_lr5e-5_decay1e-10/59999/model"
    # ckpt_path = "ckpt/pi0_224_prior_weighted_sample_bs7_gpu16_lr5e-5_decay1e-10/49999/model"
    ckpt_path = "ckpt/pi0_224_prior_weighted_sample_intra_neg_bs7_gpu16_lr5e-5_decay1e-10/59999/model"
    cfg.training.batch_size = 1

    weight_dtype = torch.bfloat16
    posevla_config = PoseVLAConfig(
        tokenizer_model_path=(cfg.model.tokenizer_model_path),
        n_action_steps=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,
        chunk_size=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,  # not used
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
        add_prior=True)

    # policy = PoseVLAPolicy(posevla_config)
    policy = PoseVLAPolicy.from_pretrained(ckpt_path, local_files_only=True, config=posevla_config)
    policy = policy.eval().to(weight_dtype).cuda().eval()

    # omni6d_dataset = Omni6dConsumerDataset(config=cfg, tokenizer=bin_tokenizer)
    omni3d_dataset = Omni3DConsumerDataset(config=cfg, tokenizer=bin_tokenizer, yaml_name="omni3d_test")
    data_vlm_collator = CollatorForDetectionDataset()
    train_vlm_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=omni3d_dataset, collate_fn=data_vlm_collator)

    all_results = []
    i = 0

    for batch in tqdm(train_vlm_dataloader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(weight_dtype).cuda()
        # i += 1
        # if i > 100:
        #     break
        output_res = policy.forward_evaluate_ntp(batch)
        pred, gt = output_res["pred"], output_res["gt"]

        pred_text = pred[0]
        gt_text = gt[0]
        pred_res = decode_text_to_scene_with_tokenizer(pred_text, bin_tokenizer)
        gt_res = decode_text_to_scene_with_tokenizer(gt_text, bin_tokenizer)
        # print(pred_res)

        # 可视化
        # images = {"image0": batch["observation.images.top_head"][0, 0],  # （B，history, 3, h, w）
        #           "image1": batch["observation.images.hand_left"][0, 0],
        #           "image2": batch["observation.images.hand_right"][0, 0]}
        # intrinsics = {"image0": batch["observation.images.top_head.intrinsics"][0],  # （B，3, 3）
        #               "image1": batch["observation.images.hand_left.intrinsics"][0],
        #               "image2": batch["observation.images.hand_right.intrinsics"][0]}
        # depths = {"image0": batch["observation.depth_priors.top_head"][0, 0],  # [B, history, 2, h, w])
        #           "image1": batch["observation.depth_priors.hand_left"][0, 0],
        #           "image2": batch["observation.depth_priors.hand_right"][0, 0]}
        # rays = {"image0": batch["observation.rays.top_head"][0, 0],  # (B,History,3,H,W)
        #         "image1": batch["observation.rays.hand_left"][0, 0],
        #         "image2": batch["observation.rays.hand_right"][0, 0]}
        # visualize_views(images, depths, rays)
        # visualize_2d_3d_all(images, gt_res, pred_res, intrinsics, batch["task"][0], )

        result = evaluate_sample(pred_res, batch, camera_idx=0)
        all_results.append(result)

    # 汇总指标
    print(f"\n汇总指标...")
    save_dir = "./eval_results_arkitscene_weight_intra_neg"
    os.makedirs(save_dir, exist_ok=True)

    metrics = compute_metrics_summary(all_results, save_dir)

    save_all_results(all_results, os.path.join(save_dir, "all_results.pkl"))

    # 打印结果
    print_metrics(metrics)

    # 保存结果（可选）
    output_path = os.path.join(save_dir, 'evaluation_metrics.json')

    metrics_serializable = convert_to_native_types(metrics)

    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    print(f"\n指标已保存到: {output_path}")



if __name__ == '__main__':
    main()
    # eval()