import os
from turtle import mode
import h5py
import numpy as np
import pickle
import argparse
from pathlib import Path
from torch import mode
import matplotlib.pyplot as plt
from tqdm import tqdm

def update_online_stats(x, count, mean, M2):
    """Welford 算法在线更新均值和方差"""
    if x.ndim == 1:
        x = x[None, :]
    for row in x:
        count += 1
        delta = row - mean
        mean += delta / count
        delta2 = row - mean
        M2 += delta * delta2
    return count, mean, M2

def compute_global_stats_from_processed(processed_root, mode='qpos'):
    print(f"正在从目录计算全局统计量 [{mode.upper()}]: {processed_root}")
    
    h5_files = list(Path(processed_root).rglob("episode_*.hdf5"))
    if not h5_files:
        raise RuntimeError(f"在 {processed_root} 下未找到任何 episode_*.hdf5 文件")
        
    print(f"共找到 {len(h5_files)} 个处理后的 HDF5 文件")

    act_key = f"actions_{mode}"
    obs_key = f"observations/state_{mode}"
    count_action, mean_action, M2_action = 0, None, None
    count_obs, mean_obs, M2_obs = 0, None, None
    
    sampled_actions = []

    for i, h5_path in enumerate(tqdm(h5_files, desc=f"计算 {mode} 统计量")):
        try:
            with h5py.File(h5_path, "r") as f:
                # 1. 读取动作数据
                actions = f[act_key][()].astype(np.float64)
                # 2. 读取观测/状态数据
                obs_data = f[obs_key][()].astype(np.float64)

                # 初始化维度 (根据读取到的数据动态确定，兼容 14/16/32 维)
                if mean_action is None:
                    mean_action = np.zeros(actions.shape[1], dtype=np.float64)
                    M2_action = np.zeros(actions.shape[1], dtype=np.float64)
                if mean_obs is None:
                    mean_obs = np.zeros(obs_data.shape[1], dtype=np.float64)
                    M2_obs = np.zeros(obs_data.shape[1], dtype=np.float64)

                # 更新统计量
                count_action, mean_action, M2_action = update_online_stats(actions, count_action, mean_action, M2_action)
                count_obs, mean_obs, M2_obs = update_online_stats(obs_data, count_obs, mean_obs, M2_obs)
                
                # 抽样用于可视化
                if i % 10 == 0:
                    sampled_actions.append(actions[::5]) 
        except Exception as e:
            print(f"读取文件 {h5_path} [Key: {act_key}/{obs_key}] 出错: {e}")

    if count_action == 0:
        return None

    # 计算最终 std
    std_action = np.sqrt(M2_action / max(count_action - 1, 1))
    std_obs = np.sqrt(M2_obs / max(count_obs - 1, 1))

    eps = 1e-4
    std_action = np.maximum(std_action, eps)
    std_obs = np.maximum(std_obs, eps)
    
    all_sampled = np.concatenate(sampled_actions, axis=0)
    
    res = {
        "action_mean": mean_action.astype(np.float32),
        "action_std": std_action.astype(np.float32),
        f"{mode}_mean": mean_obs.astype(np.float32),
        f"{mode}_std": std_obs.astype(np.float32),
        "total_count": count_action,
        "mode": mode
    }
    return res, all_sampled


def save_and_viz(stats, sampled_data, save_path):
    # 1. 保存 Pickle (整合 qpos 和 action)
    output_pkl = os.path.join(save_path, "qpos_mean_std_online.pkl")
    with open(output_pkl, "wb") as f:
        pickle.dump(stats, f)
    
    # 2. 可视化 Action 分布
    viz_dir = os.path.join(save_path, "global_distribution_plots")
    os.makedirs(viz_dir, exist_ok=True)
    
    dim_labels = ['L_X', 'L_Y', 'L_Z', 'L_Qx', 'L_Qy', 'L_Qz', 'L_Grip',
                  'R_X', 'R_Y', 'R_Z', 'R_Qx', 'R_Qy', 'R_Qz', 'R_Grip']
    
    mean = stats["action_mean"]
    std = stats["action_std"]

    print(f"正在生成可视化图表至: {viz_dir}")
    for i in range(len(dim_labels)):
        plt.figure(figsize=(10, 5))
        # 原始分布
        plt.subplot(1, 2, 1)
        plt.hist(sampled_data[:, i], bins=100, color='green', alpha=0.7)
        plt.title(f"Original: {dim_labels[i]}")
        
        # 归一化后的分布
        normed = (sampled_data[:, i] - mean[i]) / std[i]
        plt.subplot(1, 2, 2)
        plt.hist(normed, bins=100, color='orange', alpha=0.7)
        plt.title(f"Normalized: {dim_labels[i]}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"dim_{i}_{dim_labels[i]}.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, 
                        default="/home/tione/notebook/workspace/hanyangyu/RoboTwin/policy/embodied_pi0_action/mydata/robotwin_processed_depth")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/tione/notebook/workspace/hanyangyu/RoboTwin/data/global_stats_output")
    parser.add_argument("--mode", type=str, choices=['qpos', 'eep'], default='qpos',
                        help="选择统计关节空间 (qpos) 还是末端位姿空间 (eep)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    result = compute_global_stats_from_processed(args.processed_dir, mode=args.mode)
    
    if result:
        stats, samples = result
        mode_upper = args.mode.upper()
        print("\n" + "="*30)
        print(f"[{mode_upper}] 计算完成！")
        print(f"总计样本数: {stats['total_count']}")
        print(f"Action Mean (前3位): {stats['action_mean'][:3]}")
        print(f"{mode_upper} Mean (前3位): {stats[f'{args.mode}_mean'][:3]}")
        print("="*30)

        save_and_viz(stats, samples, args.output_dir)
        print(f"统计文件已保存为: {args.mode}_mean_std_online.pkl")
    else:
        print("未能计算统计量，请检查路径。")