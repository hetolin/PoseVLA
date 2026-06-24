import os
import time
import threading
import multiprocessing as mp
from functools import partial
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import normalize
import sys
from pathlib import Path

# 导入转换工具 (根据你的路径调整)
sys.path.append(str(Path(__file__).parent.parent))
try:
    import embodied_pi0_action.utils.transform_utils as trans
except ImportError:
    print("Warning: utils.transform_utils not found. Please check your sys.path.")


# ============================================================
# 核心逻辑：合并函数
# ============================================================
def merge_running_stats(s1: normalize.RunningStats, s2: normalize.RunningStats):
    if s2._count == 0: return s1
    if s1._count == 0: return s2
    global_min = np.minimum(s1._min, s2._min)
    global_max = np.maximum(s1._max, s2._max)
    s1._min, s1._max = global_min, global_max
    s1._adjust_histograms()
    s2._min, s2._max = global_min, global_max
    s2._adjust_histograms()
    for i in range(len(s1._histograms)):
        s1._histograms[i] += s2._histograms[i]
    new_count = s1._count + s2._count
    s1._mean = (s1._mean * s1._count + s2._mean * s2._count) / new_count
    s1._mean_of_squares = (s1._mean_of_squares * s1._count + s2._mean_of_squares * s2._count) / new_count
    s1._count = new_count
    return s1


# ============================================================
# Worker 进程逻辑：集成 RT 变换
# ============================================================
def stats_worker(task_info):
    ds_key, cfg, indices_chunk, shared_counter = task_info
    try:
        if ds_key == "agibot":
            from data.ds_raw.agibot import HDF5VLADataset
            dataset = HDF5VLADataset(cfg=cfg, sample_weights=cfg.dataset.agibot.sample_weights,
                                     global_downsample_rate=3)
        else:
            from data.ds_raw.mix import HDF5VLADataset
            dataset = HDF5VLADataset(cfg=cfg, sample_weights=cfg.dataset.xtrainer.sample_weights)
    except Exception as e:
        print(f"Worker failed to init {ds_key}: {e}")
        return ds_key, None

    keys = ["state", "action"]
    local_stats = {key: normalize.RunningStats() for key in keys}
    processed_count = 0

    for i in indices_chunk:
        try:
            # state_only=True 通常只返回 state 和 action，不返回图像，速度快
            batch = dataset.get_item(i, state_only=True)

            if ds_key == "agibot":
                # --- Agibot 逻辑：简单过滤 ---
                if batch["action"].shape[-1] > 20:
                    continue
                state_to_update = np.asarray(batch["state"])
                action_to_update = np.asarray(batch["action"])

            else:
                # --- Xtrainer 逻辑：RT 变换 ---
                # 1. State 转换: 16维 -> 20维 (PosQuat -> RotationMatrix)
                raw_qpos = np.asarray(batch["state"])
                if raw_qpos.ndim == 1: raw_qpos = raw_qpos[None, :]
                state_to_update = trans.convert_PosQuat2PosRotationMatrix_batch(raw_qpos)

                # 2. Action 转换: 相对位姿计算
                # 假设 MixDataset 返回的 action 已经是 (chunk_size, 16) 的序列
                raw_actions = np.asarray(batch["action"])
                # 调用相对位姿转换
                rel_actions = trans.dual_arm_poses_to_relative(raw_actions)
                # 展平为 1D 向量 (chunk_size * 20) 以便 RunningStats 统计
                action_to_update = rel_actions.reshape(-1)

            # 更新统计量
            local_stats["state"].update(state_to_update)
            local_stats["action"].update(action_to_update)

        except Exception:
            pass

        processed_count += 1
        if processed_count >= 20:
            shared_counter.value += processed_count
            processed_count = 0

    shared_counter.value += processed_count
    return ds_key, local_stats


def progress_listener(shared_counter, total_count, exit_event):
    pbar = tqdm(total=total_count, desc="Computing", dynamic_ncols=True)
    last_val = 0
    while not exit_event.is_set():
        curr_val = shared_counter.value
        if curr_val > last_val:
            pbar.update(curr_val - last_val)
            last_val = curr_val
        if curr_val >= total_count: break
        time.sleep(0.5)
    pbar.close()


# ============================================================
# 主程序
# ============================================================
@hydra.main(version_base=None, config_path="../../config", config_name="base")
def test_multi_processing(cfg: DictConfig):
    # 设置总进程数
    num_processes = 10#256

    # 定义数据集及其输出路径
    dataset_configs = [
        # {"key": "agibot", "out": "mydata/agibot_beta_stats"},
        {"key": "xtrainer", "out": "mydata/xtrainer_chuangyuan_773"}
    ]

    all_tasks = []
    total_samples = 0

    # 1. 预扫描并平分进程 (1:1 映射)
    procs_per_ds = num_processes // len(dataset_configs)

    for ds_info in dataset_configs:
        ds_key = ds_info["key"]
        print(f"Pre-scanning {ds_key}...")

        if ds_key == "agibot":
            from data.ds_raw.agibot import HDF5VLADataset
            temp_ds = HDF5VLADataset(cfg=cfg, sample_weights=cfg.dataset.agibot.sample_weights,
                                     global_downsample_rate=3)
        else:
            from data.ds_raw.mix import HDF5VLADataset
            temp_ds = HDF5VLADataset(cfg=cfg, sample_weights=cfg.dataset.xtrainer.sample_weights)

        ds_len = len(temp_ds)
        total_samples += ds_len

        # 严格按照进程数切分，不额外分块
        chunks = np.array_split(np.arange(ds_len), procs_per_ds)
        for c in chunks:
            all_tasks.append((ds_key, cfg, c))
        del temp_ds

    # 2. 运行进程池
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    shared_counter = manager.Value('i', 0)
    exit_event = threading.Event()

    monitor = threading.Thread(target=progress_listener, args=(shared_counter, total_samples, exit_event))
    monitor.start()

    # 这里的 final_tasks 长度正好等于 num_processes
    final_tasks = [(*t, shared_counter) for t in all_tasks]

    print(f"Starting Pool with {num_processes} processes...")
    with ctx.Pool(processes=num_processes) as pool:
        results = pool.map(stats_worker, final_tasks)

    exit_event.set()
    monitor.join(timeout=5)

    # 3. 合并与保存
    print("\nMerging results...")
    grouped = {}
    for ds_key, res in results:
        if res is None: continue
        if ds_key not in grouped:
            grouped[ds_key] = res
        else:
            for k in ["state", "action"]:
                grouped[ds_key][k] = merge_running_stats(grouped[ds_key][k], res[k])

    for ds_info in dataset_configs:
        ds_key = ds_info["key"]
        if ds_key in grouped:
            # 获取统计结果并保存为 JSON
            final_stats = {k: obj.get_statistics() for k, obj in grouped[ds_key].items() if obj._count >= 2}

            # 保存
            normalize.save(ds_info["out"], final_stats)
            print(f"Saved {ds_key} ({grouped[ds_key]['state']._count} samples) to {ds_info['out']}")


if __name__ == '__main__':
    test_multi_processing()