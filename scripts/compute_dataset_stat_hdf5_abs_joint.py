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
# Worker 进程逻辑
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
            batch = dataset.get_item(i, state_only=True)
            # --- 差异化过滤 ---
            if ds_key == "agibot" and batch["action"].shape[-1] > 20:
                continue
            # -----------------
            for key in keys:
                local_stats[key].update(np.asarray(batch[key]))
        except:
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
@hydra.main(version_base=None, config_path="../config", config_name="base")
def test_multi_processing(cfg: DictConfig):
    num_processes = 256
    dataset_configs = [
        # {"key": "agibot", "out": "mydata/agibot_beta_stats"},
        # {"key": "xtrainer", "out": "mydata/xtrainer_mix_stats_pnp5"}
        {"key": "xtrainer", "out": "mydata/xtrainer_mix_stats_pnp773"}
    ]

    all_tasks = []
    total_samples = 0

    # 1. 预扫描并平分进程
    # 每个数据集分配 64 个进程 (总共 128)
    procs_per_ds = num_processes // len(dataset_configs)

    for ds_info in dataset_configs:
        ds_key = ds_info["key"]
        if ds_key == "agibot":
            from data.ds_raw.agibot import HDF5VLADataset
            temp_ds = HDF5VLADataset(cfg=cfg, sample_weights=cfg.dataset.agibot.sample_weights,
                                     global_downsample_rate=3)
        else:
            from data.ds_raw.mix import HDF5VLADataset
            temp_ds = HDF5VLADataset(cfg=cfg, sample_weights=cfg.dataset.xtrainer.sample_weights)

        ds_len = len(temp_ds)
        total_samples += ds_len

        # 每个数据集切成 64 块
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

    # 这里的 final_tasks 长度正好是 128
    final_tasks = [(*t, shared_counter) for t in all_tasks]
    with ctx.Pool(processes=num_processes) as pool:
        results = pool.map(stats_worker, final_tasks)

    exit_event.set()
    monitor.join(timeout=5)

    # 3. 合并与保存
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
            final_stats = {k: obj.get_statistics() for k, obj in grouped[ds_key].items() if obj._count >= 2}
            normalize.save(ds_info["out"], final_stats)
            print(f"Saved {ds_key} to {ds_info['out']}")


if __name__ == '__main__':
    test_multi_processing()