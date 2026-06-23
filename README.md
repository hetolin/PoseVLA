# PoseVLA

PoseVLA 是一个以 **PaliGemma + π0/π0.5 Action Expert** 为骨干的 **视觉-语言-动作 (VLA) + 3D 物体位姿/检测** 联合训练框架。

项目同时支持：

- **VLM 训练**：以 Next-Token Prediction 的方式学习 3D 物体检测 / 6D 位姿 / 场景描述（基于 Omni3D、Omni6D、BOP、GraspClutter6D 等数据集）。
- **Action 训练**：以 Flow Matching 的方式学习机器人动作（基于 HDF5 / LeRobot 格式的 Agibot、Droid、RDT、xtrainer、InternData-A1 等数据）。
- **Co-Training**：VLM 与 Action 数据可在同一 step 内联合训练，支持 Knowledge Insulation。

底层基于 🤗 `transformers` + `accelerate` + `deepspeed` + `hydra`。

---

## 📁 目录结构

```
PoseVLA/
├── train.py                    # 训练主入口（hydra + accelerate + deepspeed）
├── eval_gemini.py              # 评测 / mAP 指标计算入口（Omni3D 等 3D 任务）
├── data_factory.py             # VLM / Action DataLoader 工厂（统一构建逻辑）
├── collators.py                # DataCollator（动作 / 检测两类）
├── mapping_token.py            # 文本 ↔ 3D 场景互编解码工具
├── graspclutter6dAPI.py        # GraspClutter6D 数据集 API
│
├── pi0/                        # π0 / π0.5 模型实现
│   ├── configuration_pi0.py    # PI0Config
│   ├── modeling_pi0.py         # PI0Policy（PaliGemma + Action Expert + Flow Matching）
│   ├── paligemma_with_expert.py
│   ├── patch_embed.py
│   ├── convert_jax_model_to_pytorch.py
│   └── _lerobot_compat.py
│
├── data/
│   ├── ds_raw/                 # 原始数据集读取（agibot / droid / rdt / umi / xtrainer / interndata_a1 …）
│   └── ds_train/               # 训练用 Dataset（hdf5 / lerobot / bop / clutter / omni3d / omni6d / agibot）
│
├── config/                     # Hydra 配置
│   ├── base.yaml               # 主配置入口
│   ├── zero0.json / zero2.json / zero3_offload.json   # DeepSpeed 配置
│   ├── dataset/                # action 训练数据集配置（hdf5、lerobot）
│   ├── dataset_bop/            # BOP 系列
│   ├── dataset_clutter/        # GraspClutter6D
│   ├── dataset_det/            # Omni6D
│   ├── dataset_omni3d/         # Omni3D（train/val/test）
│   ├── dataset_lerobot/        # LeRobot 分组配置
│   └── dataset_meta/           # 各数据源的样本列表 (json)
│
├── scripts/
│   ├── launch/                 # 训练启动脚本
│   │   ├── start_h20.sh        # 环境初始化（HF / wandb / apt / netrc 等）
│   │   └── train_h20_multiple.sh   # 多机多卡训练命令
│   ├── agibot/                 # Agibot 下载脚本
│   ├── interndata_a1/          # InternData-A1 下载 / 解压 / 配置生成
│   ├── compute_dataset_stat_hdf5_abs_joint.py
│   ├── compute_dataset_stat_hdf5_rel_ee.py
│   └── normalize.py
│
├── utils/                      # 通用工具（可视化、日志、变换）
│   ├── logger.py
│   ├── vis.py
│   ├── transform_utils.py
│   └── image_corrupt.py
│
└── google/paligemma-3b-pt-224/ # 本地 PaliGemma tokenizer / 配置
```

---

## 🔧 环境准备

### 1. Python / CUDA 依赖

- Python ≥ 3.10
- PyTorch（推荐 ≥ 2.2，开启 TF32 / bf16 在 Ampere / Hopper GPU 上效果最佳）
- 关键依赖：

  ```bash
  pip install transformers accelerate deepspeed hydra-core omegaconf wandb \
              tqdm matplotlib numpy peft lerobot
  ```

### 2. 预训练权重

在项目根目录下新建 `pretrain/`，并放入以下任一权重（按需选择，对应 [train.py](train.py) 中的多种加载分支）：

```
pretrain/
├── paligemma-3b-pt-224/   # 原生 PaliGemma VLM
├── lerobot_pi0/           # 已微调的 π0
└── pi05_base/             # π0.5
```

Tokenizer 已内置在 [google/paligemma-3b-pt-224/](google/paligemma-3b-pt-224/)，配置中通过 `model.tokenizer_model_path` 指向即可。

### 3. 环境变量

参考 [scripts/launch/start_h20.sh](scripts/launch/start_h20.sh)，需要设置：

```bash
export ROOT="/your/home"
export DEV_PATH="${ROOT}/robot_code"
export PYTHONPATH="$PYTHONPATH:${DEV_PATH}"
export HF_HOME=${ROOT}/.cache/huggingface
export HF_LEROBOT_HOME=${HF_HOME}/lerobot
export HYDRA_FULL_ERROR=1
```

W&B 自动登录：脚本会写入 `/root/.netrc`，把里面的 `API_KEY` 换成自己的。

---

## 🚀 启动训练

### 单机多卡（本地调试）

```bash
# 先初始化环境
bash scripts/launch/start_h20.sh

# 然后用 accelerate 启动
accelerate launch \
  --multi_gpu --num_machines 1 --num_processes 8 \
  --mixed_precision=bf16 \
  --main_process_ip 127.0.0.1 --main_process_port 56789 \
  --machine_rank 0 \
  train.py
```

### 多机多卡（H20 / Taiji 集群）

```bash
bash scripts/launch/train_h20_multiple.sh False    # 第二个参数控制 debug
```

`train_h20_multiple.sh` 会读取集群注入的环境变量 `RANK / MASTER_ADDR / MASTER_PORT / WORLD_SIZE / GPU_NUM`，并通过 `accelerate launch` 启动 [train.py](train.py)。

脚本中已内置三套常用配置（默认启用第一套，其余被注释保留）：

1. **train VLM only**：`co_training.vlm_training=True, action_training=False, data_3d=True`
2. **train VLM (robot only)**：`data_3d=False`，加载已训好的 VLM checkpoint 继续微调
3. **train VLA action only**：`vlm_training=False, action_training=True`

---

## ⚙️ 关键配置（[config/base.yaml](config/base.yaml)）

| 配置项 | 说明 |
| --- | --- |
| `training.mixed_precision` | `bf16` / `fp16` / `no` |
| `training.batch_size` / `grad_accumulation_steps` | 单卡 batch 与梯度累积步数 |
| `training.max_training_steps` | 总训练步数 |
| `training.scheduler_warmup_steps` / `decay_steps` / `decay_lr` | 学习率调度 |
| `training.is_knowledge_insulation` | 是否使用 Knowledge Insulation（VLM 与 Action 解耦） |
| `training.pi05` | 是否使用 π0.5 分支 |
| `training.vis_attn` | 验证阶段是否可视化注意力图 |
| `training.add_extra_token` / `add_image_token` / `add_prior` | NTP 任务的额外 token 开关 |
| `training.weighted_sample` | 多数据集按 `n^0.43` 加权采样 |
| `co_training.vlm_training` / `action_training` | 控制本次训练包含哪类数据 |
| `data_3d` | True → 走 Omni3D / BOP / Clutter 分支；False → 走 Agibot + InternData-A1 分支 |
| `dataset.action_chunk_size` / `img_history_size` | 动作 chunk 长度与历史帧数 |
| `dataset.image_size` | 输入图像尺寸（默认 224） |
| `deepspeed` | 指向 `config/zero0/2/3_offload.json` |
| `resume_ckpt` | 断点续训目录（包含 `model/` 与 `state/training_state.pth`） |

数据集组合通过 `defaults:` 节聚合，详见 [config/base.yaml](config/base.yaml) 末尾。

---

## 🧠 模型加载分支

[train.py](train.py) 提供多种权重组合方式（注释保留，按需启用）：

1. **直接从 π0 / π0.5 checkpoint 加载**
   ```python
   policy = PI0Policy.from_pretrained(cfg.model.pretrained_model_path, config=pi0_config, strict=False)
   ```
2. **PaliGemma + 不使用 Action Expert（当前默认）**
   ```python
   policy = PI0Policy(pi0_config)
   policy.load_pretrained_vlm("pretrain/paligemma-3b-pt-224")
   ```
3. **VLM 来自 π0，再把 Action Expert 替换为新初始化**
4. **只训 VLM，复用 π0 的 Action Expert**

> 训练前默认 `del policy.model.paligemma_with_expert.gemma_expert.model.embed_tokens / lm_head` 以减少显存。

---

## 📊 评测

3D 检测 / 位姿任务评测入口：

```bash
python eval_gemini.py \
  resume_ckpt=/path/to/exp/49999 \
  data_3d=True
```

`eval_gemini.py` 提供：

- `evaluate_sample(...)` 单样本 3D IoU / 旋转 / 平移误差
- `compute_metrics_summary(...)` mAP / PR 曲线汇总（VOC 11-point & 101-point）

训练过程中也会在 validation 时自动调用并通过 W&B 上传 PR 曲线、3D 框可视化与文本预测对比。

---

## 🗂 数据准备

各数据集对应的 raw 读取实现：

- **Agibot**：[data/ds_raw/agibot.py](data/ds_raw/agibot.py)，下载脚本 [scripts/agibot/download.sh](scripts/agibot/download.sh)
- **InternData-A1**：[data/ds_raw/interndata_a1.py](data/ds_raw/interndata_a1.py)，[scripts/interndata_a1/](scripts/interndata_a1/)
- **Droid / RDT / UMI / xtrainer**：见 `data/ds_raw/*.py`
- **BOP / GraspClutter6D / Omni3D / Omni6D**：见 `data/ds_train/dataset_*.py` 与 `config/dataset_*/`

统计量计算：

```bash
python scripts/compute_dataset_stat_hdf5_abs_joint.py
python scripts/compute_dataset_stat_hdf5_rel_ee.py
python scripts/normalize.py
```

VLM 3D 任务使用的 bin 统计文件由 `statistics_path_6d_dataset` 指定（默认 `./statistic_all_datasets/all_bins.pkl`）。

---

## 💾 Checkpoint 结构

```
ckpt/<exp_name>/<step>/
├── model/                         # save_pretrained() 输出（含 tokenizer）
└── state/training_state.pth/      # accelerator.save_state() 输出（DeepSpeed 状态）
```

`base.yaml` 同时被保存到 `ckpt_save_dir/base.yaml`，断点续训时会自动合并：

```bash
python train.py resume_ckpt=/path/to/exp/29999
```

---

## 🛠 常见研究开关速查

| 想做什么 | 怎么改 |
| --- | --- |
| 只训 VLM（NTP） | `co_training.vlm_training=True co_training.action_training=False` |
| 只训 Action（Flow Matching） | `co_training.vlm_training=False co_training.action_training=True` |
| 联合训练 + Knowledge Insulation | 同时开启两者，并 `training.is_knowledge_insulation=True` |
| 切换 3D 数据 ↔ 机器人数据 | `data_3d=True/False` |
| 切换 HDF5 ↔ LeRobot | `defaults.dataset: hdf5 / lerobot` |
| 启用 weighted sampling | `training.weighted_sample=True` |
| 启用 attention 可视化 | `training.vis_attn=True` |
| 使用 LoRA | `training.use_lora=True`，并配置 `lora.*` |

---

## 📜 License

仅供研究用途，请确认所使用的数据集与预训练权重各自的许可协议。