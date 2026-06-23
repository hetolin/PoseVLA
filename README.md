# PoseVLA

PoseVLA is a joint training framework for **Vision-Language-Action (VLA) + 3D object pose / detection**, built on top of **PaliGemma + π0 / π0.5 Action Expert**.

The project supports:

- **VLM training**: learn 3D object detection / 6D pose / scene description via Next-Token Prediction (on Omni3D, Omni6D, BOP, GraspClutter6D, etc.).
- **Action training**: learn robot actions via Flow Matching (on HDF5 / LeRobot-format data such as Agibot, Droid, RDT, xtrainer, InternData-A1, etc.).
- **Co-Training**: VLM and Action data can be jointly optimized within the same training step, with optional Knowledge Insulation.

It is built on 🤗 `transformers` + `accelerate` + `deepspeed` + `hydra`.

---

## 📁 Project Structure

```
PoseVLA/
├── train.py                    # Main training entry (hydra + accelerate + deepspeed)
├── eval_gemini.py              # Evaluation / mAP entry (Omni3D and other 3D tasks)
├── data_factory.py             # Unified VLM / Action DataLoader factory
├── collators.py                # DataCollators (action / detection)
├── mapping_token.py            # Text ↔ 3D scene encoding / decoding utilities
├── graspclutter6dAPI.py        # GraspClutter6D dataset API
│
├── pi0/                        # π0 / π0.5 model implementation
│   ├── configuration_pi0.py    # PI0Config
│   ├── modeling_pi0.py         # PI0Policy (PaliGemma + Action Expert + Flow Matching)
│   ├── paligemma_with_expert.py
│   ├── patch_embed.py
│   ├── convert_jax_model_to_pytorch.py
│   └── _lerobot_compat.py
│
├── data/
│   ├── ds_raw/                 # Raw dataset readers (agibot / droid / rdt / umi / xtrainer / interndata_a1 …)
│   └── ds_train/               # Training Datasets (hdf5 / lerobot / bop / clutter / omni3d / omni6d / agibot)
│
├── config/                     # Hydra configs
│   ├── base.yaml               # Main config entry
│   ├── zero0.json / zero2.json / zero3_offload.json   # DeepSpeed configs
│   ├── dataset/                # Action training dataset configs (hdf5, lerobot)
│   ├── dataset_bop/            # BOP series
│   ├── dataset_clutter/        # GraspClutter6D
│   ├── dataset_det/            # Omni6D
│   ├── dataset_omni3d/         # Omni3D (train/val/test)
│   ├── dataset_lerobot/        # LeRobot grouped configs
│   └── dataset_meta/           # Per-source sample lists (json)
│
├── scripts/
│   ├── launch/                 # Training launch scripts
│   │   ├── start_h20.sh        # Environment bootstrap (HF / wandb / apt / netrc …)
│   │   └── train_h20_multiple.sh   # Multi-node multi-GPU training command
│   ├── agibot/                 # Agibot download scripts
│   ├── interndata_a1/          # InternData-A1 download / extract / config generation
│   ├── compute_dataset_stat_hdf5_abs_joint.py
│   ├── compute_dataset_stat_hdf5_rel_ee.py
│   └── normalize.py
│
├── utils/                      # Common utilities (visualization, logging, transforms)
│   ├── logger.py
│   ├── vis.py
│   ├── transform_utils.py
│   └── image_corrupt.py
│
└── google/paligemma-3b-pt-224/ # Local PaliGemma tokenizer / config
```

---

## 🔧 Environment Setup

### 1. Python / CUDA dependencies

- Python ≥ 3.10
- PyTorch (≥ 2.2 recommended; TF32 / bf16 works best on Ampere / Hopper GPUs)
- Key packages:

  ```bash
  pip install transformers accelerate deepspeed hydra-core omegaconf wandb \
              tqdm matplotlib numpy peft lerobot
  ```

### 2. Pretrained weights

Create a `pretrain/` directory under the project root and place any of the following weights (pick what you need — multiple loading branches are available in [train.py](train.py)):

```
pretrain/
├── paligemma-3b-pt-224/   # Vanilla PaliGemma VLM
├── lerobot_pi0/           # Finetuned π0
└── pi05_base/             # π0.5
```

The tokenizer is already bundled under [google/paligemma-3b-pt-224/](google/paligemma-3b-pt-224/); the config points to it via `model.tokenizer_model_path`.

### 3. Environment variables

See [scripts/launch/start_h20.sh](scripts/launch/start_h20.sh). At minimum:

```bash
export ROOT="/your/home"
export DEV_PATH="${ROOT}/robot_code"
export PYTHONPATH="$PYTHONPATH:${DEV_PATH}"
export HF_HOME=${ROOT}/.cache/huggingface
export HF_LEROBOT_HOME=${HF_HOME}/lerobot
export HYDRA_FULL_ERROR=1
```

W&B auto-login: the script writes `/root/.netrc` — replace `API_KEY` with your own.

---

## 🚀 Launch Training

### Single-node multi-GPU (local debugging)

```bash
# Bootstrap the environment first
bash scripts/launch/start_h20.sh

# Then launch with accelerate
accelerate launch \
  --multi_gpu --num_machines 1 --num_processes 8 \
  --mixed_precision=bf16 \
  --main_process_ip 127.0.0.1 --main_process_port 56789 \
  --machine_rank 0 \
  train.py
```

### Multi-node multi-GPU (H20 / Taiji cluster)

```bash
bash scripts/launch/train_h20_multiple.sh False    # second arg toggles debug mode
```

`train_h20_multiple.sh` reads cluster-injected env vars `RANK / MASTER_ADDR / MASTER_PORT / WORLD_SIZE / GPU_NUM` and starts [train.py](train.py) through `accelerate launch`.

Three common configurations are pre-defined inside the script (the first is enabled by default, others are kept as comments):

1. **Train VLM only**: `co_training.vlm_training=True, action_training=False, data_3d=True`
2. **Train VLM (robot only)**: `data_3d=False`, continue finetuning from a trained VLM checkpoint
3. **Train VLA action only**: `vlm_training=False, action_training=True`

---

## ⚙️ Key Configuration ([config/base.yaml](config/base.yaml))

| Field | Description |
| --- | --- |
| `training.mixed_precision` | `bf16` / `fp16` / `no` |
| `training.batch_size` / `grad_accumulation_steps` | Per-GPU batch size and gradient accumulation |
| `training.max_training_steps` | Total training steps |
| `training.scheduler_warmup_steps` / `decay_steps` / `decay_lr` | LR schedule |
| `training.is_knowledge_insulation` | Enable Knowledge Insulation (decouple VLM and Action) |
| `training.pi05` | Use the π0.5 branch |
| `training.vis_attn` | Visualize attention maps during validation |
| `training.add_extra_token` / `add_image_token` / `add_prior` | Extra-token switches for NTP tasks |
| `training.weighted_sample` | Multi-dataset weighted sampling by `n^0.43` |
| `co_training.vlm_training` / `action_training` | Which data type is included in this run |
| `data_3d` | True → Omni3D / BOP / Clutter branch; False → Agibot + InternData-A1 branch |
| `dataset.action_chunk_size` / `img_history_size` | Action chunk length and history frames |
| `dataset.image_size` | Input image size (default 224) |
| `deepspeed` | Points to `config/zero0/2/3_offload.json` |
| `resume_ckpt` | Resume directory (contains `model/` and `state/training_state.pth`) |

Dataset combinations are aggregated via the `defaults:` section — see the tail of [config/base.yaml](config/base.yaml).

---

## 🧠 Model Loading Branches

[train.py](train.py) provides several weight-composition strategies (kept as comments, enable as needed):

1. **Load directly from a π0 / π0.5 checkpoint**
   ```python
   policy = PI0Policy.from_pretrained(cfg.model.pretrained_model_path, config=pi0_config, strict=False)
   ```
2. **PaliGemma without Action Expert (current default)**
   ```python
   policy = PI0Policy(pi0_config)
   policy.load_pretrained_vlm("pretrain/paligemma-3b-pt-224")
   ```
3. **Use the VLM from π0 and re-initialize the Action Expert**
4. **Train VLM only while reusing the Action Expert from π0**

> By default the training script runs `del policy.model.paligemma_with_expert.gemma_expert.model.embed_tokens / lm_head` to save memory.

---

## 📊 Evaluation

Entry point for 3D detection / pose tasks:

```bash
python eval_gemini.py \
  resume_ckpt=/path/to/exp/49999 \
  data_3d=True
```

`eval_gemini.py` provides:

- `evaluate_sample(...)` — per-sample 3D IoU / rotation / translation errors
- `compute_metrics_summary(...)` — mAP / PR curve aggregation (VOC 11-point & 101-point)

During training, validation automatically invokes these utilities and uploads PR curves, 3D-box visualizations, and text-prediction comparisons to W&B.

---

## 🗂 Data Preparation

Raw readers for each dataset:

- **Agibot**: [data/ds_raw/agibot.py](data/ds_raw/agibot.py), download script [scripts/agibot/download.sh](scripts/agibot/download.sh)
- **InternData-A1**: [data/ds_raw/interndata_a1.py](data/ds_raw/interndata_a1.py), [scripts/interndata_a1/](scripts/interndata_a1/)
- **Droid / RDT / UMI / xtrainer**: see `data/ds_raw/*.py`
- **BOP / GraspClutter6D / Omni3D / Omni6D**: see `data/ds_train/dataset_*.py` and `config/dataset_*/`

Statistics computation:

```bash
python scripts/compute_dataset_stat_hdf5_abs_joint.py
python scripts/compute_dataset_stat_hdf5_rel_ee.py
python scripts/normalize.py
```

The bin-statistics file used by the VLM 3D task is specified by `statistics_path_6d_dataset` (default: `./statistic_all_datasets/all_bins.pkl`).

---

## 💾 Checkpoint Layout

```
ckpt/<exp_name>/<step>/
├── model/                         # Output of save_pretrained() (with tokenizer)
└── state/training_state.pth/      # Output of accelerator.save_state() (DeepSpeed state)
```

`base.yaml` is also saved to `ckpt_save_dir/base.yaml`, and is automatically merged when resuming:

```bash
python train.py resume_ckpt=/path/to/exp/29999
```

---

## 🛠 Common Research Switches

| Goal | How |
| --- | --- |
| Train VLM only (NTP) | `co_training.vlm_training=True co_training.action_training=False` |
| Train Action only (Flow Matching) | `co_training.vlm_training=False co_training.action_training=True` |
| Joint training + Knowledge Insulation | Enable both, plus `training.is_knowledge_insulation=True` |
| Switch 3D data ↔ robot data | `data_3d=True/False` |
| Switch HDF5 ↔ LeRobot | `defaults.dataset: hdf5 / lerobot` |
| Enable weighted sampling | `training.weighted_sample=True` |
| Enable attention visualization | `training.vis_attn=True` |
| Use LoRA | `training.use_lora=True`, plus configure `lora.*`