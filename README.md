<div align="center">

# PoseVLA: Universal Pose Pretraining for Generalizable Vision-Language-Action Policies (RSS2026)

A unified framework that co-trains a **Vision-Language-Action (VLA)** policy with **3D object detection / 6D pose estimation**, built on top of **PaliGemma**. 🤗

[//]: # (Purely HuggingFace + Accelerate + DeepSpeed + Hydra based — concise code, multi-node ready, easy to extend.)

[![arXiv](https://img.shields.io/badge/arXiv-2602.19710-b31b1b.svg)](https://arxiv.org/abs/2602.19710)
[![Project Page](https://img.shields.io/badge/Project_Page-PoseVLA-2ea44f.svg)](https://hetolin.github.io/PoseVLA/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/hetolin/PoseVLA)

[\[🚀 Quick Start\]](#-quick-start) [\[🌟 Pre-train\]](#-pre-train-from-scratch) [\[🌟 Fine-tune\]](#-fine-tune--resume) [\[🎄 Custom Dataset\]](#-use-custom-datasets) [\[📊 Evaluation\]](#-evaluation) [\[🐛 Troubleshooting\]](#-troubleshooting)

</div>

---

## News 🚀🚀🚀
- `2026/06`: Initial release of **PoseVLA**: PaliGemma + π0 / π0.5 Action Expert, joint VLM + Action training with Knowledge Insulation, supports Omni3D / Omni6D / BOP / GraspClutter6D for 3D tasks and Agibot / Droid / RDT / UMI / xtrainer / InternData-A1 for robot actions.

## 📖 Documents

The project supports three orthogonal training modes that can be freely combined:

- **VLM training** — learn 3D object detection / 6D pose / scene description via **Next-Token Prediction (NTP)** on Omni3D, Omni6D, BOP, GraspClutter6D, …
- **Action training** — learn robot actions via **Flow Matching** on HDF5 / LeRobot-format data (Agibot, Droid, RDT, UMI, xtrainer, InternData-A1, …).
- **Co-Training** — VLM and Action data are interleaved within the same optimization step, with optional **Knowledge Insulation** to decouple their gradients.

### 📁 Project Structure

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

### 🚀 Quick Start

#### 1. Clone

```bash
git clone git@github.com:hetolin/PoseVLA.git
cd PoseVLA
```

#### 2. Conda env (installation from scratch)

- Python 3.10.12
- PyTorch 2.7.0 + CUDA 12.6 (TF32 / bf16 works best on Ampere / Hopper GPUs)

```bash
conda create -n vla python==3.10.12
conda activate vla

# Install PyTorch first — it is intentionally NOT pinned in requirements.txt,
# because the correct wheel depends on your local CUDA version.
# CUDA 12.6:
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# NOTE: keep the `lerobot` line commented out in requirements.txt — we install it manually below.
pip install -r requirements.txt

# Install lerobot at the pinned commit.
# NOTE: at that commit lerobot still declares the old `pyav` package name,
# which is now renamed to `av` on PyPI. We install `av` first, then install
# lerobot with `--no-deps` to bypass the stale `pyav` requirement.
pip install av
pip install --no-deps \
  "lerobot @ git+https://github.com/huggingface/lerobot@638d411cd3acf32c28d8c2120f3c41bda8bb15d4"
```

#### 3. Pretrained weights

Create a `pretrain/` directory under the project root and place any of the following weights (pick what you need — multiple loading branches are available in [train.py](train.py)):

```
pretrain/
├── paligemma-3b-pt-224/   # Vanilla PaliGemma VLM
├── lerobot_pi0/           # Finetuned π0
└── pi05_base/             # π0.5
```

The tokenizer is already bundled under [google/paligemma-3b-pt-224/](google/paligemma-3b-pt-224/); the config points to it via `model.tokenizer_model_path`.

#### 4. Environment variables

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

### 🌟 Pre-train from Scratch

PoseVLA is designed to be pre-trained on a mixture of **3D understanding data** (Omni3D / Omni6D / BOP / GraspClutter6D) and **robot action data** (Agibot / Droid / RDT / UMI / xtrainer / InternData-A1).

#### Single-node multi-GPU (local debugging)

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

#### Multi-node multi-GPU (H20 cluster)

```bash
bash scripts/launch/train_h20_multiple.sh False    # second arg toggles debug mode
```

`train_h20_multiple.sh` reads cluster-injected env vars `RANK / MASTER_ADDR / MASTER_PORT / WORLD_SIZE / GPU_NUM` and starts [train.py](train.py) through `accelerate launch`.

Three common configurations are pre-defined inside the script (the first is enabled by default, others are kept as comments):

1. **Train VLM only**: `co_training.vlm_training=True, action_training=False, data_3d=True`
2. **Train VLM (robot only)**: `data_3d=False`, continue finetuning from a trained VLM checkpoint
3. **Train VLA action only**: `vlm_training=False, action_training=True`

---

### 🌟 Fine-tune / Resume

`base.yaml` is automatically saved to `ckpt_save_dir/base.yaml` at each checkpoint and is merged back in on resume:

```bash
python train.py resume_ckpt=/path/to/exp/29999
```

The checkpoint layout is:

```
ckpt/<exp_name>/<step>/
├── model/                         # Output of save_pretrained() (with tokenizer)
└── state/training_state.pth/      # Output of accelerator.save_state() (DeepSpeed state)
```

LoRA fine-tuning is supported via `training.use_lora=True` together with the `lora.*` block in [config/base.yaml](config/base.yaml).

---

### ⚙️ Key Configuration ([config/base.yaml](config/base.yaml))

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

#### 🛠 Common Research Switches

| Goal | How |
| --- | --- |
| Train VLM only (NTP) | `co_training.vlm_training=True co_training.action_training=False` |
| Train Action only (Flow Matching) | `co_training.vlm_training=False co_training.action_training=True` |
| Joint training + Knowledge Insulation | Enable both, plus `training.is_knowledge_insulation=True` |
| Switch 3D data ↔ robot data | `data_3d=True/False` |
| Switch HDF5 ↔ LeRobot | `defaults.dataset: hdf5 / lerobot` |
| Enable weighted sampling | `training.weighted_sample=True` |
| Enable attention visualization | `training.vis_attn=True` |
| Use LoRA | `training.use_lora=True`, plus configure `lora.*` |

---

### 🧠 Model Loading Branches

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

### 📊 Evaluation

Entry point for 3D detection / pose tasks:

```bash
python eval_gemini.py
```

> ⚠️ **Note**: [eval_gemini.py](eval_gemini.py) currently hard-codes the checkpoint path and the
> output directory inside its `main()` function (it does **not** read `cfg.resume_ckpt`).
> Open the file and edit these two lines before running:
>
> ```python
> # near the top of main()
> ckpt_path = "ckpt/<your_exp_name>/<step>/model"
>
> # near the end of main()
> save_dir = "./eval_results_<your_tag>"
> ```
>
> The script always loads the `omni3d_test` split via `Omni3DConsumerDataset`
> and runs `policy.forward_evaluate_ntp(...)`; switching to other benchmarks
> requires code changes.

`eval_gemini.py` provides:

- `evaluate_sample(...)` — per-sample 3D IoU / rotation / translation errors
- `compute_metrics_summary(...)` — mAP / PR curve aggregation (VOC 11-point & 101-point)

During training, validation automatically invokes these utilities and uploads PR curves, 3D-box visualizations, and text-prediction comparisons to W&B.

---

### 🗂 Data Preparation

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

### 🎄 Use Custom Datasets

To plug a new dataset into PoseVLA:

- **For robot action data** (HDF5 / LeRobot style):
  - Add a raw reader under [data/ds_raw/](data/ds_raw/) following e.g. [data/ds_raw/agibot.py](data/ds_raw/agibot.py).
  - Register it in the corresponding training dataset wrapper under [data/ds_train/](data/ds_train/).
  - Add a Hydra config under `config/dataset/` (or `config/dataset_lerobot/`) and reference it from `defaults:` in [config/base.yaml](config/base.yaml).
  - Drop a sample-list JSON into `config/dataset_meta/` if your reader needs one.

- **For 3D understanding data** (detection / pose):
  - Add the dataset class under [data/ds_train/](data/ds_train/) (mimic `dataset_omni3d.py` / `dataset_bop.py` / `dataset_clutter.py` / `dataset_det.py`).
  - Add token mapping logic in [mapping_token.py](mapping_token.py) if a new label format is introduced.
  - Add a Hydra config under `config/dataset_omni3d/` / `config/dataset_bop/` / `config/dataset_clutter/` / `config/dataset_det/`.

Remember to re-run the statistics scripts under `scripts/` so that the normalization stats and bin definitions cover your new data.

---

## 🐛 Troubleshooting

### 1. `draccus.utils.ParsingError` when resuming from a checkpoint

Symptom:

```
draccus.utils.ParsingError: Expected a dict with a 'type' key for
<class 'lerobot.configs.policies.PreTrainedConfig'>,
got {'n_obs_steps': 1, 'normalization_mapping': {'VIS' ...}}
```

Root cause: the `draccus` version used when **saving** the checkpoint differs from the one used when **loading** it, so the serialized `PreTrainedConfig` is missing the `type` discriminator key that newer `draccus` expects.

**Solution 1 (recommended)** — pin `draccus` to the version this repo is tested with, then re-save the checkpoint:

```bash
pip install draccus==0.10.0
# retrain (or resume + immediately save_state) so the new ckpt is serialized correctly
```

**Solution 2** — bypass the discriminator by passing `config=pi0_config` explicitly when reloading. In [train.py](train.py), change the resume call to:

```python
policy = PI0Policy.from_pretrained(
    os.path.join(cfg.resume_ckpt, "model"),
    config=pi0_config,
    local_files_only=True,
)
```

## Robotwin Post-Train / Fine-Tune

After training a PoseVLM checkpoint, use `train_posttrain.py` to post-train the
policy on a specific robot dataset. For Robotwin, the dataset is HDF5 format and
uses `eep` actions.

1. Set the project root used by Hydra:

```bash
export DEV_PATH=/home/tione/notebook/home/henryhyyu/RoboTwin/policy
```

2. Check `config/base_postrain.yaml`:

- `defaults.dataset: robotwin`
- `model.pretrained_model_path`: PoseVLM checkpoint to fine-tune from
- `model.action_expert_path`: pretrained pi0 action expert
- `ckpt_save_dir`: output directory, normally `${dev_dir}/ckpt/${exp_name}`

3. Check `config/dataset/robotwin.yaml`:

- `type: hdf5`
- `action_type: "eep"`
- `hdf5_dir`: Robotwin HDF5 dataset root
- `mean_std_path`: Robotwin EEP normalization file
- `dataset_list`: Robotwin subsets and sampling weights

4. Before training, run the dataset visualization smoke test:

```bash
python dataset_wan.py
```

This writes:

- `dataset_wan_images.png`
- `dataset_wan_joints.png`

Use these plots to verify the RGB views, instruction text, and EEP action curves
look stable before launching training.

5. Launch post-training:

```bash
python train_posttrain.py
```

For multi-GPU training, use the existing launch script:

```bash
bash train.sh
```

## Robotwin Post-Train / Fine-Tune

After training a PoseVLM checkpoint, use `train_posttrain.py` to post-train the
policy on a specific robot dataset. For Robotwin, the dataset is HDF5 format and
uses `eep` actions.

1. Set the project root used by Hydra:

```bash
export DEV_PATH=/home/tione/notebook/home/henryhyyu/RoboTwin/policy
```

2. Check `config/base_postrain.yaml`:

- `defaults.dataset: robotwin`
- `model.pretrained_model_path`: PoseVLM checkpoint to fine-tune from
- `model.action_expert_path`: pretrained pi0 action expert
- `ckpt_save_dir`: output directory, normally `${dev_dir}/ckpt/${exp_name}`

3. Check `config/dataset/robotwin.yaml`:

- `type: hdf5`
- `action_type: "eep"`
- `hdf5_dir`: Robotwin HDF5 dataset root
- `mean_std_path`: Robotwin EEP normalization file
- `dataset_list`: Robotwin subsets and sampling weights

4. Before training, run the dataset visualization smoke test:

```bash
python dataset_wan.py
```

This writes:

- `dataset_wan_images.png`
- `dataset_wan_joints.png`

Use these plots to verify the RGB views, instruction text, and EEP action curves
look stable before launching training.

5. Launch post-training:

```bash
python train_posttrain.py
```

For multi-GPU training, use the existing launch script:

```bash
bash train.sh
```

---

## TODO List

- [x] Release the training / co-training code for PoseVLA.
- [x] Release the 3D evaluation entry (`eval_gemini.py`) with mAP / PR-curve aggregation.
- [x] Release support for both **π0** and **π0.5** Action Experts in a single codebase (switchable via `training.pi05`).
- [ ] Release pretrained PoseVLA checkpoints.

## 🙋 FAQs

If you encounter any issues, feel free to open an issue on GitHub or reach out through discussions. Feedback and contributions are very welcome! 🚀

## Acknowledgement

PoseVLA is built with reference to the following projects:
[lerobot](https://github.com/huggingface/lerobot),
[Transformers](https://github.com/huggingface/transformers),
[Google PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224),
[π0 / OpenPI](https://github.com/Physical-Intelligence/openpi),
[Omni3D](https://github.com/facebookresearch/omni3d),
[BOP Toolkit](https://github.com/thodan/bop_toolkit),
and [GraspClutter6D](https://github.com/SeungBack/GraspClutter6D).
Thanks for their awesome work.