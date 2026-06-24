<div align="center">

# PoseVLA: Universal Pose Pretraining for Generalizable Vision-Language-Action Policies (RSS2026)

A unified framework that co-trains a **Vision-Language-Action (VLA)** policy with **3D object detection / 6D pose estimation**, built on top of **PaliGemma**. 🤗

[//]: # (Purely HuggingFace + Accelerate + DeepSpeed + Hydra based — concise code, multi-node ready, easy to extend.)

[![arXiv](https://img.shields.io/badge/arXiv-2602.19710-b31b1b.svg)](https://arxiv.org/abs/2602.19710)
[![Project Page](https://img.shields.io/badge/Project_Page-PoseVLA-2ea44f.svg)](https://hetolin.github.io/PoseVLA/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/hetolin/PoseVLA)

[\[🚀 Quick Start\]](#-quick-start) [\[🌟 Pre-train\]](docs/PRETRAIN.md) [\[🤖 Post-train (Robotwin)\]](docs/POSTTRAIN.md) [\[🕹 RoboTwin Eval\]](robotwin/PoseVLA/README.md) [\[🐛 Troubleshooting\]](docs/PRETRAIN.md#-troubleshooting)

</div>

---

## News 🚀🚀🚀
- `2026/06`: Initial release of **PoseVLA**: PaliGemma + π0 / π0.5 Action Expert, joint VLM + Action training with Knowledge Insulation, supports Omni3D / Omni6D / BOP / GraspClutter6D for 3D tasks and Agibot / Droid / RDT / UMI / xtrainer / InternData-A1 for robot actions.

---

## 📖 Documents

PoseVLA is split into two training stages, each with its own document:

| Stage | Entry script | Config | Doc |
| --- | --- | --- | --- |
| **Pre-train** (joint VLM + Action on large-scale data) | [train_pretrain.py](train_pretrain.py) | [config/base.yaml](config/base.yaml) | [docs/PRETRAIN.md](docs/PRETRAIN.md) |
| **Post-train / Fine-tune** (Robotwin and other downstream robots) | [train_posttrain.py](train_posttrain.py) | [config/base_postrain.yaml](config/base_postrain.yaml) | [docs/POSTTRAIN.md](docs/POSTTRAIN.md) |
| **RoboTwin simulation eval** | [robotwin/PoseVLA/eval_policy.py](robotwin/PoseVLA/eval_policy.py) | [robotwin/PoseVLA/deploy_policy.yml](robotwin/PoseVLA/deploy_policy.yml) | [robotwin/PoseVLA/README.md](robotwin/PoseVLA/README.md) |

The project supports three orthogonal training modes that can be freely combined:

- **VLM training** — learn 3D object detection / 6D pose / scene description via **Next-Token Prediction (NTP)** on Omni3D, Omni6D, BOP, GraspClutter6D, …
- **Action training** — learn robot actions via **Flow Matching** on HDF5 / LeRobot-format data (Agibot, Droid, RDT, UMI, xtrainer, InternData-A1, Robotwin, …).
- **Co-Training** — VLM and Action data are interleaved within the same optimization step, with optional **Knowledge Insulation** to decouple their gradients.

### 📁 Project Structure

```
PoseVLA/
├── train_pretrain.py           # Pre-train entry (hydra + accelerate + deepspeed)
├── train_posttrain.py          # Post-train / fine-tune entry (Robotwin etc.)
├── eval_detection.py           # Evaluation / mAP entry (Omni3D and other 3D tasks)
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
│   ├── factory.py              # Unified VLM / Action DataLoader factory
│   ├── collators.py            # DataCollators (action / detection)
│   ├── ds_raw/                 # Raw dataset readers (agibot / droid / rdt / umi / xtrainer / interndata_a1 / robotwin …)
│   └── ds_train/               # Training Datasets (hdf5 / lerobot / bop / clutter / omni3d / omni6d / agibot)
│       └── graspclutter6dAPI.py    # GraspClutter6D dataset API (used by dataset_clutter)
│
├── utils/                      # Shared utilities
│   ├── process_data_all.py     # Robotwin raw → HDF5 conversion
│   ├── mapping_token.py        # Text ↔ 3D scene encoding / decoding utilities
│   ├── transform_utils.py      # SE(3) / pose math helpers
│   ├── image_corrupt.py        # Image augmentation
│   ├── vis.py                  # Visualization helpers
│   └── logger.py               # WandB / training-state logging
│
├── config/                     # Hydra configs
│   ├── base.yaml               # Pre-train main config
│   ├── base_postrain.yaml      # Post-train (Robotwin) main config
│   ├── zero0.json / zero2.json / zero3_offload.json   # DeepSpeed configs
│   ├── dataset/                # Action training dataset configs (hdf5, lerobot, robotwin)
│   ├── dataset_bop/            # BOP series
│   ├── dataset_clutter/        # GraspClutter6D
│   ├── dataset_det/            # Omni6D
│   ├── dataset_omni3d/         # Omni3D (train/val/test)
│   ├── dataset_lerobot/        # LeRobot grouped configs
│   └── dataset_meta/           # Per-source sample lists (json)
│
├── scripts/
│   ├── launch/                 # Training launch scripts
│   │   ├── pretrain.sh         # Multi-GPU launch for pre-training (train_pretrain.py)
│   │   └── posttrain.sh        # Multi-GPU launch for post-training (train_posttrain.py)
│   ├── download/               # Dataset download / extract / yaml-gen scripts
│   │   ├── agibot.sh
│   │   ├── interndata_a1.sh
│   │   ├── interndata_a1_unzip.sh
│   │   └── interndata_a1_generate_yaml.py
│   └── stats/                  # Dataset normalization stats
│       ├── compute_dataset_stat_hdf5_abs_joint.py
│       ├── compute_dataset_stat_hdf5_rel_ee.py
│       ├── norm_robotwin.py    # Robotwin EEP / qpos normalization stats
│       └── normalize.py
│
├── docs/                       # Stage-level documents
│   ├── PRETRAIN.md             # Pre-train guide
│   ├── POSTTRAIN.md            # Post-train / fine-tune guide (Robotwin)
│   └── ROBOTWIN_DATA.md        # Robotwin raw → HDF5 conversion guide
│
├── robotwin/PoseVLA/           # RoboTwin simulation deploy + eval
└── google/paligemma-3b-pt-224/ # Local PaliGemma tokenizer / config
```

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone git@github.com:hetolin/PoseVLA.git
cd PoseVLA
```

### 2. Conda env (installation from scratch)

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

### 3. Pretrained weights

Create a `pretrain/` directory under the project root and place any of the following weights (pick what you need — multiple loading branches are available in [train_pretrain.py](train_pretrain.py)):

```
pretrain/
├── paligemma-3b-pt-224/   # Vanilla PaliGemma VLM
├── lerobot_pi0/           # Finetuned π0
└── pi05_base/             # π0.5
```

The tokenizer is already bundled under [google/paligemma-3b-pt-224/](google/paligemma-3b-pt-224/); the config points to it via `model.tokenizer_model_path`.

### 4. Environment variables

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

### 5. Next step

- For **pre-training** the PoseVLA backbone on large-scale 3D + action data, follow [docs/PRETRAIN.md](docs/PRETRAIN.md).
- For **post-training / fine-tuning** on RoboTwin (or your own robot), follow [docs/POSTTRAIN.md](docs/POSTTRAIN.md).
- For **simulation evaluation** on RoboTwin, follow [robotwin/PoseVLA/README.md](robotwin/PoseVLA/README.md).

---

## TODO List

- [x] Release the training / co-training code for PoseVLA.
- [x] Release the 3D evaluation entry (`eval_detection.py`) with mAP / PR-curve aggregation.
- [x] Release support for both **π0** and **π0.5** Action Experts in a single codebase (switchable via `training.pi05`).
- [x] Release the Robotwin post-training entry (`train_posttrain.py`) and RoboTwin deployment scripts.
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