# 🤖 PoseVLA — Post-train / Fine-tune (Robotwin)

This document covers **post-training / downstream fine-tuning** of a pre-trained PoseVLA checkpoint on a specific robot dataset. The reference setup here is **RoboTwin 2.0** (HDF5, EEP actions), but the same pipeline can be reused for other robots by swapping the dataset config.

- entry script: [`train_posttrain.py`](../train_posttrain.py)
- main config: [`config/base_postrain.yaml`](../config/base_postrain.yaml)
- dataset config: [`config/dataset/robotwin.yaml`](../config/dataset/robotwin.yaml)
- dataloader / smoke test: [`dataset_wan.py`](../dataset_wan.py)
- multi-GPU launch: [`train.sh`](../train.sh)
- data conversion: [`utils/README.md`](../utils/README.md)
- simulation evaluation: [`robotwin/PoseVLA/README.md`](../robotwin/PoseVLA/README.md)

For **pre-training** the PoseVLA backbone, see [PRETRAIN.md](PRETRAIN.md).

> Make sure you have followed the env setup in [README.md → Quick Start](../README.md#-quick-start) before running anything below.

---

## 1. Set the project root used by Hydra

```bash
export DEV_PATH=/home/tione/notebook/home/henryhyyu/RoboTwin/policy
```

`base_postrain.yaml` resolves `dev_dir: ${oc.env:DEV_PATH}/PoseVLA` and writes checkpoints to `${dev_dir}/ckpt/${exp_name}`.

---

## 2. Configure [`config/base_postrain.yaml`](../config/base_postrain.yaml)

Key fields to check:

| Field | Description |
| --- | --- |
| `defaults.dataset` | `robotwin` (loads [`config/dataset/robotwin.yaml`](../config/dataset/robotwin.yaml)) |
| `model.pretrained_model_path` | PoseVLM checkpoint to fine-tune from (output of pre-training) |
| `model.action_expert_path` | Pretrained π0 action expert (loaded via `load_action_expert(...)`) |
| `model.tokenizer_model_path` | Path to the bundled PaliGemma tokenizer |
| `ckpt_save_dir` | Output directory, by default `${dev_dir}/ckpt/${exp_name}` |
| `co_training.action_training` / `vlm_training` | Robotwin post-training is **action-only** (`vlm_training=False`, `action_training=True`) |
| `data_3d` | `False` — Robotwin has no 3D NTP labels |
| `training.pi05` | Switch between π0 (`False`) and π0.5 (`True`) action expert |
| `training.batch_size` / `grad_accumulation_steps` | Per-GPU batch size and gradient accumulation |
| `training.optimizer_lr` / `scheduler_*` | LR + warmup / decay schedule |
| `training.max_training_steps` | Total post-training steps (default `200_000`) |
| `training.use_lora` | Optional LoRA fine-tuning, configured via `lora.*` |
| `deepspeed` | Points to `config/zero2.json` by default |
| `resume_ckpt` | Resume directory (contains `model/` and `state/training_state.pth`) |

---

## 3. Prepare the Robotwin dataset

Raw RoboTwin 2.0 episodes need to be converted into the HDF5 layout consumed by [`data/hdf5_robotwin_dataset.py`](../data/hdf5_robotwin_dataset.py). Follow [`utils/README.md`](../utils/README.md) for the full pipeline:

1. **Convert episodes** with `python utils/process_data_all.py` → `processed_data/<task_name>-<task_config>-<expert_data_num>/episode_*/episode_*.hdf5`.
2. **Compute normalization** with `python utils/norm_robotwin.py --mode eep` → `qpos_mean_std_online.pkl`.
3. *(Optional)* generate T5 text embeddings via `python utils/generate_t5_seen.py`. Robotwin **action-only** post-training does not require this.

Then verify [`config/dataset/robotwin.yaml`](../config/dataset/robotwin.yaml):

| Field | Value |
| --- | --- |
| `type` | `hdf5` |
| `action_type` | `"eep"` |
| `hdf5_dir` | Robotwin HDF5 dataset root (e.g. `/home/pub_data/hanyangyu/Datasets`) |
| `mean_std_path` | Path to `qpos_mean_std_online.pkl` produced in step 2 |
| `dataset_list` | Robotwin subsets and per-subset sampling weights (see `xtrainer.sample_weights`) |
| `action_chunk_size` / `img_history_size` | Action horizon and history frames |
| `image_size` | Input image size (default 224) |

> 💡 If you want to use the ready-to-train PoseVLA Robotwin data and normalization files directly, download them from
> [PoseVLA-robotwin-dataset on ModelScope](https://www.modelscope.ai/datasets/hanyangyu1021/PoseVLA-robotwin-dataset/files).

---

## 4. Smoke test the dataset

Before launching training, run the dataset visualization smoke test:

```bash
python dataset_wan.py
```

This writes:

- `dataset_wan_images.png`
- `dataset_wan_joints.png`

Use these plots to verify the RGB views, instruction text, and EEP action curves look stable before launching training.

---

## 5. Launch post-training

### Single GPU (debug)

```bash
python train_posttrain.py
```

### Multi-GPU (single node, 8 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --main_process_port 29504 --num_processes=8 train_posttrain.py
```

### Multi-node (cluster) — use the bundled script

```bash
bash train.sh
```

[`train.sh`](../train.sh) sets the standard NCCL / RDMA env vars for HCC machines and reads cluster-injected `RANK / MASTER_ADDR / MASTER_PORT / WORLD_SIZE / GPU_NUM` to dispatch `accelerate launch ... train_posttrain.py`.

The checkpoint layout (same as pre-training) is:

```
${ckpt_save_dir}/<step>/
├── model/                         # save_pretrained() output (with tokenizer)
└── state/training_state.pth/      # accelerator.save_state() (optional, controlled by save_training_state)
```

To resume from a checkpoint:

```bash
python train_posttrain.py resume_ckpt=/path/to/exp/29999
```

---

## 6. Evaluate on the RoboTwin simulator

After post-training finishes, deploy the resulting policy into the RoboTwin simulation platform for evaluation. Follow [`robotwin/PoseVLA/README.md`](../robotwin/PoseVLA/README.md) for:

- environment setup,
- registering the policy via [`robotwin/PoseVLA/deploy_policy.py`](../robotwin/PoseVLA/deploy_policy.py) and [`deploy_policy.yml`](../robotwin/PoseVLA/deploy_policy.yml),
- running parallel evaluation through [`run_auto_eval_posevla.sh`](../robotwin/PoseVLA/run_auto_eval_posevla.sh).

---

## 🐛 Troubleshooting

For checkpoint-loading issues (e.g. `draccus.utils.ParsingError`) shared with the pre-training pipeline, see [PRETRAIN.md → Troubleshooting](PRETRAIN.md#-troubleshooting).