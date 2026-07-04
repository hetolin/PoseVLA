# 🌟 PoseVLA — Pre-train

This document covers **large-scale pre-training** of the PoseVLA backbone, using:

- entry script: [`train_pretrain.py`](../train_pretrain.py)
- main config: [`config/base.yaml`](../config/base.yaml)
- launch scripts: [`scripts/launch/`](../scripts/launch/)

For Robotwin / downstream **post-training**, see [POSTTRAIN.md](POSTTRAIN.md).

> Make sure you have followed the env setup in [README.md → Quick Start](../README.md#-quick-start) before running anything below.

---

## 🌟 Pre-train from Scratch

PoseVLA is designed to be pre-trained on a mixture of **3D understanding data** (Omni3D / Omni6D / BOP / GraspClutter6D) and **robot action data** (Agibot / Droid / RDT / UMI / xtrainer / InternData-A1).

### Single-node multi-GPU (local debugging)

```bash
accelerate launch \
  --multi_gpu --num_machines 1 --num_processes 8 \
  --mixed_precision=bf16 \
  --main_process_ip 127.0.0.1 --main_process_port 56789 \
  --machine_rank 0 \
  train_pretrain.py
```

### Multi-node multi-GPU (H20 cluster)

```bash
bash scripts/launch/pretrain.sh False    # second arg toggles debug mode
```

`scripts/launch/pretrain.sh` reads cluster-injected env vars `RANK / MASTER_ADDR / MASTER_PORT / WORLD_SIZE / GPU_NUM` and starts [`train_pretrain.py`](../train_pretrain.py) through `accelerate launch`.

Three common configurations are pre-defined inside the script (the first is enabled by default, others are kept as comments):

1. **Train VLM only**: `co_training.vlm_training=True, action_training=False, data_3d=True`
2. **Train VLM (robot only)**: `data_3d=False`, continue finetuning from a trained VLM checkpoint
3. **Train VLA action only**: `vlm_training=False, action_training=True`

---

## 🌟 Fine-tune / Resume

`base.yaml` is automatically saved to `ckpt_save_dir/base.yaml` at each checkpoint and is merged back in on resume:

```bash
python train_pretrain.py resume_ckpt=/path/to/exp/29999
```

The checkpoint layout is:

```
ckpt/<exp_name>/<step>/
├── model/                         # Output of save_pretrained() (with tokenizer)
└── state/training_state.pth/      # Output of accelerator.save_state() (DeepSpeed state)
```

LoRA fine-tuning is supported via `training.use_lora=True` together with the `lora.*` block in [`config/base.yaml`](../config/base.yaml).

---

## ⚙️ Key Configuration ([`config/base.yaml`](../config/base.yaml))

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

Dataset combinations are aggregated via the `defaults:` section — see the tail of [`config/base.yaml`](../config/base.yaml).

### 🛠 Common Research Switches

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

## 🧠 Model Loading Branches

[`train_pretrain.py`](../train_pretrain.py) provides several weight-composition strategies (kept as comments, enable as needed):

1. **Load directly from a π0 / π0.5 checkpoint**
   ```python
   policy = PoseVLAPolicy.from_pretrained(cfg.model.pretrained_model_path, config=posevla_config, strict=False)
   ```
2. **PaliGemma without Action Expert (current default)**
   ```python
   policy = PoseVLAPolicy(posevla_config)
   policy.load_pretrained_vlm("pretrain/paligemma-3b-pt-224")
   ```
3. **Use the VLM from π0 and re-initialize the Action Expert**
4. **Train VLM only while reusing the Action Expert from π0**

> By default the training script runs `del policy.model.paligemma_with_expert.gemma_expert.model.embed_tokens / lm_head` to save memory.

---

## 📊 Evaluation

Entry point for 3D detection / pose tasks:

```bash
python eval_detection.py
```

> ⚠️ **Note**: [`eval_detection.py`](../eval_detection.py) currently hard-codes the checkpoint path and the
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

`eval_detection.py` provides:

- `evaluate_sample(...)` — per-sample 3D IoU / rotation / translation errors
- `compute_metrics_summary(...)` — mAP / PR curve aggregation (VOC 11-point & 101-point)

During training, validation automatically invokes these utilities and uploads PR curves, 3D-box visualizations, and text-prediction comparisons to W&B.

---

## 🗂 Data Preparation

Raw readers for each dataset:

- **Agibot**: [`data/ds_raw/agibot.py`](../data/ds_raw/agibot.py), download script [`scripts/download/agibot.sh`](../scripts/download/agibot.sh)
- **InternData-A1**: [`data/ds_raw/interndata_a1.py`](../data/ds_raw/interndata_a1.py), [`scripts/download/`](../scripts/download/) (download / unzip / yaml generation)
- **Droid / RDT / UMI / xtrainer**: see `data/ds_raw/*.py`
- **BOP / GraspClutter6D / Omni3D / Omni6D**: see `data/ds_train/dataset_*.py` and `config/dataset_*/`

Statistics computation:

```bash
python scripts/stats/compute_dataset_stat_hdf5_abs_joint.py
python scripts/stats/compute_dataset_stat_hdf5_rel_ee.py
```

The bin-statistics file used by the VLM 3D task is specified by `statistics_path_6d_dataset` (default: `./statistic_all_datasets/all_bins.pkl`).

---

## 🎄 Use Custom Datasets

To plug a new dataset into PoseVLA:

- **For robot action data** (HDF5 / LeRobot style):
  - Add a raw reader under [`data/ds_raw/`](../data/ds_raw/) following e.g. [`data/ds_raw/agibot.py`](../data/ds_raw/agibot.py).
  - Register it in the corresponding training dataset wrapper under [`data/ds_train/`](../data/ds_train/).
  - Add a Hydra config under `config/dataset/` (or `config/dataset_lerobot/`) and reference it from `defaults:` in [`config/base.yaml`](../config/base.yaml).
  - Drop a sample-list JSON into `config/dataset_meta/` if your reader needs one.

- **For 3D understanding data** (detection / pose):
  - Add the dataset class under [`data/ds_train/`](../data/ds_train/) (mimic `dataset_omni3d.py` / `dataset_bop.py` / `dataset_clutter.py` / `dataset_omni6d.py`).
  - Add token mapping logic in [`mapping_token.py`](../mapping_token.py) if a new label format is introduced.
  - Add a Hydra config under `config/dataset_omni3d/` / `config/dataset_bop/` / `config/dataset_clutter/` / `config/dataset_omni6d/`.

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

**Solution 2** — bypass the discriminator by passing `config=posevla_config` explicitly when reloading. In [`train_pretrain.py`](../train_pretrain.py), change the resume call to:

```python
policy = PoseVLAPolicy.from_pretrained(
    os.path.join(cfg.resume_ckpt, "model"),
    config=posevla_config,
    local_files_only=True,
)
```