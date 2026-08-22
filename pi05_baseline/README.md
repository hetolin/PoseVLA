# PI0.5 RoboTwin Baseline

This directory is a self-contained training and RoboTwin evaluation entry for
the 14-dimensional qpos PI0.5 baseline. It is intentionally separate from the
default PoseVLA post-training and EEP deployment paths.

The released checkpoint was originally trained in the legacy
`embodied_pi0_action` stack. The training entry here is its maintained,
runnable reproduction in this repository: it uses the same PI0.5 architecture,
qpos/action contract, 48-step horizon, normalization, and main hyperparameters.
It does not claim bit-for-bit equivalence with the historical run.

## Layout

```text
pi05_baseline/
├── config/train.yaml
├── prepare_stats.py
├── train.py
├── eval.sh
└── robotwin_eval/
    ├── deploy_policy.py
    ├── deploy_policy.yml
    └── model.py
```

None of these entries changes the default PoseVLA config, dataset
normalization, or RoboTwin deployment package.

## 1. Prerequisites

Follow the repository environment setup and place
[`pi05_base`](https://huggingface.co/hetolin/pi05_base) at:

```text
PoseVLA/pretrain/pi05_base/
```

Set the repository parent and RoboTwin HDF5 root:

```bash
export DEV_PATH=/path/to/parent/of/PoseVLA
export ROBOTWIN_HDF5_DIR=/path/to/robotwin/hdf5/root
export HYDRA_FULL_ERROR=1
```

The dataset root should contain the clean and randomized subsets used by the
baseline mixture:

```text
robotwin_processed/
robotwin_processed_random/
```

The default sampling weights are `1.0` and `10.0`.

## 2. Prepare normalization statistics

PI0.5 maps qpos state to `[-1, 1]` with min/max and normalizes actions with
mean/std. Generate one stats file used by both training and evaluation:

```bash
python pi05_baseline/prepare_stats.py \
  --dataset-root "$ROBOTWIN_HDF5_DIR" \
  --output /path/to/pi05_stats/qpos_hybrid_stats.pkl

export PI05_NORM_PATH=/path/to/pi05_stats/qpos_hybrid_stats.pkl
```

The file also stores derived `qpos_mean` and `qpos_std` values
(`midpoint`/`half-range`). This lets the unchanged shared HDF5 loader apply the
same min-max transform during training without adding PI0.5 behavior to the
default PoseVLA dataset implementation.

## 3. Train

Single-GPU smoke test:

```bash
python -m pi05_baseline.train \
  debug=True \
  training.max_training_steps=10 \
  training.max_evaluation_steps=1
```

Eight-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
  --main_process_port 29504 \
  --num_processes 8 \
  pi05_baseline/train.py
```

The baseline config fixes the important contract:

```text
training.pi05=true
training.add_extra_token=false
training.add_prior=false
dataset.action_type=qpos
dataset.action_chunk_size=48
```

Checkpoints are saved under:

```text
ckpt/pi05_robotwin_bs4_ga1/<step>/
```

Resume with:

```bash
python -m pi05_baseline.train resume_ckpt=/path/to/experiment/79999
```

## 4. Evaluate the released checkpoint

The validated 80k checkpoint is published as
[**hanyangyu1021/PoseVLA-pi05-robotwin** on ModelScope](https://www.modelscope.ai/models/hanyangyu1021/PoseVLA-pi05-robotwin).
Download it directly into the baseline checkpoint directory:

```bash
pip install modelscope
cd /path/to/PoseVLA
modelscope download \
  --model hanyangyu1021/PoseVLA-pi05-robotwin \
  --local_dir ckpt_robotwin/pi05_baseline
```

Expected local release layout:

```text
PoseVLA/ckpt_robotwin/pi05_baseline/
├── model/
│   ├── config.json
│   └── model.safetensors
└── qpos_hybrid_stats.pkl
```

Keep the released checkpoint's original model identity and configuration; no
conversion to a PoseVLA model name is required. The isolated loader accepts the
legacy config when it contains `"pi05": true`, `"add_extra_token": false`, and
`"add_prior": false`.

Install/link this repository at `RoboTwin/policy/PoseVLA`, then copy the
PoseVLA-compatible RoboTwin evaluation entry:

```bash
cd /path/to/RoboTwin
cp policy/PoseVLA/robotwin/PoseVLA/eval_policy.py script/eval_policy.py
export ROBOTWIN_ROOT=$PWD
```

Run a single task:

```bash
bash policy/PoseVLA/pi05_baseline/eval.sh \
  beat_block_hammer demo_clean 0
```

For randomized evaluation:

```bash
bash policy/PoseVLA/pi05_baseline/eval.sh \
  beat_block_hammer demo_randomized 0
```

The eval adapter is isolated at
`policy.PoseVLA.pi05_baseline.robotwin_eval`; it does not import or override the
default `policy.PoseVLA.robotwin.PoseVLA` policy package.

## 5. Released model naming

Use **PI0.5 RoboTwin Baseline** for the historical/released checkpoint. A new
model trained from this entry may also be described as a PI0.5 baseline
reproduction. Reserve **PoseVLA-PI0.5** for experiments that intentionally
initialize and train a PoseVLA backbone as part of the method.
