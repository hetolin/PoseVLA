# Robotwin Data Processing

Raw data comes from the official [RoboTwin2.0 dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
Use these scripts to convert it into the HDF5 layout used by `train_posttrain.py`.

## 1. Convert Episodes

```bash
python utils/process_data_all.py \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --expert_data_num 50
```

Default inputs:

```text
../../robotwin_raw_depth/<task_name>/<task_config>/data
../../robotwin_raw_dataset/<task_name>/<task_config>/instructions
```

Default output:

```text
processed_data/<task_name>-<task_config>-<expert_data_num>/episode_0/
  episode_0.hdf5
  cam_high.mp4
  cam_left_wrist.mp4
  cam_right_wrist.mp4
  instructions.json
```

The HDF5 contains `actions_qpos`, `actions_eep`, `observations/state_qpos`,
and `observations/state_eep`. Robotwin post-training uses `action_type: "eep"`.

Put converted data under the root configured in `config/dataset/robotwin.yaml`,
for example:

```text
/home/pub_data/hanyangyu/Datasets/robotwin_processed/
/home/pub_data/hanyangyu/Datasets/robotwin_processed_random/
```

If data changes, delete the dataset cache before rerunning:

```bash
rm config/dataset_meta/hdf5_robotwin_list.json
```

## 2. Compute Normalization

```bash
python utils/norm_robotwin.py \
  --processed_dir /home/pub_data/hanyangyu/Datasets/robotwin_processed_random \
  --output_dir /home/pub_data/hanyangyu/Datasets/robotwin_processed_random/global_stats_output_eep \
  --mode eep
```

This writes `qpos_mean_std_online.pkl` containing `action_mean`,
`action_std`, `eep_mean`, and `eep_std`. Point `robotwin.yaml` to it:

```yaml
mean_std_path: "/home/pub_data/hanyangyu/Datasets/robotwin_processed_random/global_stats_output_eep/qpos_mean_std_online.pkl"
```

Distribution plots are saved to `global_distribution_plots/`.

## 3. Optional T5 Embeddings

For pipelines that need precomputed text embeddings, generate one `t5_seen.pt`
next to each episode's `instructions.json`:

```bash
python utils/generate_t5_seen.py \
  --dataset_roots /home/pub_data/hanyangyu/Datasets/robotwin_processed_random \
  --wan_path pretrained_models/Wan2.2-TI2V-5B \
  --devices 0,1
```

The output is a list of T5 tensors, matching the order of
`instructions.json["seen"]`. Robotwin action-only post-training does not require
this step.

## 4. Smoke Test

```bash
python dataset_wan.py
```

Check `dataset_wan_images.png` and `dataset_wan_joints.png`, then launch:

```bash
python train_posttrain.py
```
