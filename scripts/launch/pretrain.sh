#!/bin/bash

#source ./taiji/start_h20.sh
NCCL_IB_GID_INDEX=3
NCCL_IB_SL=3
NCCL_CHECK_DISABLE=1
NCCL_P2P_DISABLE=0
NCCL_IB_DISABLE=0
NCCL_LL_THRESHOLD=16384
NCCL_IB_CUDA_SUPPORT=1
NCCL_IB_HCA=mlx5_bond
NCCL_NET_GDR_LEVEL=2
NCCL_IB_QPS_PER_CONNECTION=4
NCCL_IB_TC=160
NCCL_PXN_DISABLE=1
NCCL_IB_TIMEOUT=24
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=eth0
GLOO_SOCKET_IFNAME=eth0
TCCL_TOPO_AFFINITY=4
echo "针对 HCC 机型多机训练场景会开启 RDMA，并且会内置以下 NCCL 环境变量，用户使用 TI 平台的时候，无需显式设置。"


echo "========================"
echo "machine_rank: $RANK"
echo "主节点IP: $MASTER_ADDR"
echo "主节点端口: $MASTER_PORT"
echo "总机器数: $WORLD_SIZE"
echo "总GPU数量: $GPU_NUM"

DEBUG=$1


BRANCH="pi0"
BATCHES_PER_GPU=7
GPU_NUM=16
LR=5e-5 #5e-5 #2.5e-5 #5e-5
DECAY_LR=5e-6 #1e-5
OPTIMIZER_WEIGHT_DECAY=1e-10 #1e-2

#vlm
MAX_TRAINING_STEPS=100_000 #80_000 #800_000
SCHEDULER_DECAY_STEPS=80_000 #50_000 #600_000

IMG_HISTORY_SIZE=1
ACTION_CHUNK_SIZE=50

#EXP_NAME=${BRANCH}_224_prior_weighted_sample_intra_neg_bs${BATCHES_PER_GPU}_gpu${GPU_NUM}_lr${LR}_decay${OPTIMIZER_WEIGHT_DECAY}
EXP_NAME=${BRANCH}_224_prior_weighted_sample_intra_neg_nooverlap1_bs${BATCHES_PER_GPU}_gpu${GPU_NUM}_lr${LR}_decay${OPTIMIZER_WEIGHT_DECAY}
#EXP_NAME=${BRANCH}_224_prior_weighted_sample_intra_neg_nooverlap1_omni3d_objectron_bs${BATCHES_PER_GPU}_gpu${GPU_NUM}_lr${LR}_decay${OPTIMIZER_WEIGHT_DECAY}

#EXP_NAME=${BRANCH}_vlm224_traj_real_sim_aug_wo3d_bs${BATCHES_PER_GPU}_gpu${GPU_NUM}_lr${LR}_decay${OPTIMIZER_WEIGHT_DECAY}
#EXP_NAME=${BRANCH}_vlm224_traj_real_sim_aug_wo3d_direct_pose_bs${BATCHES_PER_GPU}_gpu${GPU_NUM}_lr${LR}_decay${OPTIMIZER_WEIGHT_DECAY}

#EXP_NAME=${BRANCH}_vlm224_traj_pretrain3d_bs${BATCHES_PER_GPU}_gpu${GPU_NUM}_lr${LR}_decay${OPTIMIZER_WEIGHT_DECAY}
#EXP_NAME=${BRANCH}_vlm224_traj_real_sim_aug_pretrain3d_bs${BATCHES_PER_GPU}_gpu${GPU_NUM}_lr${LR}_decay${OPTIMIZER_WEIGHT_DECAY}
#EXP_NAME=${BRANCH}_vlm224_traj_real_sim_aug_pretrain3d_direct_pose_bs${BATCHES_PER_GPU}_gpu${GPU_NUM}_lr${LR}_decay${OPTIMIZER_WEIGHT_DECAY}
#EXP_NAME=${BRANCH}_paligemma_noexpert_bs${BATCHES_PER_GPU}_gpu${GPU_NUM}_lr${LR}_decay${OPTIMIZER_WEIGHT_DECAY}

echo "========================"
echo "BRANCH: $BRANCH"
echo "DEBUG: $DEBUG"
echo "EXP_NAME: $EXP_NAME"
echo "BATCHES_PER_GPU: $BATCHES_PER_GPU"
echo "MAX_TRAINING_STEPS: $MAX_TRAINING_STEPS"
echo "SCHEDULER_DECAY_STEPS: $SCHEDULER_DECAY_STEPS"
echo "LR: $LR"
echo "DECAY_LR: $DECAY_LR"
echo "IMG_HISTORY_SIZE: $IMG_HISTORY_SIZE"
echo "ACTION_CHUNK_SIZE: $ACTION_CHUNK_SIZE"

# conda config --append envs_dirs /home/tione/notebook/workspace/shanejhuang/conda/envs
# conda activate val_omni3d
# cd /home/tione/notebook/workspace/hetolin/robot_code/embodied_pi0_action
# source taiji/start_h20.sh

# train VLM only
accelerate launch \
  --multi_gpu \
  --num_processes $GPU_NUM \
  --num_machines $WORLD_SIZE \
  --machine_rank $RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT  \
  train.py \
  wandb_project="pi0_ntp" \
  branch=$BRANCH \
  exp_name=$EXP_NAME \
  dataset.action_chunk_size=$ACTION_CHUNK_SIZE \
  dataset.img_history_size=$IMG_HISTORY_SIZE \
  training.batch_size=$BATCHES_PER_GPU \
  training.optimizer_lr=$LR \
  training.optimizer_weight_decay=$OPTIMIZER_WEIGHT_DECAY \
  training.max_training_steps=$MAX_TRAINING_STEPS \
  training.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
  training.scheduler_decay_lr=$DECAY_LR \
  training.is_knowledge_insulation=False \
  debug=$DEBUG \
  co_training.vlm_training=True \
  co_training.action_training=False \
  dataset.image_size=224 \
  training.add_extra_token=True \
  training.add_image_token=True \
  training.add_prior=True \
  training.weighted_sample=True \
  data_3d=True

## train VLM robot only
#accelerate launch \
#  --multi_gpu \
#  --num_processes $GPU_NUM \
#  --num_machines $WORLD_SIZE \
#  --machine_rank $RANK \
#  --main_process_ip $MASTER_ADDR \
#  --main_process_port $MASTER_PORT  \
#  train_pi0.py \
#  wandb_project="pi0_ntp" \
#  branch=$BRANCH \
#  exp_name=$EXP_NAME \
#  dataset.action_chunk_size=$ACTION_CHUNK_SIZE \
#  dataset.img_history_size=$IMG_HISTORY_SIZE \
#  training.batch_size=$BATCHES_PER_GPU \
#  training.optimizer_lr=$LR \
#  training.optimizer_weight_decay=$OPTIMIZER_WEIGHT_DECAY \
#  training.max_training_steps=$MAX_TRAINING_STEPS \
#  training.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
#  training.scheduler_decay_lr=$DECAY_LR \
#  training.is_knowledge_insulation=False \
#  debug=$DEBUG \
#  co_training.vlm_training=True \
#  co_training.action_training=False \
#  dataset.image_size=224 \
#  training.add_extra_token=True \
#  training.add_image_token=True \
#  training.add_prior=True \
#  training.weighted_sample=True \
#  data_3d=False \
#  model.pretrained_model_path="${DEV_PATH}/embodied_pi0_action/ckpt/pi0_224_prior_weighted_sample_intra_neg_bs7_gpu16_lr5e-5_decay1e-10/49999/model"

# train VLA action only
#accelerate launch \
#  --multi_gpu \
#  --num_processes $GPU_NUM \
#  --num_machines $WORLD_SIZE \
#  --machine_rank $RANK \
#  --main_process_ip $MASTER_ADDR \
#  --main_process_port $MASTER_PORT  \
#  train_pi0.py \
#  wandb_project="pi0_hy" \
#  branch=$BRANCH \
#  exp_name=$EXP_NAME \
#  dataset.action_chunk_size=$ACTION_CHUNK_SIZE \
#  dataset.img_history_size=$IMG_HISTORY_SIZE \
#  training.batch_size=$BATCHES_PER_GPU \
#  training.optimizer_lr=$LR \
#  training.optimizer_weight_decay=$OPTIMIZER_WEIGHT_DECAY \
#  training.max_training_steps=$MAX_TRAINING_STEPS \
#  training.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
#  training.scheduler_decay_lr=$DECAY_LR \
#  training.is_knowledge_insulation=False \
#  debug=$DEBUG \
#  co_training.vlm_training=False \
#  co_training.action_training=True \
#  dataset.image_size=224 \
#  training.add_extra_token=False \
#  training.add_image_token=False \
#  training.add_prior=False \
#  data_3d=False \


#python taiji/monitor_gpu.py --mode cpu
sleep 1000d