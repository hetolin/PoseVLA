# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  accelerate launch --main_process_port 29504 --num_processes=8 train_posttrain.py 

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



accelerate launch \
  --multi_gpu \
  --num_processes $GPU_NUM \
  --num_machines $WORLD_SIZE \
  --machine_rank $RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT  \
  train_posttrain.py 
