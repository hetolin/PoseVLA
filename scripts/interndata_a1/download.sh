#!/bin/bash

# 定义要下载的数据集
# important, will accelerate the speed
pip install hf-xet==1.1.2
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

REPO_NAME="InternRobotics/InternData-A1"
LOCAL_DIR=""

# 创建本地目录
#mkdir -p "$LOCAL_DIR"

# 下载函数
download_dataset() {
    while true; do
        echo "尝试下载数据集: $REPO_NAME"
        # 使用 huggingface-cli 下载数据集
        huggingface-cli download --resume-download --repo-type dataset $REPO_NAME --local-dir $LOCAL_DIR --local-dir-use-symlinks False
#         huggingface-cli download --resume-download --repo-type dataset $REPO_NAME --local-dir $LOCAL_DIR --include "sim_updated_lerobotv30/*" --local-dir-use-symlinks False

        # 检查下载是否成功
        if [ $? -eq 0 ]; then
            echo "数据集 $REPO_NAME 下载成功！"
            break
        else
            echo "下载数据集 $REPO_NAME 时出错，正在重试..."
            sleep 5  # 等待 5 秒后重试
        fi
    done
}

# 主程序
download_dataset