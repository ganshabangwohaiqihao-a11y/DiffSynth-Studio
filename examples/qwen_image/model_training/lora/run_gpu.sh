#!/bin/bash
#SBATCH --job-name=train_v1      # 任务名
#SBATCH --partition=gpuA800      # 【重点】指定 A800 分区
#SBATCH --nodes=1                # 只要 1 个节点
#SBATCH --ntasks-per-node=8      # 需要 8 个 CPU 核 (根据需要调整)
#SBATCH --gres=gpu:4            # 【重点】申请 2 张 GPU 卡
#SBATCH --output=log_%j.out      # 标准输出日志 (%j 会变成 Job ID)
#SBATCH --error=log_%j.err       # 错误日志# 


source ~/.bashrc
source /share/home/202230550120/anaconda3/bin/activate
conda activate DiffSynth-Studio

bash /share/home/202230550120/DiffSynth-Studio/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh