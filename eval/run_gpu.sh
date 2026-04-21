#!/bin/bash
#SBATCH --job-name=train_v1      # 任务名
#SBATCH --partition=gpuA800      # 【重点】指定 A800 分区
#SBATCH --nodes=1                # 只要 1 个节点
#SBATCH --ntasks-per-node=8      # 需要 8 个 CPU 核 (根据需要调整)
#SBATCH --gres=gpu:4            # 【重点】申请 2 张 GPU 卡
#SBATCH --output=log_%j.out      # 标准输出日志 (%j 会变成 Job ID)
#SBATCH --error=log_%j.err       # 错误日志# 


# source ~/.bashrc
# source /share/home/202230550120/anaconda3/bin/activate
# conda activate DiffSynth-Studio



# cd /share/home/202230550120/DiffSynth-Studio/评估

export PYTHONUNBUFFERED=1

python3 scripts/evaluate_sldn_layouts.py \
  --gt-data-dir /share/home/202230550120/diffusers \
  --gt-template output_2026.3.27/chinese_{split}_1024x1024.npy \
  --split test \
  --pred-dir /share/home/202230550120/diffusers/output_2026.3.27_nolength/predictions_nochidu_v4/npy_slic_c10_n1000_morph5 \
  --pred-glob "*_pred_slic_c10.npy" \
  --pred-index-regex "sample_(\\d+)" \
  --output-dir DiffSynth-Studio/评估/evaluation_results/nochidu_v9  \
  --channel-index 1 \
  --id2c-path /share/home/202230550120/diffusers/metadata/id2c.json \
  --c2rgb-path /share/home/202230550120/diffusers/metadata/c2rgb.json \

  2>&1 | tee train_nochidu_v9_fid.log 