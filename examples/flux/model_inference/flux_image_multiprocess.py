#!/usr/bin/env python3
"""
多GPU推理脚本：从metadata.csv读取数据，使用多个GPU并行推理FLUX.1-Kontext-dev模型
用法:
  python flux_image_multiprocess.py \
    --metadata-csv /share/home/202230550120/DiffSynth-Studio/data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata1.csv \
    --lora-weights /share/home/202230550120/DiffSynth-Studio/models/train12/FLUX.1-Kontext-dev_lora/epoch-9.safetensors \
    --output-dir /share/home/202230550120/DiffSynth-Studio/examples/flux/model_inference/infer_results1 \
    --num-gpus 4
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict


def main():
    parser = argparse.ArgumentParser(description="多GPU FLUX.1-Kontext-dev推理脚本")
    
    parser.add_argument(
        "--metadata-csv",
        type=str,
        required=True,
        help="元数据CSV文件路径 (包含image, edit_image, prompt列)"
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        default=None,
        help="LoRA权重文件路径 (可选)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="使用的GPU数量"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="初始种子"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="生成图像高度"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="生成图像宽度"
    )
    parser.add_argument(
        "--embedded-guidance",
        type=float,
        default=2.5,
        help="嵌入制导强度"
    )
    
    args = parser.parse_args()
    
    # 读取CSV文件并检测样本数量
    metadata_list = []
    try:
        with open(args.metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_list.append(row)
        num_samples = len(metadata_list)
    except Exception as e:
        print(f"错误: 无法读取CSV文件: {e}")
        sys.exit(1)
    
    if num_samples == 0:
        print("错误: CSV文件中没有数据")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"多GPU FLUX.1-Kontext-dev推理脚本")
    print(f"{'='*70}")
    print(f"总样本数:           {num_samples}")
    print(f"GPU数量:            {args.num_gpus}")
    print(f"每GPU样本数:        {num_samples // args.num_gpus} ~ {(num_samples + args.num_gpus - 1) // args.num_gpus}")
    print(f"输出目录:           {output_dir}")
    print(f"LoRA权重:           {args.lora_weights if args.lora_weights else '(基础模型)'}")
    print(f"推理步数:           {args.num_inference_steps}")
    print(f"{'='*70}\n")
    
    # 分配样本到GPU
    samples_per_gpu = num_samples // args.num_gpus
    processes = []
    
    # 临时保存样本到files供子进程读取
    temp_metadata_dir = output_dir / "temp_metadata"
    temp_metadata_dir.mkdir(exist_ok=True)
    
    import subprocess
    
    for gpu_id in range(args.num_gpus):
        sample_start = gpu_id * samples_per_gpu
        sample_end = sample_start + samples_per_gpu if gpu_id < args.num_gpus - 1 else num_samples
        
        # 为该GPU保存对应的样本到临时文件
        temp_csv = temp_metadata_dir / f"metadata_gpu{gpu_id}.csv"
        with open(temp_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'edit_image', 'prompt'])
            writer.writeheader()
            writer.writerows(metadata_list[sample_start:sample_end])
        
        # 设置环境变量限制此进程只使用指定的GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd = [
            "python3",
            "flux_image_inference.py",
            "--metadata-csv", str(temp_csv),
            "--lora-weights", args.lora_weights if args.lora_weights else "",
            "--output-dir", str(output_dir),
            "--num-inference-steps", str(args.num_inference_steps),
            "--seed-start", str(args.seed_start + sample_start),
            "--height", str(args.height),
            "--width", str(args.width),
            "--embedded-guidance", str(args.embedded_guidance),
            "--gpu-id", str(gpu_id),
            "--sample-start", str(sample_start),
        ]
        
        print(f"[GPU {gpu_id}] 启动推理进程：处理样本 {sample_start}-{sample_end-1}")
        
        # 启动子进程
        p = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(Path(__file__).parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        processes.append((gpu_id, p, sample_start, sample_end))
    
    print(f"\n[主进程] 已启动 {args.num_gpus} 个推理进程，等待完成...\n")
    
    # 实时输出每个进程的结果
    process_results = {}
    for gpu_id, p, sample_start, sample_end in processes:
        print(f"\n{'='*70}")
        print(f"[GPU {gpu_id}] 输出 (样本 {sample_start}-{sample_end-1}):")
        print(f"{'='*70}")
        
        stdout, _ = p.communicate()
        if stdout:
            print(stdout)
        
        returncode = p.returncode
        process_results[gpu_id] = returncode
        
        if returncode == 0:
            print(f"[GPU {gpu_id}] ✓ 推理成功")
        else:
            print(f"[GPU {gpu_id}] ✗ 推理失败 (返回码: {returncode})")
    
    print(f"\n{'='*70}")
    print(f"[主进程] 所有推理进程已完成")
    print(f"{'='*70}")
    
    # 统计结果
    successful = sum(1 for r in process_results.values() if r == 0)
    failed = sum(1 for r in process_results.values() if r != 0)
    
    print(f"\n推理结果统计:")
    print(f"  ✓ 成功的GPU: {successful}/{args.num_gpus}")
    if failed > 0:
        print(f"  ✗ 失败的GPU: {failed}")
        failed_gpus = [g for g, r in process_results.items() if r != 0]
        print(f"  失败的GPU ID: {failed_gpus}")
    
    print(f"\n输出目录: {output_dir}")
    print(f"预期输出文件结构:")
    print(f"  {output_dir}/")
    print(f"    - sample_{{idx}}_00_condition.jpg         (条件输入图像)")
    print(f"    - sample_{{idx}}_01_generated.jpg         (推理结果)")
    print(f"    - sample_{{idx}}_02_gt.jpg                (GT图像)")
    print(f"    - sample_{{idx}}_03_combined.jpg          (三图拼接)")
    print()


if __name__ == "__main__":
    main()
