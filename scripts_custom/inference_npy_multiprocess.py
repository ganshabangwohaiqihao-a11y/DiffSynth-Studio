#!/usr/bin/env python3
"""
多GPU推理脚本：分配任务到多个GPU实现真正的并行推理
"""

# python3 scripts_custom/inference_npy_multiprocess.py \
#   --lora-weights /share/home/202230550120/DiffSynth-Studio/models/train10.3.27.23:37/FLUX.1-Kontext-dev_lora_learning_rate_constant_nochidu/epoch-9.safetensors \
#   --output-dir /share/home/202230550120/diffusers/output_2026.3.27_nolength/predictions_nochidu_v4 \
#   --metadata-json /share/home/202230550120/diffusers/output_2026.3.27_nolength/chinese_meta_test_enhanced_llm_mixed_clean_en.json \
#   --condition-png-dir /share/home/202230550120/diffusers/output_2026.3.27_nolength/viz_samples \
#   --id2c-path /share/home/202230550120/diffusers/metadata/id2c.json \
#   --c2rgb-path /share/home/202230550120/diffusers/metadata/c2rgb.json \
#   --num-gpus 4 \
#   --gt-npy-path /share/home/202230550120/diffusers/output_2026.3.27_nolength/chinese_test_1024x1024.npy

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run_inference_on_gpu(
    gpu_id: int,
    sample_start: int,
    sample_end: int,
    condition_npy_dir: str,
    condition_split: str,
    lora_weights: str,
    output_dir: str,
    metadata_json: str,
    condition_png_dir: str,
    gt_npy_path: str,
    embedded_guidance: float,
    seed_start: int,
) -> int:
    """在指定GPU上运行推理任务"""
    
    # 设置环境变量限制此进程只使用指定的GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 构建命令行
    cmd = [
        "python3",
        "scripts_custom/inference_npy.py",
        "--condition-npy-dir", condition_npy_dir,
        "--condition-split", condition_split,
        "--lora-weights", lora_weights,
        "--output-dir", output_dir,
        "--metadata-json", metadata_json,
        "--condition-png-dir", condition_png_dir,
        "--sample-start", str(sample_start),
        "--sample-end", str(sample_end),
        "--embedded-guidance", str(embedded_guidance),
        "--seed-start", str(seed_start),
    ]
    
    if gt_npy_path:
        cmd.extend(["--gt-npy-path", gt_npy_path])
    
    print(f"\n[GPU {gpu_id}] 启动推理进程：处理样本 {sample_start}-{sample_end-1}")
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={gpu_id}")
    
    # 在指定GPU上运行
    result = subprocess.run(cmd, env=env, cwd="/share/home/202230550120/diffusers")
    
    return result.returncode


def main():
    """主程序：分配任务到多个GPU"""
    
    parser = argparse.ArgumentParser(description="多GPU推理脚本")
    
    parser.add_argument(
        "--condition-npy-dir",
        type=str,
        default="output_2026.3.27",
        help="条件 NPY 文件所在目录"
    )
    parser.add_argument(
        "--condition-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="数据集分割"
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        required=True,
        help="LoRA 权重文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        required=True,
        help="元数据 JSON 文件"
    )
    parser.add_argument(
        "--condition-png-dir",
        type=str,
        required=True,
        help="条件PNG目录"
    )
    parser.add_argument(
        "--gt-npy-path",
        type=str,
        default=None,
        help="GT NPY文件路径"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="使用的GPU数量"
    )
    parser.add_argument(
        "--embedded-guidance",
        type=float,
        default=2.5,
        help="嵌入制导强度"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="初始种子"
    )
    
    args = parser.parse_args()
    
    # 检测样本数量
    condition_npy_path = Path(args.condition_npy_dir) / f"chinese_{args.condition_split}_condition_1024x1024.npy"
    import numpy as np
    try:
        condition_data = np.load(condition_npy_path)
        num_samples = condition_data.shape[0]
    except Exception as e:
        print(f"错误: 无法加载条件NPY文件: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"多GPU推理脚本")
    print(f"{'='*70}")
    print(f"总样本数:       {num_samples}")
    print(f"GPU数量:        {args.num_gpus}")
    print(f"每GPU样本数:    {num_samples // args.num_gpus} ~ {(num_samples + args.num_gpus - 1) // args.num_gpus}")
    print(f"输出目录:       {args.output_dir}")
    print(f"{'='*70}\n")
    
    # 分配样本到GPU
    samples_per_gpu = num_samples // args.num_gpus
    processes = []
    process_results = {}
    
    import subprocess
    
    for gpu_id in range(args.num_gpus):
        sample_start = gpu_id * samples_per_gpu
        sample_end = sample_start + samples_per_gpu if gpu_id < args.num_gpus - 1 else num_samples
        
        # 使用 subprocess 启动子进程
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd = [
            "python3",
            "scripts_custom/inference_npy.py",
            "--condition-npy-dir", args.condition_npy_dir,
            "--condition-split", args.condition_split,
            "--lora-weights", args.lora_weights,
            "--output-dir", args.output_dir,
            "--metadata-json", args.metadata_json,
            "--condition-png-dir", args.condition_png_dir,
            "--sample-start", str(sample_start),
            "--sample-end", str(sample_end),
            "--embedded-guidance", str(args.embedded_guidance),
            "--seed-start", str(args.seed_start),
        ]
        
        if args.gt_npy_path:
            cmd.extend(["--gt-npy-path", args.gt_npy_path])
        
        print(f"[GPU {gpu_id}] 启动推理进程：处理样本 {sample_start}-{sample_end-1}")
        print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={gpu_id}")
        
        # 启动子进程
        p = subprocess.Popen(
            cmd,
            env=env,
            cwd="/share/home/202230550120/diffusers",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        processes.append((gpu_id, p, sample_start, sample_end))
    
    print(f"\n[主进程] 已启动 {args.num_gpus} 个推理进程，等待完成...\n")
    
    # 实时输出每个进程的结果
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
    
    print(f"\n输出目录: {args.output_dir}")
    print(f"预期输出文件结构:")
    print(f"  {args.output_dir}/npy/")
    print(f"    - sample_{{idx}}_pred.npy (共 {num_samples} 个)")
    print(f"  {args.output_dir}/images/")
    print(f"    - sample_{{idx}}_00_condition.png")
    print(f"    - sample_{{idx}}_01_generated.png")
    print(f"    - sample_{{idx}}_02_gt_with_furn.png")
    print(f"    - sample_{{idx}}_03_combined.png")
    print()


if __name__ == "__main__":
    main()
