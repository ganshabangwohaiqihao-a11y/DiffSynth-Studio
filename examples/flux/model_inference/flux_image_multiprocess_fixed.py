#!/usr/bin/env python3
"""
FLUX多GPU推理脚本 - 显存碎片化修复版
解决OOM问题：启用扩展段+动态显存管理
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ⚠️ 关键：在导入torch之前设置环境变量
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def main():
    parser = argparse.ArgumentParser(description="多GPU FLUX.1-Kontext-dev推理脚本（碎片化修复版）")
    
    parser.add_argument(
        "--metadata-csv",
        type=str,
        required=True,
        help="元数据CSV文件路径"
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        default=None,
        help="LoRA权重文件路径"
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
        default=20,
        help="推理步数（默认20以减少显存占用）"
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
        default=768,
        help="生成图像高度（必须是8的倍数）"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="生成图像宽度（必须是8的倍数）"
    )
    parser.add_argument(
        "--embedded-guidance",
        type=float,
        default=2.5,
        help="嵌入制导强度"
    )
    
    args = parser.parse_args()
    
    # 验证尺寸是否为8的倍数
    if args.height % 8 != 0 or args.width % 8 != 0:
        print(f"⚠️  错误: 高度和宽度必须是8的倍数")
        print(f"   当前: height={args.height}, width={args.width}")
        sys.exit(1)
    
    # 读取CSV文件
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
    
    print(f"\n{'='*80}")
    print(f"多GPU FLUX.1-Kontext-dev推理脚本 - 显存碎片化修复版")
    print(f"{'='*80}")
    print(f"总样本数:              {num_samples}")
    print(f"GPU数量:               {args.num_gpus}")
    print(f"每GPU样本数:           {num_samples // args.num_gpus} ~ {(num_samples + args.num_gpus - 1) // args.num_gpus}")
    print(f"推理步数:              {args.num_inference_steps}")
    print(f"输出分辨率:            {args.height}×{args.width}")
    print(f"显存优化:              ✓ expandable_segments enabled")
    print(f"{'='*80}\n")
    
    # 分配样本到GPU
    samples_per_gpu = num_samples // args.num_gpus
    processes = []
    
    # 临时保存样本到files供子进程读取
    temp_metadata_dir = output_dir / "temp_metadata"
    temp_metadata_dir.mkdir(exist_ok=True)
    
    for gpu_id in range(args.num_gpus):
        sample_start = gpu_id * samples_per_gpu
        sample_end = sample_start + samples_per_gpu if gpu_id < args.num_gpus - 1 else num_samples
        
        # 为该GPU保存对应的样本到临时文件
        temp_csv = temp_metadata_dir / f"metadata_gpu{gpu_id}.csv"
        with open(temp_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'edit_image', 'prompt'])
            writer.writeheader()
            writer.writerows(metadata_list[sample_start:sample_end])
        
        # 设置环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        
        cmd = [
            "python3",
            "flux_image_inference_fixed.py",
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
        
        print(f"[GPU {gpu_id}] 启动推理进程：样本 {sample_start}-{sample_end-1}")
        
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
        print(f"\n{'='*80}")
        print(f"[GPU {gpu_id}] 输出 (样本 {sample_start}-{sample_end-1}):")
        print(f"{'='*80}")
        
        stdout, _ = p.communicate()
        if stdout:
            print(stdout)
        
        returncode = p.returncode
        process_results[gpu_id] = returncode
        
        if returncode == 0:
            print(f"[GPU {gpu_id}] ✓ 推理成功")
        else:
            print(f"[GPU {gpu_id}] ✗ 推理失败 (返回码: {returncode})")
    
    print(f"\n{'='*80}")
    print(f"[主进程] 所有推理进程已完成")
    print(f"{'='*80}")
    
    # 统计结果
    successful = sum(1 for r in process_results.values() if r == 0)
    failed = sum(1 for r in process_results.values() if r != 0)
    
    print(f"\n📊 推理结果统计:")
    print(f"  ✓ 成功的GPU: {successful}/{args.num_gpus}")
    if failed > 0:
        print(f"  ✗ 失败的GPU: {failed}")
        failed_gpus = [g for g, r in process_results.items() if r != 0]
        print(f"  失败的GPU ID: {failed_gpus}")
    
    print(f"\n📁 输出目录: {output_dir}")
    print(f"📋 生成的文件:")
    print(f"  - sample_{{idx}}_00_condition.jpg")
    print(f"  - sample_{{idx}}_01_generated.jpg")
    print(f"  - sample_{{idx}}_02_gt.jpg")
    print(f"  - sample_{{idx}}_03_combined.jpg")
    print()


if __name__ == "__main__":
    main()
