#!/usr/bin/env python3
"""
单GPU FLUX.1-Kontext-dev推理脚本
被flux_image_multiprocess.py调用
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig


def main():
    parser = argparse.ArgumentParser(description="单GPU FLUX.1-Kontext-dev推理脚本")
    
    parser.add_argument("--metadata-csv", type=str, required=True, help="元数据CSV文件")
    parser.add_argument("--lora-weights", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="推理步数")
    parser.add_argument("--seed-start", type=int, default=42, help="初始种子")
    parser.add_argument("--height", type=int, default=1024, help="生成图像高度")
    parser.add_argument("--width", type=int, default=1024, help="生成图像宽度")
    parser.add_argument("--embedded-guidance", type=float, default=2.5, help="嵌入制导强度")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--sample-start", type=int, default=0, help="样本起始索引")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[GPU {args.gpu_id}] 初始化FLUX.1-Kontext-dev模型...")
    
    # 加载模型
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path="/share/home/202230550120/models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors"),
            ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/text_encoder/model.safetensors"),
            ModelConfig(path=[
                "/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors",
                "/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors"
            ]),
            ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/ae.safetensors"),
        ],
        tokenizer_1_config=ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/tokenizer/"),
        tokenizer_2_config=ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/tokenizer_2/"),
    )
    
    # 加载LoRA权重
    if args.lora_weights and args.lora_weights.strip() and os.path.exists(args.lora_weights):
        print(f"[GPU {args.gpu_id}] 加载 LoRA 权重: {args.lora_weights}")
        lora_config = ModelConfig(path=args.lora_weights)
        pipe.load_lora(pipe.dit, lora_config, alpha=1.0)
        print(f"[GPU {args.gpu_id}] ✓ LoRA 权重加载完成")
    else:
        print(f"[GPU {args.gpu_id}] 使用基础模型进行推理 (未加载LoRA)")
    
    # 读取CSV
    metadata_list = []
    try:
        with open(args.metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_list.append(row)
    except Exception as e:
        print(f"[GPU {args.gpu_id}] 错误: 无法读取CSV文件: {e}")
        sys.exit(1)
    
    total_samples = len(metadata_list)
    print(f"[GPU {args.gpu_id}] 需推理样本数: {total_samples}")
    
    # 推理
    successful = 0
    failed = 0
    
    for idx, sample in enumerate(metadata_list):
        try:
            sample_id = args.sample_start + idx
            
            # 获取路径和提示词
            image_path = sample.get('image', '').strip()
            edit_image_path = sample.get('edit_image', '').strip()
            prompt = sample.get('prompt', '').strip()
            
            if not image_path or not edit_image_path or not prompt:
                print(f"[GPU {args.gpu_id}] ⚠ 样本 {sample_id}: 缺少必要字段，跳过")
                failed += 1
                continue
            
            # 检查文件是否存在
            if not os.path.exists(edit_image_path):
                print(f"[GPU {args.gpu_id}] ⚠ 样本 {sample_id}: 条件图像不存在: {edit_image_path}")
                failed += 1
                continue
            
            if not os.path.exists(image_path):
                print(f"[GPU {args.gpu_id}] ⚠ 样本 {sample_id}: GT图像不存在: {image_path}")
                failed += 1
                continue
            
            # 加载图像
            kontext_image = Image.open(edit_image_path).convert("RGB")
            gt_image = Image.open(image_path).convert("RGB")
            
            # 对齐图像尺寸到8的倍数（FLUX模型要求）
            def align_to_multiple(size, multiple=8):
                return (size // multiple) * multiple
            
            aligned_height = align_to_multiple(args.height, 8)
            aligned_width = align_to_multiple(args.width, 8)
            
            # 确保最小尺寸为256
            aligned_height = max(256, aligned_height)
            aligned_width = max(256, aligned_width)
            
            print(f"[GPU {args.gpu_id}] 推理样本 {sample_id}/{total_samples}: {Path(edit_image_path).name}")
            
            # 推理
            seed = args.seed_start + idx
            try:
                generated_image = pipe(
                    prompt=prompt,
                    kontext_images=kontext_image,
                    embedded_guidance=args.embedded_guidance,
                    seed=seed,
                    num_inference_steps=args.num_inference_steps,
                    height=aligned_height,
                    width=aligned_width,
                )
            except torch.cuda.OutOfMemoryError:
                # 显存不足时的回退方案：清空缓存并降低步数
                print(f"[GPU {args.gpu_id}] ⚠ 样本 {sample_id}: 显存不足，清空缓存并重试...")
                torch.cuda.empty_cache()
                generated_image = pipe(
                    prompt=prompt,
                    kontext_images=kontext_image,
                    embedded_guidance=args.embedded_guidance,
                    seed=seed,
                    num_inference_steps=max(10, args.num_inference_steps - 20),  # 降低步数
                    height=aligned_height - 64,  # 进一步降低分辨率
                    width=aligned_width - 64,
                )
            
            # 调整GT和生成图像到相同大小
            gt_resized = gt_image.resize(generated_image.size)
            
            # 保存单个图像
            condition_path = output_dir / f"sample_{sample_id:04d}_00_condition.jpg"
            generated_path = output_dir / f"sample_{sample_id:04d}_01_generated.jpg"
            gt_path = output_dir / f"sample_{sample_id:04d}_02_gt.jpg"
            
            kontext_image.save(condition_path)
            generated_image.save(generated_path)
            gt_resized.save(gt_path)
            
            # 拼接图像：[条件 | 推理结果 | GT]
            gap_width = 20
            label_height = 40
            
            total_width = kontext_image.width + gap_width + generated_image.width + gap_width + gt_resized.width
            total_height = generated_image.height + label_height
            
            combined_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
            
            # 拼接图像
            combined_image.paste(kontext_image, (0, label_height))
            combined_image.paste(generated_image, (kontext_image.width + gap_width, label_height))
            combined_image.paste(gt_resized, (kontext_image.width + gap_width + generated_image.width + gap_width, label_height))
            
            # 添加标签
            draw = ImageDraw.Draw(combined_image)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # 标签：条件、推理、GT
            labels = ["Condition", "Generated", "GT"]
            x_positions = [
                kontext_image.width // 2,
                kontext_image.width + gap_width + generated_image.width // 2,
                kontext_image.width + gap_width + generated_image.width + gap_width + gt_resized.width // 2,
            ]
            
            for label, x in zip(labels, x_positions):
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                draw.text((x - text_width // 2, 10), label, fill=(0, 0, 0), font=font)
            
            # 保存拼接图像
            combined_path = output_dir / f"sample_{sample_id:04d}_03_combined.jpg"
            combined_image.save(combined_path)
            
            print(f"[GPU {args.gpu_id}] ✓ 样本 {sample_id} 完成，已保存:")
            print(f"  - {condition_path.name}")
            print(f"  - {generated_path.name}")
            print(f"  - {gt_path.name}")
            print(f"  - {combined_path.name}")
            
            # 清空显存
            torch.cuda.empty_cache()
            successful += 1
            
        except Exception as e:
            print(f"[GPU {args.gpu_id}] ✗ 样本 {sample_id} 失败: {str(e)}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n[GPU {args.gpu_id}] 推理完成:")
    print(f"  ✓ 成功: {successful}")
    print(f"  ✗ 失败: {failed}")


if __name__ == "__main__":
    main()
