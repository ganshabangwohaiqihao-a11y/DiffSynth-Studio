#!/usr/bin/env python3
"""
单GPU Qwen-Image-Edit推理脚本
被qwen_image_multiprocess.py调用
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig


def main():
    parser = argparse.ArgumentParser(description="单GPU Qwen-Image-Edit推理脚本")
    
    parser.add_argument("--metadata-csv", type=str, required=True, help="元数据CSV文件")
    parser.add_argument("--lora-weights", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num-inference-steps", type=int, default=40, help="推理步数")
    parser.add_argument("--seed-start", type=int, default=42, help="初始种子")
    parser.add_argument("--height", type=int, default=800, help="生成图像高度")
    parser.add_argument("--width", type=int, default=848, help="生成图像宽度")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--sample-start", type=int, default=0, help="样本起始索引")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[GPU {args.gpu_id}] 初始化模型...")
    
    # 加载模型
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model.safetensors.index.json"),
            ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/text_encoder/model.safetensors.index.json"),
            ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/tokenizer/"),
        processor_config=ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/processor"),
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
            edit_image = Image.open(edit_image_path).convert("RGB")
            gt_image = Image.open(image_path).convert("RGB")
            
            print(f"[GPU {args.gpu_id}] 推理样本 {sample_id}/{total_samples}: {Path(edit_image_path).name}")
            
            # 推理
            seed = args.seed_start + idx
            generated_image = pipe(
                prompt,
                edit_image=[edit_image],
                seed=seed,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                edit_image_auto_resize=True,
                zero_cond_t=True,
            )

            # 统一三张图分辨率，确保拼图严格一致 (width x height)
            target_size = (args.width, args.height)
            condition_resized = edit_image.resize(target_size, Image.Resampling.LANCZOS)
            generated_resized = generated_image.resize(target_size, Image.Resampling.LANCZOS)
            gt_resized = gt_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # 保存单个图像
            condition_path = output_dir / f"sample_{sample_id:04d}_00_condition.jpg"
            generated_path = output_dir / f"sample_{sample_id:04d}_01_generated.jpg"
            gt_path = output_dir / f"sample_{sample_id:04d}_02_gt.jpg"

            condition_resized.save(condition_path)
            generated_resized.save(generated_path)
            gt_resized.save(gt_path)
            
            # 拼接图像：[条件 | 推理结果 | GT]
            gap_width = 20
            label_height = 40

            total_width = condition_resized.width + gap_width + generated_resized.width + gap_width + gt_resized.width
            total_height = generated_resized.height + label_height
            
            combined_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
            
            # 拼接图像
            combined_image.paste(condition_resized, (0, label_height))
            combined_image.paste(generated_resized, (condition_resized.width + gap_width, label_height))
            combined_image.paste(gt_resized, (condition_resized.width + gap_width + generated_resized.width + gap_width, label_height))
            
            # 添加标签
            draw = ImageDraw.Draw(combined_image)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # 标签：条件、推理、GT
            labels = ["Condition", "Generated", "GT"]
            x_positions = [
                condition_resized.width // 2,
                condition_resized.width + gap_width + generated_resized.width // 2,
                condition_resized.width + gap_width + generated_resized.width + gap_width + gt_resized.width // 2,
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
