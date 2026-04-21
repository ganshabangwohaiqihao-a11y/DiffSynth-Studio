#!/usr/bin/env python3
"""
单GPU FLUX推理脚本 - 显存碎片化修复版
关键改进：
1. 启用 PYTORCH_ALLOC_CONF=expandable_segments:True
2. 动态显存管理，定期清理缓存
3. 递进式降级策略（OOM时）
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ⚠️ CRITICAL: 在导入torch前设置环境变量
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig


def main():
    parser = argparse.ArgumentParser(description="单GPU FLUX推理脚本 - 碎片化修复版")
    
    parser.add_argument("--metadata-csv", type=str, required=True, help="元数据CSV文件")
    parser.add_argument("--lora-weights", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="推理步数")
    parser.add_argument("--seed-start", type=int, default=42, help="初始种子")
    parser.add_argument("--height", type=int, default=768, help="生成图像高度")
    parser.add_argument("--width", type=int, default=768, help="生成图像宽度")
    parser.add_argument("--embedded-guidance", type=float, default=2.5, help="嵌入制导强度")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--sample-start", type=int, default=0, help="样本起始索引")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[GPU {args.gpu_id}] 初始化FLUX.1-Kontext-dev模型...")
    print(f"[GPU {args.gpu_id}] 显存优化: PYTORCH_ALLOC_CONF=expandable_segments:True")
    
    # 加载模型
    try:
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
    except Exception as e:
        print(f"[GPU {args.gpu_id}] ✗ 模型加载失败: {e}")
        sys.exit(1)
    
    # 加载LoRA权重
    if args.lora_weights and args.lora_weights.strip() and os.path.exists(args.lora_weights):
        try:
            print(f"[GPU {args.gpu_id}] 加载LoRA权重: {args.lora_weights}")
            lora_config = ModelConfig(path=args.lora_weights)
            pipe.load_lora(pipe.dit, lora_config, alpha=1.0)
            print(f"[GPU {args.gpu_id}] ✓ LoRA权重加载完成")
        except Exception as e:
            print(f"[GPU {args.gpu_id}] ⚠️  LoRA加载失败，继续使用基础模型: {e}")
    else:
        print(f"[GPU {args.gpu_id}] 使用基础模型进行推理")
    
    # 清空初始化后的缓存
    torch.cuda.empty_cache()
    
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
    print(f"[GPU {args.gpu_id}] 需推理样本数: {total_samples}\n")
    
    # 推理
    successful = 0
    failed = 0
    current_steps = args.num_inference_steps
    current_height = args.height
    current_width = args.width
    
    for idx, sample in enumerate(metadata_list):
        sample_id = args.sample_start + idx
        
        try:
            # 获取路径和提示词
            image_path = sample.get('image', '').strip()
            edit_image_path = sample.get('edit_image', '').strip()
            prompt = sample.get('prompt', '').strip()
            
            if not image_path or not edit_image_path or not prompt:
                print(f"[GPU {args.gpu_id}] ⚠ 样本 {sample_id}: 缺少必要字段，跳过")
                failed += 1
                continue
            
            # 检查文件存在
            if not os.path.exists(edit_image_path) or not os.path.exists(image_path):
                print(f"[GPU {args.gpu_id}] ⚠ 样本 {sample_id}: 文件不存在，跳过")
                failed += 1
                continue
            
            # 加载图像
            kontext_image = Image.open(edit_image_path).convert("RGB")
            gt_image = Image.open(image_path).convert("RGB")
            
            print(f"[GPU {args.gpu_id}] 推理样本 {sample_id}/{total_samples}: {Path(edit_image_path).name}")
            
            # 推理
            seed = args.seed_start + idx
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    # 在重试前清空显存
                    if retry_count > 0:
                        print(f"[GPU {args.gpu_id}]   → 重试 {retry_count}/{max_retries} (步数: {current_steps}, 分辨率: {current_height}×{current_width})")
                        torch.cuda.empty_cache()
                    
                    generated_image = pipe(
                        prompt=prompt,
                        kontext_images=kontext_image,
                        embedded_guidance=args.embedded_guidance,
                        seed=seed,
                        num_inference_steps=current_steps,
                        height=current_height,
                        width=current_width,
                    )
                    break  # 成功，跳出重试循环
                    
                except torch.cuda.OutOfMemoryError as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise  # 达到最大重试次数，抛出异常
                    
                    # 降级策略
                    print(f"[GPU {args.gpu_id}]   ⚠ OOM检测，触发降级策略...")
                    torch.cuda.empty_cache()
                    
                    # 逐步降低
                    if current_steps > 10:
                        current_steps = max(10, current_steps - 10)
                    elif current_height > 512 or current_width > 512:
                        current_height = max(512, current_height - 128)
                        current_width = max(512, current_width - 128)
                        current_steps = args.num_inference_steps  # 恢复步数
                    else:
                        raise  # 无法再降级
            
            # 调整GT到相同大小
            gt_resized = gt_image.resize(generated_image.size)
            
            # 保存单个图像
            condition_path = output_dir / f"sample_{sample_id:04d}_00_condition.jpg"
            generated_path = output_dir / f"sample_{sample_id:04d}_01_generated.jpg"
            gt_path = output_dir / f"sample_{sample_id:04d}_02_gt.jpg"
            
            kontext_image.save(condition_path)
            generated_image.save(generated_path)
            gt_resized.save(gt_path)
            
            # 拼接图像
            gap_width = 20
            label_height = 40
            
            total_width = kontext_image.width + gap_width + generated_image.width + gap_width + gt_resized.width
            total_height = generated_image.height + label_height
            
            combined_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
            combined_image.paste(kontext_image, (0, label_height))
            combined_image.paste(generated_image, (kontext_image.width + gap_width, label_height))
            combined_image.paste(gt_resized, (kontext_image.width + gap_width + generated_image.width + gap_width, label_height))
            
            # 添加标签
            draw = ImageDraw.Draw(combined_image)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
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
            
            combined_path = output_dir / f"sample_{sample_id:04d}_03_combined.jpg"
            combined_image.save(combined_path)
            
            print(f"[GPU {args.gpu_id}] ✓ 样本 {sample_id} 完成 (步数: {current_steps}, 分辨率: {current_height}×{current_width})")
            
            # 清空显存以供下一个样本使用
            torch.cuda.empty_cache()
            
            successful += 1
            
        except Exception as e:
            print(f"[GPU {args.gpu_id}] ✗ 样本 {sample_id} 失败: {str(e)}")
            failed += 1
            # 失败后重置参数
            current_steps = args.num_inference_steps
            current_height = args.height
            current_width = args.width
            torch.cuda.empty_cache()
    
    print(f"\n[GPU {args.gpu_id}] 推理完成:")
    print(f"  ✓ 成功: {successful}")
    print(f"  ✗ 失败: {failed}")
    print(f"  成功率: {successful / (successful + failed) * 100:.1f}%")


if __name__ == "__main__":
    main()
