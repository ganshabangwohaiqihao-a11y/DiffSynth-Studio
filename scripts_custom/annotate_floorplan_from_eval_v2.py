#!/usr/bin/env python3
"""
根据 evaluation_summary.json 给户型图添加标注
同时显示 gt_counts（绿色）和 pred_counts（蓝色）
参照表形式：原图 + 右侧参照表（基于 visualize_floorplan_with_legend.py 的布局）
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
import colorsys


def load_evaluation_data(eval_json_path):
    """加载评估结果文件"""
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ckl_data = data.get('ckl', {})
    class_names = ckl_data.get('class_names', [])
    per_sample_counts = ckl_data.get('per_sample_counts', [])
    
    return class_names, per_sample_counts


def generate_category_color(cat_id, num_classes):
    """为类别生成颜色（HSV色谱）"""
    hue = cat_id / max(num_classes, 1)
    saturation = 0.7
    value = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def generate_legend_image_dual(class_names, gt_counts, pred_counts, num_classes,
                               swatch_width=28, swatch_height=14):
    """
    生成参照表，同时显示 GT（绿色标记）和 Pred（蓝色标记）
    
    返回: (legend_image, legend_width, legend_height)
    """
    # 合并所有出现的类别
    all_cat_ids = set()
    for cat_id_str in gt_counts.keys():
        all_cat_ids.add(int(cat_id_str))
    for cat_id_str in pred_counts.keys():
        all_cat_ids.add(int(cat_id_str))
    
    display_cats = sorted(all_cat_ids)
    
    if not display_cats:
        # 空参照表
        display_cats = []
    
    num_cats = len(display_cats)
    
    # 尝试加载字体
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 第一遍：测量文本大小
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    
    text_heights = []
    text_widths = []
    for cat_id in display_cats:
        cat_name = class_names[cat_id] if cat_id < len(class_names) else f"unknown_{cat_id}"
        gt_count = int(gt_counts.get(str(cat_id), 0))
        pred_count = int(pred_counts.get(str(cat_id), 0))
        label = f"[{cat_id:2d}] {cat_name} GT:{gt_count} Pred:{pred_count}"
        bbox = temp_draw.textbbox((0, 0), label, font=font_label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_widths.append(text_width)
        text_heights.append(text_height)
    
    max_text_height = max(text_heights) if text_heights else 12
    max_text_width = max(text_widths) if text_widths else 150
    
    # 参照表布局：优先竖排
    if num_cats <= 12:
        cols = 1
        rows = num_cats
    elif num_cats <= 24:
        cols = 2
        rows = (num_cats + cols - 1) // cols
    else:
        cols = 3
        rows = (num_cats + cols - 1) // cols
    
    # 计算单元格尺寸
    gap_after_swatch = 8
    cell_width = swatch_width + gap_after_swatch + max_text_width + 16
    cell_height = max(max_text_height, swatch_height) + 12
    
    horizontal_gap = 6
    vertical_gap = 4
    
    legend_width = cols * cell_width + (cols - 1) * horizontal_gap + 20
    legend_height = rows * cell_height + (rows - 1) * vertical_gap + 50
    
    # 创建参照表
    legend_img = Image.new('RGB', (legend_width, legend_height), color=(255, 255, 255))
    legend_draw = ImageDraw.Draw(legend_img)
    
    # 绘制边框
    legend_draw.rectangle([0, 0, legend_width - 1, legend_height - 1],
                         outline=(0, 0, 0), width=2)
    
    # 绘制标题
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        title_font = font_label
    
    title = "Furniture Counts"
    title_bbox = legend_draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (legend_width - title_width) // 2
    legend_draw.text((title_x, 8), title, fill=(0, 0, 0), font=title_font)
    
    # 绘制说明
    legend_draw.text((12, 28), "GT(绿) Pred(蓝)", fill=(80, 80, 80), font=font_small)
    
    # 绘制每个类别
    for idx, cat_id in enumerate(display_cats):
        row = idx % rows
        col = idx // rows
        
        x = col * (cell_width + horizontal_gap) + 10
        y = row * (cell_height + vertical_gap) + 42
        
        # 获取颜色
        color = generate_category_color(cat_id, num_classes)
        
        # 绘制色块（左边，作为底色）
        swatch_x = x
        swatch_y = y + (max_text_height - swatch_height) // 2
        legend_draw.rectangle(
            [swatch_x, swatch_y, swatch_x + swatch_width, swatch_y + swatch_height],
            fill=color,
            outline=(0, 0, 0),
            width=1
        )
        
        # 获取计数
        gt_count = int(gt_counts.get(str(cat_id), 0))
        pred_count = int(pred_counts.get(str(cat_id), 0))
        
        # 在色块上标注：左上角绿色GT，右下角蓝色Pred
        if gt_count > 0:
            gt_text = str(gt_count)
            legend_draw.text((swatch_x + 2, swatch_y + 1), gt_text,
                           fill=(0, 180, 0), font=font_small)  # 绿色GT
        
        if pred_count > 0:
            pred_text = str(pred_count)
            pred_bbox = legend_draw.textbbox((0, 0), pred_text, font=font_small)
            pred_width = pred_bbox[2] - pred_bbox[0]
            legend_draw.text((swatch_x + swatch_width - pred_width - 2,
                            swatch_y + swatch_height - 9), pred_text,
                           fill=(0, 100, 255), font=font_small)  # 蓝色Pred
        
        # 绘制标签（右边）
        cat_name = class_names[cat_id] if cat_id < len(class_names) else f"unknown_{cat_id}"
        label = f"[{cat_id:2d}] {cat_name}"
        text_x = x + swatch_width + gap_after_swatch
        legend_draw.text((text_x, y), label, fill=(0, 0, 0), font=font_label)
    
    return legend_img, legend_width, legend_height


def composite_image_with_legend(image_path, class_names, gt_counts, pred_counts, 
                               num_classes, output_path=None):
    """
    将图像和参照表拼接，原图在左，参照表在右
    """
    if not Path(image_path).exists():
        print(f"  ✗ 图像不存在: {image_path}")
        return
    
    # 加载原图
    img_pil = Image.open(image_path).convert('RGB')
    img_width, img_height = img_pil.size
    
    # 生成参照表
    legend_img, legend_width, legend_height = generate_legend_image_dual(
        class_names, gt_counts, pred_counts, num_classes)
    
    # 创建拼接图像：原图 + 间隙 + 参照表
    margin = 10
    total_width = img_width + legend_width + margin * 3
    total_height = max(img_height, legend_height) + margin * 2
    
    composite = Image.new('RGB', (total_width, total_height), color=(240, 240, 240))
    
    # 放置原图（左）
    composite.paste(img_pil, (margin, margin))
    
    # 放置参照表（右）
    composite.paste(legend_img, (img_width + margin * 2, margin))
    
    # 保存
    if output_path is None:
        output_path = str(image_path).replace('.png', '_with_legend.png')
    
    composite.save(output_path, 'PNG', quality=95)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="根据 evaluation_summary.json 给户型图添加标注（同时显示GT和Pred）"
    )
    parser.add_argument('--eval-json', 
                       default='/share/home/202230550120/DiffSynth-Studio/评估/evaluation_results/chidu_v9/evaluation_summary.json',
                       help='评估结果 JSON 文件路径')
    parser.add_argument('--images-dir',
                       default='/share/home/202230550120/diffusers/output_2026.3.27/predictions_chidu_v4/images',
                       help='图像目录路径')
    parser.add_argument('--output-dir',
                       default='/share/home/202230550120/diffusers/output_2026.3.27/predictions_chidu_v4/images_with_legend',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 加载评估数据
    print(f"加载评估数据: {args.eval_json}")
    class_names, per_sample_counts = load_evaluation_data(args.eval_json)
    num_classes = len(class_names)
    print(f"  类别数: {num_classes}")
    print(f"  样本数: {len(per_sample_counts)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 建立样本索引到计数的映射
    sample_to_counts = {}
    for sample_data in per_sample_counts:
        sample_idx = sample_data['sample_index']
        gt_counts = sample_data['gt_counts']
        pred_counts = sample_data['pred_counts']
        sample_to_counts[sample_idx] = {
            'gt': gt_counts,
            'pred': pred_counts
        }
    
    # 处理所有 combined.png
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"错误：图像目录不存在: {images_dir}")
        return
    
    combined_files = sorted(images_dir.glob('*_03_combined.png'))
    print(f"\n找到 {len(combined_files)} 个 combined.png 文件")
    print(f"开始添加标注...\n")
    
    success_count = 0
    skip_count = 0
    
    for img_path in combined_files:
        # 从文件名提取样本索引
        filename = img_path.stem  # e.g., "sample_0_03_combined"
        parts = filename.split('_')
        try:
            sample_idx = int(parts[1])
        except (ValueError, IndexError):
            print(f"  ⚠ 跳过 {img_path.name}：无法解析样本索引")
            skip_count += 1
            continue
        
        # 获取计数
        if sample_idx not in sample_to_counts:
            print(f"  ⚠ 跳过 {img_path.name}：未找到评估结果")
            skip_count += 1
            continue
        
        counts = sample_to_counts[sample_idx]
        gt_counts = counts['gt']
        pred_counts = counts['pred']
        
        # 如果都为空则跳过
        if not gt_counts and not pred_counts:
            print(f"  - sample_{sample_idx}: 无预测结果")
            skip_count += 1
            continue
        
        # 确定输出路径
        output_path = Path(args.output_dir) / f"{img_path.stem}_with_legend.png"
        
        # 添加标注
        try:
            success = composite_image_with_legend(
                str(img_path), class_names, gt_counts, pred_counts,
                num_classes, str(output_path))
            if success:
                gt_list = ', '.join([
                    class_names[int(cid)] if int(cid) < len(class_names) else f"unknown_{int(cid)}"
                    for cid in sorted(gt_counts.keys())
                ])
                pred_list = ', '.join([
                    class_names[int(cid)] if int(cid) < len(class_names) else f"unknown_{int(cid)}"
                    for cid in sorted(pred_counts.keys())
                ])
                if len(gt_list) > 35:
                    gt_list = gt_list[:35] + "..."
                if len(pred_list) > 35:
                    pred_list = pred_list[:35] + "..."
                print(f"  ✓ sample_{sample_idx}:")
                if gt_list:
                    print(f"     GT: {gt_list}")
                if pred_list:
                    print(f"     Pred: {pred_list}")
                success_count += 1
        except Exception as e:
            print(f"  ✗ {img_path.name}: 错误 - {e}")
    
    print(f"\n完成！")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()
