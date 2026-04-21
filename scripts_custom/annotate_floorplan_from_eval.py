#!/usr/bin/env python3
"""
给预测的户型图添加标注（基于评估结果）。
使用 evaluation_summary.json 中的 per_sample_counts 精准获取每个房间的预测家具列表。
在 combined.png 右上角添加家具标注和颜色块。
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def load_evaluation_data(eval_json_path):
    """加载评估结果文件"""
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ckl_data = data.get('ckl', {})
    class_names = ckl_data.get('class_names', [])
    per_sample_counts = ckl_data.get('per_sample_counts', [])
    
    return class_names, per_sample_counts


def generate_legend_colors(num_classes):
    """为每个类别生成颜色（按类别顺序）"""
    # 使用简单的 HSV 色谱
    colors = []
    for i in range(num_classes):
        h = (i * 360 // num_classes) % 360
        # 简单转换为 RGB（使用 PIL 的颜色值）
        # 这里使用一个预先定义的调色板
        hue = i / num_classes
        saturation = 0.8
        value = 0.9
        
        # HSV 转 RGB（简化版）
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return colors


def create_legend_box(class_names, pred_counts, colors, font_size=12, 
                      box_width=200, max_items=15):
    """
    创建图例框（右上角标注）
    
    参数：
        class_names: 所有类别名称列表
        pred_counts: 预测的家具字典 {class_id_str: count}
        colors: 颜色列表
        font_size: 字体大小
        box_width: 输出图例宽度
        max_items: 最多显示的家具项数
    
    返回：
        (legend_image, legend_width, legend_height)
    """
    # 准备要显示的项目列表
    items = []
    for class_id_str, count in sorted(pred_counts.items(), 
                                      key=lambda x: int(x[0])):
        class_id = int(class_id_str)
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            # 超出范围的类别 ID，使用默认名称
            class_name = f"unknown_{class_id}"
        
        # 简化类名（去掉下划线，转为可读形式）
        display_name = class_name.replace('_', ' ').title()
        
        # 为超出范围的 ID 生成颜色
        if class_id < len(colors):
            color = colors[class_id]
        else:
            # 生成随机颜色
            color = (128 + (class_id * 17) % 128, 
                    128 + (class_id * 31) % 128, 
                    128 + (class_id * 47) % 128)
        
        items.append((class_id, display_name, count, color))
    
    # 限制显示项数
    if len(items) > max_items:
        items = items[:max_items]
    
    # 创建图例
    try:
        # 尝试加载系统字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                 size=font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                                     size=font_size)
        except:
            font = ImageFont.load_default()
    
    # 计算大小
    line_height = font_size + 4
    padding = 5
    color_box_size = font_size - 2
    spacing = 3
    
    legend_height = len(items) * line_height + padding * 2
    legend_width = box_width
    
    # 创建图例图像
    legend_img = Image.new('RGBA', (legend_width, legend_height), 
                          color=(255, 255, 255, 230))
    draw = ImageDraw.Draw(legend_img)
    
    # 绘制边框
    draw.rectangle([0, 0, legend_width - 1, legend_height - 1], 
                  outline=(0, 0, 0, 255), width=2)
    
    # 绘制每一项
    y = padding
    for class_id, display_name, count, color in items:
        # 绘制颜色块
        x_color = padding
        draw.rectangle([x_color, y, x_color + color_box_size, y + color_box_size],
                      fill=color, outline=(0, 0, 0, 100))
        
        # 绘制文本
        x_text = x_color + color_box_size + spacing + padding
        text = f"{display_name} (x{count})"
        draw.text((x_text, y), text, fill=(0, 0, 0, 255), font=font)
        
        y += line_height
    
    return legend_img, legend_width, legend_height


def annotate_image(image_path, class_names, pred_counts, colors, 
                   output_path=None, position='top-right'):
    """
    在图像上添加标注
    
    参数：
        image_path: 原始图像路径
        class_names: 所有类别名称
        pred_counts: 预测家具字典
        colors: 颜色列表
        output_path: 输出路径（如果为 None，覆盖原文件）
        position: 标注位置 ('top-right', 'bottom-right', etc.)
    """
    if not pred_counts:
        print(f"  跳过 {image_path}：没有预测结果")
        return
    
    # 打开原始图像
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    
    # 创建图例
    legend, legend_width, legend_height = create_legend_box(
        class_names, pred_counts, colors, font_size=11, box_width=200)
    
    # 创建带透明度的新图像
    img_with_alpha = img.convert('RGBA')
    
    # 决定放置位置
    if position == 'top-right':
        x = img_width - legend_width - 5
        y = 5
    elif position == 'top-left':
        x = 5
        y = 5
    elif position == 'bottom-right':
        x = img_width - legend_width - 5
        y = img_height - legend_height - 5
    else:  # bottom-left
        x = 5
        y = img_height - legend_height - 5
    
    # 确保不超出边界
    x = max(0, min(x, img_width - legend_width))
    y = max(0, min(y, img_height - legend_height))
    
    # 粘贴图例
    img_with_alpha.paste(legend, (x, y), legend)
    
    # 转回 RGB 并保存
    result = img_with_alpha.convert('RGB')
    if output_path is None:
        output_path = image_path
    result.save(output_path, 'PNG', quality=95)


def main():
    parser = argparse.ArgumentParser(
        description="根据 evaluation_summary.json 给户型图添加精准标注"
    )
    parser.add_argument('--eval-json', 
                       default='/share/home/202230550120/DiffSynth-Studio/评估/evaluation_results/chidu_v9/evaluation_summary.json',
                       help='评估结果 JSON 文件路径')
    parser.add_argument('--images-dir',
                       default='/share/home/202230550120/diffusers/output_2026.3.27/predictions_chidu_v4/images',
                       help='图像目录路径')
    parser.add_argument('--output-dir',
                       help='输出目录（如果不指定则覆盖原文件）')
    parser.add_argument('--position', default='top-right',
                       choices=['top-right', 'top-left', 'bottom-right', 'bottom-left'],
                       help='标注位置')
    parser.add_argument('--suffix', default='_annotated',
                       help='输出文件后缀（仅在指定 output-dir 时使用）')
    
    args = parser.parse_args()
    
    # 加载评估数据
    print(f"加载评估数据: {args.eval_json}")
    class_names, per_sample_counts = load_evaluation_data(args.eval_json)
    print(f"  类别数: {len(class_names)}")
    print(f"  样本数: {len(per_sample_counts)}")
    
    # 生成颜色
    colors = generate_legend_colors(len(class_names))
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 遍历每个样本
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"错误：图像目录不存在: {images_dir}")
        return
    
    # 建立样本索引到预测结果的映射
    sample_to_pred = {}
    for sample_data in per_sample_counts:
        sample_idx = sample_data['sample_index']
        pred_counts = sample_data['pred_counts']
        sample_to_pred[sample_idx] = pred_counts
    
    # 处理所有 combined.png
    combined_files = sorted(images_dir.glob('*_03_combined.png'))
    print(f"\n找到 {len(combined_files)} 个 combined.png 文件")
    print(f"开始添加标注...\n")
    
    for i, img_path in enumerate(combined_files):
        # 从文件名提取样本索引
        filename = img_path.stem  # e.g., "sample_0_03_combined"
        parts = filename.split('_')
        try:
            sample_idx = int(parts[1])
        except (ValueError, IndexError):
            print(f"  ⚠ 跳过 {img_path.name}：无法解析样本索引")
            continue
        
        # 获取预测结果
        if sample_idx not in sample_to_pred:
            print(f"  ⚠ 跳过 {img_path.name}：未找到评估结果")
            continue
        
        pred_counts = sample_to_pred[sample_idx]
        
        # 确定输出路径
        if args.output_dir:
            output_path = Path(args.output_dir) / f"{img_path.stem}{args.suffix}.png"
        else:
            output_path = None  # 覆盖原文件
        
        # 添加标注
        try:
            annotate_image(str(img_path), class_names, pred_counts, colors,
                          output_path=str(output_path) if output_path else None,
                          position=args.position)
            status = "✓" if output_path else "✓ (覆盖原文件)"
            furniture_list = ', '.join([
                (class_names[int(cid)] if int(cid) < len(class_names) else f"unknown_{int(cid)}")
                for cid in pred_counts.keys()
            ])
            if len(furniture_list) > 60:
                furniture_list = furniture_list[:60] + "..."
            print(f"  {status} sample_{sample_idx}: {furniture_list}")
        except Exception as e:
            import traceback
            print(f"  ✗ {img_path.name}: 错误 - {e}")
            traceback.print_exc()
    
    print(f"\n完成！共处理 {len(combined_files)} 个图像")
    if args.output_dir:
        print(f"输出目录: {args.output_dir}")
    else:
        print("已覆盖原文件")


if __name__ == '__main__':
    main()
