#!/usr/bin/env python3
"""
将2D户型图与识别的家具类别结合，在右上角生成对应的参照表
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict
import sys

# ── 从 parse_json_floorplan.py 复制的类别定义 ──
CATEGORIES = {
    0:  "void",
    1:  "floor",
    2:  "kids_bed",
    3:  "single_bed",
    4:  "double_bed",
    5:  "corner_side_table",
    6:  "round_end_table",
    7:  "coffee_table",
    8:  "console_table",
    9:  "tv_stand",
    10: "desk",
    11: "dressing_table",
    12: "table",
    13: "dining_table",
    14: "stool",
    15: "dressing_chair",
    16: "dining_chair",
    17: "chinese_chair",
    18: "armchair",
    19: "chair",
    20: "lounge_chair",
    21: "loveseat_sofa",
    22: "lazy_sofa",
    23: "sofa",
    24: "multi_seat_sofa",
    25: "chaise_longue_sofa",
    26: "l_shaped_sofa",
    27: "nightstand",
    28: "shelf",
    29: "bookshelf",
    30: "children_cabinet",
    31: "wine_cabinet",
    32: "cabinet",
    33: "wardrobe",
    34: "pendant_lamp",
    35: "ceiling_lamp",
    36: "door",
    37: "window",
    38: "bathtub",
    39: "toilet",
    40: "wash_basin",
    41: "shower",
    42: "refrigerator",
    43: "cooking_appliance",
    44: "piano",
    45: "washing_machine",
    46: "air_conditioner",
    47: "display_screen",
    48: "kitchen_cabinet",
    49: "bathroom_cabinet",
    50: "fitness_equipment",
    51: "shoe_cabinet",
    52: "tatami",
    53: "kitchen_island",
    54: "cove_ceiling",
}

# 已知的颜色映射（从parse_json_floorplan.py）
_VIZ_COLORMAP = {
    0:   (0,   0,   0),      # void - black
    1:   (200, 200, 200),    # floor - light gray
    36:  (255, 120,  0),     # door - orange
    37:  (0,  180, 255),     # window - cyan
    # Sample furniture colours
    2: (200,  80,  80), 3: (200,  80,  80), 4: (200,  80,  80),
    10: (80, 140, 200), 13: (100, 200, 100), 23: (200, 130,  50),
    33: (160,  60, 160), 39: (80, 200, 200), 42: (200, 200,  60),
}

# 颜色匹配置信度阈值（平方距离）
MAX_COLOR_DIST = 4200
MIN_SECOND_BEST_RATIO = 1.08
COFFEE_TABLE_MAX_DIST = 2600
BED_COLOR_MAX_DIST = 3600
BED_CATEGORIES = {2, 3, 4}
DOOR_COLOR_MAX_DIST = 8500
REMAP_DOUBLE_BED_TO_SINGLE = True


def generate_category_color(cat_id):
    """使用与parse_json_floorplan.py相同的算法生成category颜色"""
    if cat_id in _VIZ_COLORMAP:
        return _VIZ_COLORMAP[cat_id]
    rng = np.random.RandomState(cat_id * 17 + 3)
    return tuple(int(x) for x in rng.randint(60, 230, 3))


def find_color_regions(rgb_image):
    """使用OpenCV的inRange和findContours进行精确的颜色区域检测"""
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    h, w = bgr_image.shape[:2]
    regions_by_color = defaultdict(list)
    
    # 使用K-means聚类找出主要颜色
    pixels = bgr_image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 60, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    
    for cluster_id in range(len(centers)):
        mask = (labels.reshape(h, w) == cluster_id).astype(np.uint8) * 255
        
        if cv2.countNonZero(mask) < 100:
            continue
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color_bgr = tuple(centers[cluster_id])
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            x, y, w_c, h_c = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w_c // 2, y + h_c // 2
            
            regions_by_color[color_rgb].append({
                'x': x,
                'y': y,
                'width': w_c,
                'height': h_c,
                'area': int(area),
                'cx': cx,
                'cy': cy,
            })
    
    return regions_by_color


def find_closest_category(rgb_color):
    """根据RGB颜色找到最接近的category_id"""
    min_dist = float('inf')
    second_dist = float('inf')
    closest_cat = 0
    
    r0, g0, b0 = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])

    # door 优先判别
    door_r, door_g, door_b = _VIZ_COLORMAP[36]
    dist_door = (r0 - door_r) ** 2 + (g0 - door_g) ** 2 + (b0 - door_b) ** 2
    if dist_door <= DOOR_COLOR_MAX_DIST and r0 >= 175 and g0 >= 70 and b0 <= 85:
        return 36, dist_door, dist_door + 1

    # bed 优先判别
    bed_r, bed_g, bed_b = _VIZ_COLORMAP[2]
    dist_bed = (r0 - bed_r) ** 2 + (g0 - bed_g) ** 2 + (b0 - bed_b) ** 2
    if dist_bed <= BED_COLOR_MAX_DIST and r0 >= g0 + 20 and r0 >= b0 + 20 and b0 >= 50:
        return 4, dist_bed, dist_bed + 1

    # window / toilet 区分
    win_r, win_g, win_b = _VIZ_COLORMAP[37]
    toi_r, toi_g, toi_b = _VIZ_COLORMAP[39]
    dist_window = (r0 - win_r) ** 2 + (g0 - win_g) ** 2 + (b0 - win_b) ** 2
    dist_toilet = (r0 - toi_r) ** 2 + (g0 - toi_g) ** 2 + (b0 - toi_b) ** 2

    if dist_window <= 4500 and b0 >= g0 and (b0 - r0) >= 60:
        return 37, dist_window, dist_toilet

    if dist_window < dist_toilet and b0 > g0:
        return 37, dist_window, dist_toilet
    
    # 为所有category生成颜色并比较
    for cat_id in range(55):
        r, g, b = generate_category_color(cat_id)
        dist = (r0 - r)**2 + (g0 - g)**2 + (b0 - b)**2
        if dist < min_dist:
            second_dist = min_dist
            min_dist = dist
            closest_cat = cat_id
        elif dist < second_dist:
            second_dist = dist
    
    return closest_cat, min_dist, second_dist


def detect_categories_in_image(image_path):
    """
    检测图像中出现的家具类别
    返回: 检测到的 category_id 集合
    """
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    print(f"✓ 加载图像: {image_path}")
    print(f"  形状: {rgb_image.shape}")
    
    # 提取颜色区域
    regions = find_color_regions(rgb_image)
    print(f"✓ 识别出 {len(regions)} 个不同颜色区域")
    
    # 为每个颜色找到最接近的category
    detected_categories = set()
    
    for color, regions_list in regions.items():
        cat_id, color_dist, second_dist = find_closest_category(color)

        if REMAP_DOUBLE_BED_TO_SINGLE and cat_id == 4:
            cat_id = 3

        # 低置信度颜色过滤
        ratio = second_dist / max(color_dist, 1)
        is_exact_match = color_dist <= 5
        is_bed_like = cat_id in BED_CATEGORIES and color_dist <= BED_COLOR_MAX_DIST
        is_door_like = cat_id == 36 and color_dist <= DOOR_COLOR_MAX_DIST
        
        if color_dist > MAX_COLOR_DIST or ((ratio < MIN_SECOND_BEST_RATIO) and not is_exact_match and not is_bed_like and not is_door_like):
            continue

        if cat_id == 7 and color_dist > COFFEE_TABLE_MAX_DIST:
            continue

        # 跳过背景类别
        if cat_id not in (0, 1):
            detected_categories.add(cat_id)
    
    print(f"✓ 检测到 {len(detected_categories)} 个类别")
    for cat_id in sorted(detected_categories):
        cat_name = CATEGORIES.get(cat_id, f"unknown_{cat_id}")
        print(f"  [{cat_id:2d}] {cat_name}")
    
    return detected_categories


def generate_legend_image(detected_categories, swatch_width: int = 28, swatch_height: int = 14):
    """
    根据检测到的类别生成参照表图像
    返回: PIL Image
    """
    # 排序并过滤要显示的类别（去掉void和floor）
    display_cats = sorted([c for c in detected_categories if c not in (0, 1)])
    
    if not display_cats:
        # 如果没有家具，返回一个空的参照表
        display_cats = []
    
    num_cats = len(display_cats)
    
    # 尝试加载字体
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_label = ImageFont.load_default()
    
    # 第一遍：测量文本大小
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    
    text_heights = []
    text_widths = []
    for cat_id in display_cats:
        cat_name = CATEGORIES.get(cat_id, "unknown")
        label = f"[{cat_id:2d}] {cat_name}"
        bbox = temp_draw.textbbox((0, 0), label, font=font_label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_widths.append(text_width)
        text_heights.append(text_height)
    
    max_text_height = max(text_heights) if text_heights else 12
    max_text_width = max(text_widths) if text_widths else 100
    
    # 参照表布局：优先竖排成一列，只有数量多时才分列
    # 单列最多装不超过15个，超过则分成多列
    if num_cats <= 15:
        cols = 1
        rows = num_cats
    elif num_cats <= 30:
        cols = 2
        rows = (num_cats + cols - 1) // cols
    else:
        cols = 3
        rows = (num_cats + cols - 1) // cols
    
    # 计算单元格尺寸 - 加大间距
    gap_after_swatch = 10
    cell_width = swatch_width + gap_after_swatch + max_text_width + 16
    cell_height = max(max_text_height, swatch_height) + 16
    
    horizontal_gap = 6
    vertical_gap = 4
    
    legend_width = cols * cell_width + (cols - 1) * horizontal_gap + 20
    legend_height = rows * cell_height + (rows - 1) * vertical_gap + 50
    
    # 创建参照表图像
    legend_img = Image.new('RGB', (legend_width, legend_height), color=(255, 255, 255))
    legend_draw = ImageDraw.Draw(legend_img)
    
    # 绘制标题
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        title_font = ImageFont.load_default()
    
    title = "Furniture Legend"
    title_bbox = legend_draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (legend_width - title_width) // 2
    legend_draw.text((title_x, 8), title, fill=(0, 0, 0), font=title_font)
    
    # 绘制每个类别
    for idx, cat_id in enumerate(display_cats):
        # 竖排优先：从上往下，然后换列
        row = idx % rows
        col = idx // rows
        
        x = col * (cell_width + horizontal_gap) + 10
        y = row * (cell_height + vertical_gap) + 36
        
        # 获取颜色
        color = generate_category_color(cat_id)
        
        # 绘制色块（左边）
        swatch_x = x
        swatch_y = y + (max_text_height - swatch_height) // 2
        legend_draw.rectangle(
            [swatch_x, swatch_y, swatch_x + swatch_width, swatch_y + swatch_height],
            fill=color,
            outline=(0, 0, 0),
            width=2
        )
        
        # 绘制标签（右边）
        cat_name = CATEGORIES.get(cat_id, "unknown")
        label = f"[{cat_id:2d}] {cat_name}"
        text_x = x + swatch_width + gap_after_swatch
        legend_draw.text((text_x, y), label, fill=(0, 0, 0), font=font_label)
    
    return legend_img, legend_width, legend_height


def composite_image_with_legend(image_path, output_path):
    """
    1. 检测图像中的家具类别
    2. 生成参照表
    3. 将参照表拼接到原图右上角
    4. 保存结果
    """
    # 加载原图
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_image)
    img_width, img_height = img_pil.size
    
    print(f"\n{'='*60}")
    print(f"处理图像: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # 检测类别
    detected_cats = detect_categories_in_image(image_path)
    
    # 生成参照表
    legend_img, legend_width, legend_height = generate_legend_image(detected_cats)
    
    print(f"\n参照表尺寸: {legend_width} x {legend_height}")
    
    # 创建新图像：原图 + 右边留白放参照表
    margin = 10
    total_width = img_width + legend_width + margin * 3
    total_height = max(img_height, legend_height + margin * 2)
    
    # 如果参照表太高，创建背景扩展原图高度
    composite = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # 放置原图（左上角）
    composite.paste(img_pil, (margin, margin))
    
    # 放置参照表（右上角）
    composite.paste(legend_img, (img_width + margin * 2, margin))
    
    # 保存结果
    composite.save(output_path)
    print(f"\n✓ 保存结果: {output_path}")
    print(f"  最终尺寸: {total_width} x {total_height}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="给户型图添加家具参照表")
    parser.add_argument('--input', type=str, help='单个图像文件或包含图像的目录')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录（批量处理时指定）')
    parser.add_argument('--pattern', type=str, default='*_03_combined.png', help='图像文件模式（用于目录处理）')
    
    args = parser.parse_args()
    
    if not args.input:
        # 默认路径
        image_path = "/share/home/202230550120/DiffSynth-Studio/examples/flux/model_training/lora/inference_results_20260327_094313_train7.3.26.21:48/task_2462429_8073402_主卧_combined.jpg"
        if not Path(image_path).exists():
            print(f"✗ 错误: 找不到文件 {image_path}")
            sys.exit(1)
        image_name = Path(image_path).name
        output_path = Path(image_path).parent / f"with_legend_{image_name}"
        composite_image_with_legend(image_path, str(output_path))
    else:
        input_path = Path(args.input)
        
        # 判断是文件还是目录
        if input_path.is_file():
            # 单个文件处理
            if not input_path.exists():
                print(f"✗ 错误: 找不到文件 {input_path}")
                sys.exit(1)
            
            if args.output_dir:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                output_path = Path(args.output_dir) / f"with_legend_{input_path.name}"
            else:
                output_path = input_path.parent / f"with_legend_{input_path.name}"
            
            composite_image_with_legend(str(input_path), str(output_path))
        
        elif input_path.is_dir():
            # 目录批量处理
            if not args.output_dir:
                print("✗ 错误: 处理目录时必须指定 --output-dir")
                sys.exit(1)
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 查找匹配的图像文件
            image_files = sorted(input_path.glob(args.pattern))
            
            if not image_files:
                print(f"✗ 错误: 在 {input_path} 中找不到匹配 {args.pattern} 的文件")
                sys.exit(1)
            
            print(f"找到 {len(image_files)} 个图像文件，开始处理...\n")
            
            for i, image_file in enumerate(image_files, 1):
                output_path = output_dir / f"with_legend_{image_file.name}"
                print(f"[{i}/{len(image_files)}] 处理: {image_file.name}")
                try:
                    composite_image_with_legend(str(image_file), str(output_path))
                except Exception as e:
                    print(f"  ✗ 错误: {e}")
                    continue
                print()
            
            print(f"✓ 完成！已处理 {len(image_files)} 个文件")
            print(f"输出目录: {output_dir}")
        else:
            print(f"✗ 错误: {input_path} 既不是文件也不是目录")
            sys.exit(1)


if __name__ == "__main__":
    main()
