#!/usr/bin/env python3
"""
从RGB图像直接标注家具类别（通过颜色反向推导）
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
# 不显示 double_bed：将4重映射到3，避免家具块失去标签
REMAP_DOUBLE_BED_TO_SINGLE = True


def generate_category_color(cat_id):
    """
    使用与parse_json_floorplan.py相同的算法生成category颜色
    """
    if cat_id in _VIZ_COLORMAP:
        return _VIZ_COLORMAP[cat_id]
    # 对于unmapped的category，使用确定性随机颜色生成
    rng = np.random.RandomState(cat_id * 17 + 3)
    return tuple(int(x) for x in rng.randint(60, 230, 3))


def find_color_regions(rgb_image):
    """
    使用OpenCV的inRange和findContours进行精确的颜色区域检测
    """
    # 转换为BGR用于cv2处理
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    h, w = bgr_image.shape[:2]
    regions_by_color = defaultdict(list)
    
    # 使用K-means聚类找出主要颜色
    # 将图像reshape为2D数组
    pixels = bgr_image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # K-means聚类（找出最多60个主要颜色 - 增加以捕捉所有家具）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 60, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    
    # 为每个聚类颜色找区域
    for cluster_id in range(len(centers)):
        # 创建mask - 只保留该聚类的像素
        mask = (labels.reshape(h, w) == cluster_id).astype(np.uint8) * 255
        
        # 跳过太小的区域
        if cv2.countNonZero(mask) < 100:
            continue
        
        # 使用形态学操作清理噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 为每个轮廓创建区域
        color_bgr = tuple(centers[cluster_id])
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR to RGB
        
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
    """
    根据RGB颜色找到最接近的category_id
    为所有55个可能的category生成颜色，找最匹配的
    """
    min_dist = float('inf')
    second_dist = float('inf')
    closest_cat = 0
    
    # 转为整数避免溢出
    r0, g0, b0 = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])

    # door 优先判别：避免door被bed规则抢走
    door_r, door_g, door_b = _VIZ_COLORMAP[36]
    dist_door = (r0 - door_r) ** 2 + (g0 - door_g) ** 2 + (b0 - door_b) ** 2
    # door通常偏橙：R高、G中等、B低
    if dist_door <= DOOR_COLOR_MAX_DIST and r0 >= 175 and g0 >= 70 and b0 <= 85:
        return 36, dist_door, dist_door + 1

    # bed 优先判别：避免被随机颜色类别（如54）吸附
    bed_r, bed_g, bed_b = _VIZ_COLORMAP[2]
    dist_bed = (r0 - bed_r) ** 2 + (g0 - bed_g) ** 2 + (b0 - bed_b) ** 2
    # bed颜色通常偏红，且与bed参考色接近时直接判为bed
    if dist_bed <= BED_COLOR_MAX_DIST and r0 >= g0 + 20 and r0 >= b0 + 20 and b0 >= 50:
        return 4, dist_bed, dist_bed + 1

    # window / toilet 颜色容易互相混淆，先做优先判别（不改变k-means流程）
    win_r, win_g, win_b = _VIZ_COLORMAP[37]
    toi_r, toi_g, toi_b = _VIZ_COLORMAP[39]
    dist_window = (r0 - win_r) ** 2 + (g0 - win_g) ** 2 + (b0 - win_b) ** 2
    dist_toilet = (r0 - toi_r) ** 2 + (g0 - toi_g) ** 2 + (b0 - toi_b) ** 2

    # window 应更偏蓝（B通道明显高于G/R），并且与window参考色足够接近
    if dist_window <= 4500 and b0 >= g0 and (b0 - r0) >= 60:
        return 37, dist_window, dist_toilet

    # 对于边界颜色：若更接近window并且蓝通道占优，也优先判为window
    if dist_window < dist_toilet and b0 > g0:
        return 37, dist_window, dist_toilet
    
    # 为所有category生成颜色并比较
    for cat_id in range(55):  # 0-54 共55个category
        r, g, b = generate_category_color(cat_id)
        dist = (r0 - r)**2 + (g0 - g)**2 + (b0 - b)**2
        if dist < min_dist:
            second_dist = min_dist
            min_dist = dist
            closest_cat = cat_id
        elif dist < second_dist:
            second_dist = dist
    
    return closest_cat, min_dist, second_dist


def annotate_rgb_image(image_path, output_path="annotated_floorplan.png"):
    """
    使用OpenCV进行更精确的颜色识别和标注
    """
    # 使用cv2加载图像
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    print(f"✓ 加载图像: {image_path}")
    print(f"  形状: {rgb_image.shape}")
    
    # 提取颜色区域
    regions = find_color_regions(rgb_image)
    print(f"\n✓ 识别出 {len(regions)} 个不同颜色区域")
    
    # 为每个颜色找到最接近的category
    color_to_cat = {}
    color_to_dist = {}
    
    rejected_colors = 0
    for color, regions_list in sorted(regions.items(), key=lambda x: -sum(len(r) for r in x[1])):
        cat_id, color_dist, second_dist = find_closest_category(color)

        # 去掉double_bed标签，但保留床类识别
        if REMAP_DOUBLE_BED_TO_SINGLE and cat_id == 4:
            cat_id = 3

        # 低置信度颜色不标注：距离过远，或第一/第二候选过近（易误判）
        ratio = second_dist / max(color_dist, 1)
        # 对 exact color 和床类放宽，避免床被误删（2/3/4共享同色）
        is_exact_match = color_dist <= 5
        is_bed_like = cat_id in BED_CATEGORIES and color_dist <= BED_COLOR_MAX_DIST
        is_door_like = cat_id == 36 and color_dist <= DOOR_COLOR_MAX_DIST
        if color_dist > MAX_COLOR_DIST or ((ratio < MIN_SECOND_BEST_RATIO) and not is_exact_match and not is_bed_like and not is_door_like):
            cat_id = -1
            rejected_colors += 1

        # coffee_table误报常见：对7类使用更严格阈值
        if cat_id == 7 and color_dist > COFFEE_TABLE_MAX_DIST:
            cat_id = -1
            rejected_colors += 1

        color_to_cat[color] = cat_id
        color_to_dist[color] = color_dist

    if rejected_colors > 0:
        print(f"  已过滤低置信度颜色簇: {rejected_colors}")
    
    # 转成PIL Image用于绘制
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=12)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SimHei.ttf", size=12)
        except:
            font = ImageFont.load_default()
    
    # 标注每个category（去重后标注，避免同一位置叠多个标签）
    print("\n颜色→类别映射:")
    print("-" * 60)
    
    # 按category分组
    cat_to_regions = defaultdict(list)
    for color, regions_list in regions.items():
        cat_id = color_to_cat[color]
        for region in regions_list:
            region_with_meta = dict(region)
            region_with_meta['color'] = color
            region_with_meta['dist'] = color_to_dist[color]
            cat_to_regions[cat_id].append(region_with_meta)

    def _iou(a, b):
        ax1, ay1 = a['x'], a['y']
        ax2, ay2 = ax1 + a['width'], ay1 + a['height']
        bx1, by1 = b['x'], b['y']
        bx2, by2 = bx1 + b['width'], by1 + b['height']
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        union = a['width'] * a['height'] + b['width'] * b['height'] - inter
        return inter / max(1, union)

    # 收集候选标签（保留所有实例）
    candidates = []
    img_h, img_w = rgb_image.shape[:2]
    for cat_id in sorted(cat_to_regions.keys()):
        if cat_id < 0:
            continue
        if cat_id == 0:
            continue
        if cat_id == 1:
            continue
        for region in cat_to_regions[cat_id]:
            if region['area'] <= 200:
                continue

            # 床类过滤：床通常不是极细长条
            if cat_id in BED_CATEGORIES:
                rw = max(1, region['width'])
                rh = max(1, region['height'])
                aspect = max(rw, rh) / min(rw, rh)
                fill_ratio = region['area'] / max(1, rw * rh)

                # 过滤明显非床区域：过细长 / 过小 / 过稀疏
                if aspect > 5.5 or region['area'] < 900 or fill_ratio < 0.40:
                    continue

            # coffee_table 专属过滤：抑制空区域误报
            if cat_id == 7:
                rw = max(1, region['width'])
                rh = max(1, region['height'])
                aspect = max(rw, rh) / min(rw, rh)
                fill_ratio = region['area'] / max(1, rw * rh)
                near_border = region['x'] <= 8 or region['y'] <= 8 or (region['x'] + rw) >= (img_w - 8) or (region['y'] + rh) >= (img_h - 8)

                # 过细长、过稀疏、贴边、或面积异常的区域不认为是coffee_table
                if aspect > 3.2 or fill_ratio < 0.30 or near_border or region['area'] < 350 or region['area'] > 20000:
                    continue

            item = dict(region)
            item['cat_id'] = cat_id
            candidates.append(item)

    # 大区域优先；同面积时优先颜色距离更小（更可信）
    candidates.sort(key=lambda r: (-r['area'], r['dist']))

    selected = []
    for cand in candidates:
        keep = True
        replace_index = -1
        for idx, sel in enumerate(selected):
            center_dist2 = (cand['cx'] - sel['cx']) ** 2 + (cand['cy'] - sel['cy']) ** 2
            overlap = _iou(cand, sel)
            # 同一位置判定：中心很近或框重叠明显
            if center_dist2 <= 45 * 45 or overlap >= 0.30:
                c_cat, s_cat = cand['cat_id'], sel['cat_id']

                # 特判：window 与 toilet 冲突时，优先保留 window
                if (c_cat, s_cat) in ((37, 39), (39, 37)):
                    if c_cat == 37:
                        replace_index = idx
                    keep = (c_cat == 37)
                else:
                    # 其它冲突：保留面积更大或颜色距离更小者
                    if cand['area'] > sel['area'] * 1.15 or (cand['area'] >= sel['area'] and cand['dist'] < sel['dist']):
                        replace_index = idx
                    else:
                        keep = False
                if not keep and replace_index == -1:
                    break

        if keep:
            if replace_index >= 0:
                selected[replace_index] = cand
            else:
                selected.append(cand)

    # 同类别NMS：去掉同一色块被切成多个小区域导致的重复标签
    def _iou_same_cat(a, b):
        ax1, ay1 = a['x'], a['y']
        ax2, ay2 = ax1 + a['width'], ay1 + a['height']
        bx1, by1 = b['x'], b['y']
        bx2, by2 = bx1 + b['width'], by1 + b['height']
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        union = a['width'] * a['height'] + b['width'] * b['height'] - inter
        return inter / max(1, union)

    nms_selected = []
    for cat_id in sorted({r['cat_id'] for r in selected}):
        same_cat = [r for r in selected if r['cat_id'] == cat_id]
        same_cat.sort(key=lambda r: (-r['area'], r['dist']))

        kept = []
        for cand in same_cat:
            suppressed = False
            for ex in kept:
                center_dist2 = (cand['cx'] - ex['cx']) ** 2 + (cand['cy'] - ex['cy']) ** 2
                iou = _iou_same_cat(cand, ex)
                # 同类别且中心接近/重叠较大时，视为同一色块
                if center_dist2 <= 150 * 150 or iou >= 0.08:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(cand)
        nms_selected.extend(kept)

    selected = nms_selected

    # 按类别输出统计
    cat_count = defaultdict(int)
    for r in selected:
        cat_count[r['cat_id']] += 1
    for cat_id in sorted(cat_count.keys()):
        cat_name = CATEGORIES.get(cat_id, f"unknown_{cat_id}")
        print(f"  [{cat_id:2d}] {cat_name:20} ({cat_count[cat_id]} 个区域)")

    # 绘制去重后的标签
    placed_label_boxes = []
    drawn_cat_count = defaultdict(int)

    def _box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / max(1, union)

    for region in selected:
        cat_id = region['cat_id']
        cat_name = CATEGORIES.get(cat_id, f"unknown_{cat_id}")
        cx, cy = region['cx'], region['cy']
        text = f"{cat_id}:{cat_name}"
        label_box = (cx - 40, cy - 11, cx + 40, cy + 11)

        # 文字框防重叠：同一位置只保留一个标签
        overlapped = False
        for ex in placed_label_boxes:
            if _box_iou(label_box, ex) >= 0.15:
                overlapped = True
                break
        if overlapped:
            continue

        draw.rectangle(
            [(label_box[0], label_box[1]), (label_box[2], label_box[3])],
            fill=(0, 0, 0, 200)
        )
        draw.text((cx-35, cy-6), text, fill=(255, 255, 255), font=font)
        placed_label_boxes.append(label_box)
        drawn_cat_count[cat_id] += 1

    print("\n实际绘制标签数量:")
    print("-" * 60)
    for cat_id in sorted(drawn_cat_count.keys()):
        cat_name = CATEGORIES.get(cat_id, f"unknown_{cat_id}")
        print(f"  [{cat_id:2d}] {cat_name:20} ({drawn_cat_count[cat_id]} 个标签)")
    
    # 保存
    pil_image.save(output_path)
    print(f"\n✓ 保存标注图像: {output_path}")


def main():
    # ── 在这里指定输入图像路径 ──
    image_path = "/home/chengjiajia/diffusers/inference_results/task_1_guest_bath_00_generated.jpg"
    
    if not Path(image_path).exists():
        print(f"✗ 错误: 找不到文件 {image_path}")
        sys.exit(1)
    
    # 生成输出文件名：annotated_ + 原始文件名
    image_name = Path(image_path).name
    output_path = Path(image_path).parent / f"annotated_{image_name}"
    
    print(f"処理中: {image_name}")
    annotate_rgb_image(image_path, str(output_path))


if __name__ == "__main__":
    main()
