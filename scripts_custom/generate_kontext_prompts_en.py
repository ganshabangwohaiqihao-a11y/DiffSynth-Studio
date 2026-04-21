#!/usr/bin/env python3
"""
Generate English prompts with room-relative item positions for Kontext CSV.

Target format example:
2D semantic segmentation map, top-down floorplan, simple solid color blocks.
room: bedroom. items: double bed at the bottom, chair in the top-left corner.
"""
# 根据家具位置生成英文Prompt

# 读取 CSV 文件和 JSON 房间数据
# 检测房间内的家具及其位置（top-left, bottom, center等）
# 生成描述性英文Prompt，包含家具位置信息
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Room type rules (same intent as parse_json_floorplan.py)
ROOM_TYPE_RULES: List[Tuple[List[str], str]] = [
    (["主卧", "主人房", "主人卧室", "次卧", "客卧", "儿童房", "小孩房", "保姆房", "女儿房", "男儿房", "次卧室", "儿童卧室", "长辈房", "老人房", "男孩房", "女孩房", "小女儿房", "大女儿房", "卧室"], "bedroom"),
    (["客餐厅", "客厅餐厅", "起居餐厅", "起居/餐厅", "大客餐厅", "客卧一体"], "living dining room"),
    (["客厅", "起居室", "会客厅", "大厅", "厅房"], "living room"),
    (["餐厅", "餐区", "用餐区", "饭厅", "就餐区"], "dining room"),
    (["厨房", "西厨", "中厨", "开放式厨房", "封闭厨房", "厨卫", "厨餐"], "kitchen"),
    (["主卫", "主人卫生间", "主浴室", "卫生间", "洗手间", "公卫", "次卫", "浴室", "盥洗室", "卫浴", "厕所", "干区", "湿区", "洗衣房", "客卫", "公共卫生间", "共用卫生间", "淋浴房"], "bathroom"),
    (["书房", "工作室", "多功能间", "多功能室"], "study"),
    (["阳台", "生活阳台", "景观阳台", "服务阳台", "露台", "花园阳台", "小阳台", "花园", "入户花园", "建筑景观"], "balcony"),
    (["玄关", "门厅", "入户区", "入口区", "过厅", "入户"], "entrance"),
    (["衣帽间", "步入式衣帽间", "储物间", "储藏间", "储藏室", "杂物间"], "storage room"),
]


# Furniture keyword → normalized English label
FURNITURE_RULES_EN: List[Tuple[List[str], str]] = [
    (["儿童床", "婴儿床", "上下床", "双层床", "高低床", "bunk"], "kids bed"),
    (["单人床", "1.0米", "0.9米", "单床"], "single bed"),
    (["双人床", "大床", "1.5米", "1.6米", "1.8米", "2.0米", "主卧床", "床垫", "床架", "床板", "软床", "硬床", "箱体床", "榻榻米床", "床"], "double bed"),
    (["角几", "边几", "角桌", "角台", "小边柜"], "corner side table"),
    (["圆形茶几", "圆几", "圆形小桌"], "round end table"),
    (["茶几", "咖啡桌", "客厅桌"], "coffee table"),
    (["玄关台", "玄关桌", "端景台", "条案", "条几"], "console table"),
    (["电视柜", "TV柜", "视听柜", "电视架", "电视台"], "tv stand"),
    (["书桌", "写字台", "工作台", "办公桌", "电脑桌", "学习桌", "电脑台", "工作桌", "工作站", "学习台"], "desk"),
    (["梳妆台", "梳妆", "化妆台", "化妆桌", "梳妆柜"], "dressing table"),
    (["餐桌", "饭桌", "用餐桌", "吃饭桌"], "dining table"),
    (["换鞋凳", "矮凳", "凳", "脚凳", "踏脚凳", "茶几凳", "圆凳", "方凳"], "stool"),
    (["梳妆椅", "化妆椅"], "dressing chair"),
    (["餐椅", "饭椅", "靠背椅"], "dining chair"),
    (["中式椅", "古典椅", "明式椅", "太师椅", "圈椅", "官帽椅"], "chinese chair"),
    (["扶手椅", "休闲单椅", "单椅", "主人椅", "书椅", "布艺椅"], "armchair"),
    (["躺椅", "懒人椅", "休闲椅"], "lounge chair"),
    (["双人沙发", "二人沙发", "两人沙发"], "loveseat sofa"),
    (["懒人沙发", "豆袋", "蛋形沙发", "懒沙发"], "lazy sofa"),
    (["L型沙发", "L形沙发", "转角沙发", "拐角沙发", "U型沙发", "U形沙发"], "L-shaped sofa"),
    (["贵妃", "贵妃椅", "贵妃沙发", "chaise"], "chaise longue sofa"),
    (["多人沙发", "多位沙发", "四人沙发", "五人沙发"], "multi-seat sofa"),
    (["三人沙发", "三座沙发", "布艺沙发", "皮艺沙发", "皮沙发", "布沙发", "沙发"], "sofa"),
    (["床头柜", "床边柜", "床头桌", "床头台"], "nightstand"),
    (["书架", "书柜", "文件架", "杂志架", "书报架"], "bookshelf"),
    (["酒柜", "红酒柜"], "wine cabinet"),
    (["衣柜", "大衣柜", "走入式衣", "步入式衣", "衣帽柜", "移门衣柜", "推拉门衣柜", "嵌入式衣柜"], "wardrobe"),
    (["置物架", "搁板", "层板架", "展示架", "陈列架", "花架", "储物架", "收纳篮", "收纳架", "编织篮"], "shelf"),
    (["儿童柜", "玩具柜", "儿童收纳箱"], "shelf"),
    (["吊灯", "垂吊灯", "悬挂灯", "吊线灯", "铁艺吊灯", "北欧吊灯", "餐厅吊灯", "饭厅吊灯"], "pendant lamp"),
    (["吊顶", "石膏线", "顶面", "天花板", "天花造型", "灯槽", "顶角", "造型顶", "集成吊顶", "灯带", "线性灯", "灯条"], "cove ceiling"),
    (["吸顶灯", "射灯", "筒灯", "格栅灯", "主灯", "厨卫灯", "灯"], "ceiling lamp"),
    (["浴缸", "泡澡盆", "按摩浴缸", "浴盆", "独立浴缸"], "bathtub"),
    (["马桶", "坐便器", "蹲便器", "智能马桶", "坐厕"], "toilet"),
    (["洗手台", "洗脸盆", "洗手盆", "洗漱台", "面盆", "洗面台", "台盆", "洗脸台", "盥洗台"], "wash basin"),
    (["淋浴", "花洒", "淋浴区", "淋浴房", "沐浴", "淋浴隔断", "花洒头", "喷头", "淋浴喷头"], "shower"),
    (["浴室柜", "卫浴柜", "镜柜", "洗手间柜", "卫生间柜"], "bathroom cabinet"),
    (["冰箱", "冰柜", "冷藏柜", "冰粮", "小冰箱", "双门冰箱"], "refrigerator"),
    (["灶台", "油烟机", "烟机", "蒸烤箱", "微波炉", "洗碗机", "蒸箱", "烤箱", "燃气灶", "电磁炉", "集成灶", "水槽", "洗菜盆", "厨房水盆", "水盆台"], "cooking appliance"),
    (["橱柜", "厨柜", "整体厨房", "厨房柜", "吊柜", "地柜", "厨房收纳"], "kitchen cabinet"),
    (["岛台", "西厨岛", "厨房岛台", "中岛台", "岛形台"], "kitchen island"),
    (["钢琴", "三角钢琴", "立式钢琴", "电钢琴", "电子琴", "数码钢琴"], "piano"),
    (["电视", "电视机"], "display screen"),
    (["洗衣机", "烘干机", "洗烘一体机", "洗烘"], "washing machine"),
    (["空调", "中央空调", "空调挂机", "空调柜机", "立式空调", "暖气"], "air conditioner"),
    (["鞋柜", "玄关柜", "玄关收纳", "入户柜", "门厅柜"], "shoe cabinet"),
    (["榻榻米", "和室", "地台", "坐席"], "tatami"),
]


IGNORE_KEYWORDS = {
    "摆件", "摆饰", "装饰", "窗帘", "纱帘", "地毯", "抱枕", "画", "相框", "香薰", "绿植", "花瓶", "镜前灯", "壁灯", "浴霸"
}


LIGHT_LIKE_LABELS = {
    "ceiling lamp",
    "cove ceiling",
    "pendant lamp",
}

# Furniture types to ignore when generating prompts (should match items ignored in 2D conversion)
IGNORE_FURNITURE_LABELS = {
    "ceiling lamp",
    "cove ceiling",
}


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y + 1e-12) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def normalize_room_type(room_name: str) -> str:
    name = room_name or ""
    for keywords, room_type in ROOM_TYPE_RULES:
        if any(k in name for k in keywords):
            return room_type
    return "room"


def normalize_item_name(raw_name: str) -> Optional[str]:
    name = (raw_name or "").strip()
    if not name:
        return None

    if any(k in name for k in IGNORE_KEYWORDS):
        return None

    for keywords, label in FURNITURE_RULES_EN:
        if any(k in name for k in keywords):
            return label
    return None


def room_bbox(room_points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in room_points]
    ys = [p[1] for p in room_points]
    return min(xs), min(ys), max(xs), max(ys)


def furniture_footprint_bbox(furniture: Dict, fallback_xy: Tuple[float, float]) -> Tuple[float, float, float, float]:
    points = furniture.get("points", []) or []
    xy: List[Tuple[float, float]] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        if "x" not in p or "y" not in p:
            continue
        xy.append((float(p["x"]), float(p["y"])))

    if not xy:
        fx, fy = fallback_xy
        return fx, fy, fx, fy

    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    return min(xs), min(ys), max(xs), max(ys)


def position_phrase_from_bbox(
    item_bbox: Tuple[float, float, float, float],
    room_bbox_: Tuple[float, float, float, float],
) -> str:
    room_min_x, room_min_y, room_max_x, room_max_y = room_bbox_
    room_w = max(room_max_x - room_min_x, 1e-6)
    room_h = max(room_max_y - room_min_y, 1e-6)

    item_min_x, item_min_y, item_max_x, item_max_y = item_bbox
    item_w = max(item_max_x - item_min_x, 1e-6)
    item_h = max(item_max_y - item_min_y, 1e-6)

    d_left = max(item_min_x - room_min_x, 0.0)
    d_right = max(room_max_x - item_max_x, 0.0)
    d_top = max(item_min_y - room_min_y, 0.0)
    d_bottom = max(room_max_y - item_max_y, 0.0)

    near_x_t = max(0.12 * room_w, 0.65 * item_w)
    near_y_t = max(0.12 * room_h, 0.65 * item_h)

    near_left = d_left <= near_x_t
    near_right = d_right <= near_x_t
    near_top = d_top <= near_y_t
    near_bottom = d_bottom <= near_y_t

    center_x = (item_min_x + item_max_x) * 0.5
    center_y = (item_min_y + item_max_y) * 0.5
    nx = (center_x - room_min_x) / room_w
    ny = (center_y - room_min_y) / room_h

    if near_top and near_left and not near_bottom and not near_right:
        return "in the top-left corner"
    if near_top and near_right and not near_bottom and not near_left:
        return "in the top-right corner"
    if near_bottom and near_left and not near_top and not near_right:
        return "in the bottom-left corner"
    if near_bottom and near_right and not near_top and not near_left:
        return "in the bottom-right corner"

    if near_top and not near_bottom:
        return "at the top"
    if near_bottom and not near_top:
        return "at the bottom"
    if near_left and not near_right:
        return "on the left side"
    if near_right and not near_left:
        return "on the right side"

    if 0.34 <= nx <= 0.66 and 0.34 <= ny <= 0.66:
        return "in the center of the room"

    if ny < 0.33 and nx < 0.33:
        return "in the top-left corner"
    if ny < 0.33 and nx > 0.67:
        return "in the top-right corner"
    if ny > 0.67 and nx < 0.33:
        return "in the bottom-left corner"
    if ny > 0.67 and nx > 0.67:
        return "in the bottom-right corner"
    if ny < 0.33:
        return "at the top"
    if ny > 0.67:
        return "at the bottom"
    if nx < 0.33:
        return "on the left side"
    if nx > 0.67:
        return "on the right side"

    return "in the center of the room"


def parse_image_basename(image_path: str) -> Optional[Tuple[str, str, str]]:
    basename = os.path.basename(image_path)
    match = re.match(r"(\d+)_(\d+)_(.+?)_with_furn\.png", basename)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def build_prompt(room_name: str, item_labels: List[str]) -> str:
    prefix = "2D semantic segmentation map, top-down floorplan, simple solid color blocks."
    room_type = normalize_room_type(room_name)

    if not item_labels:
        return f"{prefix} room: {room_type}."

    # Remove duplicates while preserving order
    unique_labels = []
    seen = set()
    for label in item_labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)

    return f"{prefix} room: {room_type}, including: {', '.join(unique_labels)}."


def select_item_phrases(
    furniture_hits: List[Dict[str, str]],
    max_items: int,
    max_lights: int = -1,
    max_repeat_phrase: int = 999999,
) -> List[str]:
    """
    Pick item phrases with ordering constraints:
    1) keep non-light-like furniture first
    2) then append light-like furniture
    3) optional caps can still be applied via arguments
    """
    sorted_hits = sorted(furniture_hits, key=lambda h: (h["py"], h["px"], h["label"], h["phrase"]))
    non_light = [h for h in sorted_hits if h["label"] not in LIGHT_LIKE_LABELS]
    light_like = [h for h in sorted_hits if h["label"] in LIGHT_LIKE_LABELS]
    phrase_count: Dict[str, int] = defaultdict(int)

    ordered = non_light + light_like
    selected: List[str] = []

    light_taken = 0
    for hit in ordered:
        phrase = hit["phrase"]
        label = hit["label"]

        if max_repeat_phrase > 0 and phrase_count[phrase] >= max_repeat_phrase:
            continue

        if label in LIGHT_LIKE_LABELS and max_lights >= 0 and light_taken >= max_lights:
            continue

        selected.append(phrase)
        phrase_count[phrase] += 1
        if label in LIGHT_LIKE_LABELS:
            light_taken += 1

    if max_items > 0:
        return selected[:max_items]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate English prompts with item positions.")
    parser.add_argument("--input_csv", required=True, help="Input CSV path, e.g., metadata_kontext_train_v3.csv")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--plan_json_dir", required=True, help="Directory containing room_<plan_id>.json")
    parser.add_argument("--max_items", type=int, default=0, help="Max items kept in each prompt; 0 means keep all")
    parser.add_argument("--max_lights", type=int, default=-1, help="Max light-like items (ceiling/cove/pendant); -1 means keep all")
    parser.add_argument("--max_repeat_phrase", type=int, default=999999, help="Max repeat count for identical 'item + position' phrase")
    args = parser.parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    json_cache: Dict[str, Dict] = {}

    updated = 0
    skipped = 0

    for row in rows:
        parsed = parse_image_basename(row.get("image", ""))
        if not parsed:
            skipped += 1
            continue

        plan_id, room_id, room_name_from_file = parsed
        json_name = f"room_{plan_id}.json"
        json_path = os.path.join(args.plan_json_dir, json_name)

        if json_name not in json_cache:
            if not os.path.isfile(json_path):
                skipped += 1
                continue
            with open(json_path, "r", encoding="utf-8") as jf:
                json_cache[json_name] = json.load(jf)

        data = json_cache[json_name]

        room_data = None
        for room in data.get("roomData", []):
            if str(room.get("id")) == str(room_id):
                room_data = room
                break

        if not room_data:
            skipped += 1
            continue

        room_points = [(p["x"], p["y"]) for p in room_data.get("points", [])]
        if len(room_points) < 3:
            skipped += 1
            continue

        furniture_labels: List[str] = []
        for furn in data.get("furnitureData", []):
            pos = furn.get("pos", {})
            px = float(pos.get("x", 0))
            py = float(pos.get("y", 0))
            if not point_in_polygon((px, py), room_points):
                continue

            item_name = normalize_item_name(furn.get("name", ""))
            if not item_name:
                continue

            # Skip furniture types that are ignored in 2D conversion
            if item_name in IGNORE_FURNITURE_LABELS:
                continue

            furniture_labels.append(item_name)

        # Apply max_items limit if specified
        if args.max_items > 0:
            furniture_labels = furniture_labels[:args.max_items]

        room_name = room_data.get("name", "") or room_name_from_file
        row["prompt"] = build_prompt(room_name, furniture_labels)
        updated += 1

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote: {args.output_csv}")
    print(f"updated={updated}, skipped={skipped}, total={len(rows)}")

    print("\nSample prompts:")
    for sample in rows[:5]:
        print(f"- {sample.get('image', '')}: {sample.get('prompt', '')}")


if __name__ == "__main__":
    main()
