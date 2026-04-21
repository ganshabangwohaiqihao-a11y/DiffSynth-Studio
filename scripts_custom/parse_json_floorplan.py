#!/usr/bin/env python3
"""
Phase 0: JSON Floor Plan Parser
================================
Parses plan_json_0228/*.json and produces training-ready data for the adapted
SemLayoutDiff model.

Coordinate convention
---------------------
JSON coordinates are in cm (0.01 m).  The paper uses 1 px = 1 cm, so JSON
units map 1-to-1 to pixels — no conversion needed.

Outputs (written to --output_dir)
----------------------------------
    chinese_train_128x128.npy   shape (N, 2, 128, 128) uint8
    chinese_val_128x128.npy     shape (N, 2, 128, 128) uint8
    chinese_test_128x128.npy    shape (N, 2, 128, 128) uint8
  chinese_global_train_64x64.npy  shape (N, 64, 64) uint8
  chinese_global_val_64x64.npy
  chinese_global_test_64x64.npy
  chinese_meta_train.json / val / test   per-sample metadata (incl. prompt)
  chinese_train_splits.csv / val / test

Written to preprocess/metadata/
---------------------------------
  chinese_room_types.json          room-name → type-id
  chinese_furniture_categories.json  furniture-name → {cat_id, cat_name, count}
  chinese_idx_to_label.json        category-id → English label  (training config)

Usage
-----
  python parse_json_floorplan.py \
      --input_dir  C:/Users/紫燕/Desktop/plan_json_0228 \
      --output_dir C:/Users/紫燕/Desktop/SemLayoutDiff-main/datasets/chinese \
      --visualize      # optional: save sample PNGs for verification
"""
# python /home/chengjiajia/diffusers/scripts_custom/parse_json_floorplan.py \
#     --room_h 1024 \
#     --room_w 1024 \
#     --visualize
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Category Definitions
# ──────────────────────────────────────────────────────────────────────────────

#: Unified category index → English label.
#: IDs 0-37 mirror the original SemLayoutDiff unified set.
#: IDs 38+ are Chinese-specific extensions.
CATEGORIES: Dict[int, str] = {
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
    # ── Chinese-specific extensions ────────────────────────────────────────
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
    # ── Lighting extensions ────────────────────────────────────────────────
    54: "cove_ceiling",     # 吊顶（石膏线、天花造型、灯槽）
}

NUM_CATEGORIES = len(CATEGORIES)   # 55 total

# Semantic IDs for arch elements
VOID_ID   = 0
FLOOR_ID  = 1
DOOR_ID   = 36
WINDOW_ID = 37

# Hole type string → semantic ID
HOLE_TYPE_MAP: Dict[str, int] = {
    "门":    DOOR_ID,
    "推拉门": DOOR_ID,
    "折叠门": DOOR_ID,
    "洞":    DOOR_ID,   # open passage
    "窗":    WINDOW_ID,
    "窗户":  WINDOW_ID,
    "飘窗":  WINDOW_ID,
}

# ── 渲染层级（小值先画，大值后画 = 优先级更高）──────────────────────────────
# - 34/35/46/54 需要保留，但不能压住地面家具
_TOP_IMPORTANT_CAT_IDS = {
    34,  # pendant_lamp (吊灯)
    35,  # ceiling_lamp（吸顶灯/筒灯/射灯）
    46,  # air_conditioner（通常挂墙/吊顶）
    54,  # cove_ceiling（吊顶/石膏线/天花造型）
}


def _render_priority(cat_id: int) -> int:
    """返回渲染优先级：0=普通，1=顶部重要类。"""
    if cat_id in _TOP_IMPORTANT_CAT_IDS:
        return 1
    return 0

# ──────────────────────────────────────────────────────────────────────────────
# Rule-based Furniture Matcher
# Each entry: (keyword_list, category_id)
# Rules are tried in order; first keyword match wins.
# More specific patterns come before generic ones.
# ──────────────────────────────────────────────────────────────────────────────
FURNITURE_RULES: List[Tuple[List[str], int]] = [
    # ── Beds ────────────────────────────────────────────────────────────────
    # IMPORTANT: More specific bed-related items MUST come before generic "床" rule
    (["床头柜", "床边柜", "床头桌", "床头台"], 27),  # nightstand - specific
    (["儿童床", "婴儿床", "上下床", "双层床", "高低床", "bunk"], 2),
    (["单人床", "1.0米", "0.9米", "单床"], 3),
    (["双人床", "大床", "1.5米", "1.6米", "1.8米", "2.0米", "主卧床", "床垫",
      "床架", "床板", "软床", "硬床", "箱体床", "榻榻米床"], 4),
    (["床"], 4),                           # fallback bed → double bed
    # ── Tables ──────────────────────────────────────────────────────────────
    (["角几", "边几", "角桌", "角台", "小边柜"], 5),
    (["圆形茶几", "圆几", "圆形小桌"], 6),
    (["茶几", "咖啡桌", "客厅桌"], 7),
    (["玄关台", "玄关桌", "端景台", "条案", "条几"], 8),
    (["电视柜", "TV柜", "视听柜", "电视架", "电视台"], 9),
    (["书桌", "写字台", "工作台", "办公桌", "电脑桌", "学习桌", "电脑台",
      "工作桌", "工作站", "学习台"], 10),
    (["梳妆台", "梳妆", "化妆台", "化妆桌", "梳妆柜"], 11),
    (["餐桌", "饭桌", "用餐桌", "吃饭桌"], 13),
    (["桌", "台面", "桌案"], 12),           # fallback table
    # ── Seating ─────────────────────────────────────────────────────────────
    (["换鞋凳", "矮凳", "凳", "脚凳", "踏脚凳", "茶几凳", "圆凳", "方凳"], 14),
    (["梳妆椅", "化妆椅"], 15),
    (["餐椅", "饭椅", "靠背椅"], 16),
    (["中式椅", "古典椅", "明式椅", "太师椅", "圈椅", "官帽椅"], 17),
    (["扶手椅", "休闲单椅", "单椅", "主人椅", "书椅", "布艺椅"], 18),
    (["躺椅", "懒人椅", "休闲椅"], 20),
    (["双人沙发", "二人沙发", "两人沙发"], 21),
    (["懒人沙发", "豆袋", "蛋形沙发", "懒沙发"], 22),
    (["L型沙发", "L形沙发", "转角沙发", "拐角沙发", "U型沙发", "U形沙发"], 26),
    (["贵妃", "贵妃椅", "贵妃沙发", "chaise"], 25),
    (["多人沙发", "多位沙发", "四人沙发", "五人沙发"], 24),
    (["三人沙发", "三座沙发", "布艺沙发", "皮艺沙发", "皮沙发", "布沙发"], 23),
    (["沙发"], 23),                        # fallback sofa
    (["椅"], 19),                          # fallback chair
    # ── Storage ─────────────────────────────────────────────────────────────
    (["书架", "书柜", "文件架", "杂志架", "书报架"], 29),
    (["酒柜", "红酒柜"], 31),
    (["衣柜", "大衣柜", "走入式衣", "步入式衣", "衣帽柜", "移门衣柜",
      "推拉门衣柜", "嵌入式衣柜"], 33),
    (["置物架", "搁板", "层板架", "展示架", "陈列架", "花架", "储物架"], 28),
    (["儿童柜", "玩具柜", "儿童收纳箱"], 30),
    # ── Lamps ───────────────────────────────────────────────────────────────
    # 吊灯 (pendant) — most specific first
    (["吊灯", "垂吊灯", "悬挂灯", "吊线灯", "铁艺吊灯",
      "北欧吊灯", "餐厅吊灯", "饭厅吊灯"], 34),
    # 吊顶 / 顶面造型 / 灯槽 / 灯带 → cove_ceiling (cat 54)
    (["吊顶", "石膏线", "顶面", "天花板", "天花造型",
      "灯槽", "顶角", "造型顶", "集成吊顶", "灯带", "线性灯", "灯条"], 54),
    # 落地灯 / 台灯 / 床头灯 → 装饰品，映射为 void（不学习生成）
    (["落地灯", "台灯", "床头灯", "小夜灯", "阅读灯",
      "立式灯", "护眼灯"], VOID_ID),
    # 其余小型照明/设备仍视作装饰或噪声，映射为 void
    (["壁灯", "感应灯", "浴霸", "排气扇", "镜前灯", "扫地机"], VOID_ID),
    # 天花主灯 / 点光源 → ceiling_lamp (cat 35)（参与训练）
    # 注意：放在 VOID 灯规则之后，避免“落地LED灯”一类混合词被误归到 35。
    (["吸顶灯", "射灯", "筒灯", "格栅灯", "主灯", "厨卫灯"], 35),
    # ── Bathroom ────────────────────────────────────────────────────────────
    (["浴缸", "泡澡盆", "按摩浴缸", "浴盆", "独立浴缸"], 38),
    (["马桶", "坐便器", "蹲便器", "智能马桶", "坐厕"], 39),
    (["洗手台", "洗脸盆", "洗手盆", "洗漱台", "面盆", "洗面台",
      "台盆", "洗脸台", "盥洗台"], 40),
    (["淋浴", "花洒", "淋浴区", "淋浴房", "沐浴", "淋浴隔断"], 41),
    (["浴室柜", "卫浴柜", "镜柜", "洗手间柜", "卫生间柜"], 49),
    # ── Kitchen ─────────────────────────────────────────────────────────────
    (["冰箱", "冰柜", "冷藏柜", "冰粮", "小冰箱", "双门冰箱"], 42),
    (["灶台", "油烟机", "烟机", "蒸烤箱", "微波炉", "洗碗机",
      "蒸箱", "烤箱", "燃气灶", "电磁炉", "集成灶"], 43),
    (["水槽", "洗菜盆", "厨房水盆", "水盆台"], 43),
    (["橱柜", "厨柜", "整体厨房", "厨房柜", "吊柜", "地柜", "厨房收纳"], 48),
    (["岛台", "西厨岛", "厨房岛台", "中岛台", "岛形台"], 53),
    # ── Electronics / Entertainment ─────────────────────────────────────────
    (["钢琴", "三角钢琴", "立式钢琴", "电钢琴", "电子琴", "数码钢琴"], 44),
    (["电视",  "电视机"], 47),
    (["跑步机", "健身单车", "划船机", "哑铃台", "健身器", "运动器材", "椭圆机"], 50),
    # ── Laundry ─────────────────────────────────────────────────────────────
    (["洗衣机", "烘干机", "洗烘一体机", "洗烘"], 45),
    # ── Climate ─────────────────────────────────────────────────────────────
    (["空调", "中央空调", "空调挂机", "空调柜机", "立式空调", "暖气"], 46),
    # ── Entrance ────────────────────────────────────────────────────────────
    (["鞋柜", "玄关柜", "玄关收纳", "入户柜", "门厅柜"], 51),
    # ── Decorative / soft furnishings (map to generic cabinet, not furniture) ─
    (["窗帘", "纱帘", "帘子", "帘布", "布帘", "遮光帘"], VOID_ID),
    (["摆件", "摆饰", "工艺品", "装饰品", "书籍", "绿植", "花瓶",
      "器皿", "香薰", "鹅卵石", "酒-", "洋酒", "插座", "面板",
      "开关", "出风口", "格栅板", "玻璃隔断", "玻璃门", "踢脚线",
      "地毯", "画框", "挂画", "壁画", "飘窗垫", "窗台垫"], VOID_ID),
    # ── Asian specific ──────────────────────────────────────────────────────
    (["榻榻米", "和室", "地台", "坐席"], 52),
    # ── Generic fallbacks (must be last) ────────────────────────────────────
    (["柜", "收纳", "储物", "cabinet"], 32),
]

# ──────────────────────────────────────────────────────────────────────────────
# Room Type Rules
# Longer / more specific strings come first.
# ──────────────────────────────────────────────────────────────────────────────

#: room-type-id → English label (for reference)
# 主卧+次卧 合并为 bedroom；主卫+次卫 合并为 bathroom
ROOM_TYPES: Dict[int, str] = {
    0:  "bedroom",            # 主卧 + 次卧 合并（论文 type 0）
    1:  "dining_room",        # 论文 type 1，ID 对齐预训练权重
    2:  "living_room",        # 论文 type 2，ID 对齐预训练权重
    3:  "living_dining_room", # 中文客餐厅（独立类别，随机初始化）
    4:  "kitchen",
    5:  "bathroom",           # 主卫 + 次卫 合并
    6:  "study",
    7:  "balcony",
    8:  "entrance",
    9:  "storage_room",
    10: "entertainment_room",
    11: "gym",
    12: "piano_room",
    13: "tea_room",
    14: "garage",
    15: "corridor",
}
NUM_ROOM_TYPES = len(ROOM_TYPES)  # 16

ROOM_TYPE_RULES: List[Tuple[List[str], int]] = [
    # ── 卧室：主卧和次卧合并为同一类型 ────────────────────────────────────
    (["主卧", "主人房", "主人卧室",
      "次卧", "客卧", "儿童房", "小孩房", "保姆房", "女儿房",
      "男儿房", "次卧室", "儿童卧室", "长辈房", "老人房",
      "男孩房", "女孩房", "小女儿房", "大女儿房", "卧室"], 0),
    (["客餐厅", "客厅餐厅", "起居餐厅", "起居/餐厅", "大客餐厅", "客卧一体"], 3),  # living_dining_room
    (["客厅", "起居室", "会客厅", "大厅", "厅房"], 2),              # living_room
    (["餐厅", "餐区", "用餐区", "饭厅", "就餐区"], 1),              # dining_room
    (["厨房", "西厨", "中厨", "开放式厨房", "封闭厨房", "厨卫", "厨餐"], 4),
    # ── 卫生间：主卫和次卫合并为同一类型 ──────────────────────────────────
    (["主卫", "主人卫生间", "主浴室",
      "卫生间", "洗手间", "公卫", "次卫", "浴室", "盥洗室", "卫浴",
      "厕所", "干区", "湿区", "洗衣房", "客卫", "公共卫生间", "共用卫生间"], 5),
    (["书房", "工作室", "多功能间", "多功能室"], 6),
    (["阳台", "生活阳台", "景观阳台", "服务阳台", "露台", "花园阳台",
      "小阳台", "花园", "入户花园", "建筑景观"], 7),
    (["玄关", "门厅", "入户区", "入口区", "过厅", "入户"], 8),
    (["衣帽间", "步入式衣帽间", "储物间", "储藏间", "储藏室", "杂物间"], 9),
    (["影音室", "媒体室", "娱乐室", "家庭影院", "视听室", "视听间"], 10),
    (["健身房", "健身区", "运动室", "运动房"], 11),
    (["钢琴房", "音乐室", "练琴室", "琴房"], 12),
    (["茶室", "茶房", "品茗区", "茶厅"], 13),
    (["车库", "停车位", "地下室", "车位"], 14),
    (["过道", "走廊", "通道", "走道", "廊道", "电梯", "消防梯"], 15),
    (["榻榻米"], 6),   # 榻榻米房 → study/flex
]

# Rooms matching these keywords are dropped from dataset generation.
# This is a data-cleaning policy to remove ambiguous bucket names.
DROP_ROOM_NAME_KEYWORDS: Tuple[str, ...] = (
    "其他空间",
)


# ──────────────────────────────────────────────────────────────────────────────
# Rule Lookup Helpers
# ──────────────────────────────────────────────────────────────────────────────

# 强制映射为 VOID 的黑名单（优先于通用规则）
_FORCE_VOID_KEYWORDS: Tuple[str, ...] = (
    # 软装小物
    "杯具", "花瓶", "摆件", "包包", "食物储藏罐",
    # 小五金/配件
    "挂钩", "挂架", "毛巾架", "插座", "开关",
    # 易噪声照明
    "壁灯", "镜前灯", "感应灯", "浴霸",
    # 非目标小电器
    "风筒", "扫地机", "饮水机",
    # 衣柜内部软装/配饰，容易在同一区域大量重叠，作为训练噪声忽略
    "衣柜装饰", "服饰装饰", "鞋包衣帽", "衣架衣服", "折叠衣物", "行李箱",
    # 通用软装/装饰品：统一忽略，不参与训练目标
    "装饰", "摆件", "摆饰", "装饰品", "工艺品", "陈设", "饰品",
    "挂画", "画框", "壁画", "装饰画", "相框",
    "地毯", "抱枕", "靠垫", "桌旗", "桌布",
    "窗帘", "纱帘", "帘布", "遮光帘",
    "书籍", "绿植", "盆栽", "花瓶", "花艺", "假花", "香薰", "器皿",
    "玩偶", "玩具", "蜡烛", "镜子",
    "衣架", "衣服", "鞋包", "衣帽",
)


def _resolve_light_conflict(name: str) -> Optional[int]:
    """灯具冲突词优先级判定（用于“灯具/灯饰/灯”等泛词场景）。

    优先级：
    1) 同时出现 浴霸/壁灯/镜前灯 -> VOID
    2) 同时出现 吊 -> 34
    3) 同时出现 灯带/灯槽 -> 54
    4) 同时出现 吸顶/筒/射/格栅/轨道 -> 35
    """
    n = (name or "").strip()
    if not n:
        return None

    # 仅在灯具语义上下文中触发冲突判定
    if ("灯" not in n) and ("照明" not in n):
        return None

    if any(k in n for k in ("浴霸", "壁灯", "镜前灯")):
        return VOID_ID
    if "吊" in n:
        return 34
    if any(k in n for k in ("灯带", "灯槽")):
        return 54
    if any(k in n for k in ("吸顶", "筒", "射", "格栅", "轨道")):
        return 35
    return None


def lookup_furniture_category(name: str) -> int:
    """Map a Chinese furniture product name to a unified category ID."""
    n = (name or "").strip()
    if not n:
        return VOID_ID

    # 1) 黑名单强制 VOID
    if any(k in n for k in _FORCE_VOID_KEYWORDS):
        return VOID_ID

    # 2) 灯具冲突优先级判定
    light_cat = _resolve_light_conflict(n)
    if light_cat is not None:
        return light_cat

    # 3) 常规 first-match 规则
    for keywords, cat_id in FURNITURE_RULES:
        for kw in keywords:
            if kw in n:
                return cat_id
    return VOID_ID


def lookup_room_type(name: str) -> int:
    """Map a Chinese room name to a room-type ID."""
    for keywords, type_id in ROOM_TYPE_RULES:
        for kw in keywords:
            if kw in name:
                return type_id
    return 3  # default: living_dining_room


def should_drop_room(name: str) -> bool:
    """Return True if the room name should be removed from the dataset."""
    n = (name or "").strip()
    if not n:
        return True
    return any(kw in n for kw in DROP_ROOM_NAME_KEYWORDS)


# ──────────────────────────────────────────────────────────────────────────────
# JSON Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_floorplan(json_path: str) -> Dict[str, Any]:
    """
    Parse a floor plan JSON file into structured Python objects.

    Returns
    -------
    dict with keys:
        canvas_size : [W, H] in cm
        level_height: int (cm)
        rooms       : list of room dicts
        walls       : dict  wall_id → wall dict
        holes       : dict  hole_id → hole dict
        furniture   : list of furniture dicts
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    canvas_size: List[float] = data.get("size", [1200, 1200])
    level_height: int = data.get("levelHeight", 280)

    # ── Holes ─────────────────────────────────────────────────────────────
    holes: Dict[str, Dict] = {}
    for h in data.get("holeData", []):
        holes[h["id"]] = {
            "id": h["id"],
            "points": [(p["x"], p["y"]) for p in h.get("points", [])],
            "type": h.get("type", "门"),
            "wall_id": h.get("wallId", ""),
            "room_ids": h.get("roomIds", []),
            "ground_height": h.get("groundHeight", 0),
        }

    # ── Walls ─────────────────────────────────────────────────────────────
    walls: Dict[str, Dict] = {}
    for w in data.get("wallData", []):
        walls[w["id"]] = {
            "id": w["id"],
            "points": [(p["x"], p["y"]) for p in w.get("points", [])],
            "thickness": w.get("thickness", 20),
            "height": w.get("height", level_height),
            "hole_ids": w.get("holeIds", []),
            "room_ids": w.get("roomIds", []),
        }

    # ── Rooms ─────────────────────────────────────────────────────────────
    rooms: List[Dict] = []
    for r in data.get("roomData", []):
        rooms.append({
            "id": r["id"],
            "name": r.get("name", ""),
            "points": [(p["x"], p["y"]) for p in r.get("points", [])],
            "wall_ids": r.get("wallIds", []),
        })

    # ── Furniture ─────────────────────────────────────────────────────────
    furniture: List[Dict] = []
    for item in data.get("furnitureData", []):
        pts_2d = [(p["x"], p["y"]) for p in item.get("points", [])]
        pos = item.get("pos", {"x": 0, "y": 0, "z": 0})
        furniture.append({
            "id": item["id"],
            "model_id": item.get("modelId", ""),
            "name": item.get("name", ""),
            "size": item.get("size", [0, 0, 0]),
            "pos": (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)),
            "points_2d": pts_2d,
        })

    return {
        "canvas_size": canvas_size,
        "level_height": level_height,
        "rooms": rooms,
        "walls": walls,
        "holes": holes,
        "furniture": furniture,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Point-in-Polygon (pure Python, no extra dependencies)
# ──────────────────────────────────────────────────────────────────────────────

def _point_in_polygon(px: float, py: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_area(pts: List[Tuple[float, float]]) -> float:
    """Shoelace formula for signed polygon area."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def assign_furniture_to_rooms(
    furniture: List[Dict],
    rooms: List[Dict],
) -> Dict[str, List[Dict]]:
    """
    Assign each furniture item to the room it belongs to.
    Uses the furniture centroid (pos) and a point-in-polygon test.

    Returns
    -------
    dict  room_id → list of furniture dicts
    """
    # Pre-compute room polygons (filter degenerate rooms)
    room_polys: List[Tuple[str, List[Tuple[float, float]]]] = []
    for room in rooms:
        pts = room["points"]
        if len(pts) >= 3 and _polygon_area(pts) > 100:  # >100 cm² = 0.01 m²
            room_polys.append((room["id"], pts))

    result: Dict[str, List[Dict]] = defaultdict(list)

    for item in furniture:
        px, py, _ = item["pos"]

        # Test each room
        assigned: Optional[str] = None
        for room_id, poly in room_polys:
            if _point_in_polygon(px, py, poly):
                assigned = room_id
                break

        # Fallback: assign to closest room centroid
        if assigned is None and room_polys:
            best_dist = float("inf")
            for room_id, poly in room_polys:
                cx = sum(p[0] for p in poly) / len(poly)
                cy = sum(p[1] for p in poly) / len(poly)
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < best_dist:
                    best_dist = d
                    assigned = room_id

        if assigned is not None:
            result[assigned].append(item)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Rendering Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_int32(pts: List[Tuple[float, float]]) -> np.ndarray:
    """Convert list of (x, y) to int32 array suitable for cv2.fillPoly."""
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _footprint_hull(pts_2d: List[Tuple[float, float]]) -> Optional[np.ndarray]:
    """
    Compute the 2-D convex hull of the furniture's 3-D bounding box vertices
    projected onto the XY-plane.  Returns an int32 array for cv2, or None.
    """
    if len(pts_2d) < 3:
        return None
    arr = np.array(pts_2d, dtype=np.float32)
    hull = cv2.convexHull(arr.reshape(-1, 1, 2))
    return hull.astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Global Context Mask Renderer (64 × 64)
# ──────────────────────────────────────────────────────────────────────────────

def render_global_mask(
    data: Dict[str, Any],
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """
    Render a low-resolution whole-apartment context mask.

    Values
    ------
    0 = void
    1 = floor (any room interior)
    2 = door
    3 = window

    Returns
    -------
    np.ndarray of shape (H, W) uint8
    """
    cw, ch = data["canvas_size"]
    W, H = max(int(cw), 1), max(int(ch), 1)
    canvas = np.zeros((H, W), dtype=np.uint8)

    # Fill each room polygon as floor
    for room in data["rooms"]:
        pts = room["points"]
        if len(pts) >= 3:
            cv2.fillPoly(canvas, [_to_int32(pts)], FLOOR_ID)

    # Overlay holes (doors / windows)
    for hole in data["holes"].values():
        pts = hole["points"]
        if len(pts) >= 3:
            cat = HOLE_TYPE_MAP.get(hole["type"], DOOR_ID)
            cv2.fillPoly(canvas, [_to_int32(pts)], cat)

    # Resize to target resolution
    GW, GH = target_size
    return cv2.resize(canvas, (GW, GH), interpolation=cv2.INTER_NEAREST)


# ──────────────────────────────────────────────────────────────────────────────
# Per-Room Semantic Map Renderer (128 × 128)
# ──────────────────────────────────────────────────────────────────────────────

def render_room_semantic_map(
    room: Dict[str, Any],
    holes: Dict[str, Dict],
    room_furniture: List[Dict],
    target_size: Tuple[int, int] = (128, 128),
    include_furniture: bool = True,
) -> Optional[Tuple[np.ndarray, float, float, List[Tuple[float, float]], List[float]]]:
    """
    Render a per-room semantic layout map at 1 px = 1 cm scale.

    Pixel values are unified category IDs:
        0          → void (outside room or padding)
        1          → floor
        2-35       → furniture categories (original paper set)
        36         → door
        37         → window
        38-53      → Chinese-specific furniture categories

    Parameters
    ----------
    include_furniture : bool, default True
        If False, only render floor + doors/windows (no furniture).
        Useful for generating condition maps for conditional generation.

    Returns
    -------
    Tuple[np.ndarray, float, float, List[Tuple[float, float]], List[float]] or None if the room is degenerate.
        - np.ndarray of shape (SH, SW) uint8
        - room_width_cm : float (actual room width in cm)
        - room_height_cm : float (actual room height in cm)
        - room_polygon_padded : room polygon vertices in padded semantic-map coordinates
        - edge_lengths_cm : per-edge lengths in cm, following polygon vertex order
    """
    pts = room["points"]
    if len(pts) < 3:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    W_room = x_max - x_min
    H_room = y_max - y_min

    # Skip rooms smaller than 10 cm in any dimension (degenerate)
    if W_room < 10 or H_room < 10:
        return None

    SH, SW = target_size
    # Scale down only if room exceeds target canvas
    scale = min(SW / W_room, SH / H_room)
    canvas_w = max(int(np.ceil(W_room * scale)), 1)
    canvas_h = max(int(np.ceil(H_room * scale)), 1)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    def tx(pts_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Translate → scale to local canvas coordinates."""
        return [((p[0] - x_min) * scale, (p[1] - y_min) * scale) for p in pts_list]

    # ── 1. Fill room polygon with floor ───────────────────────────────────
    room_local = tx(pts)
    cv2.fillPoly(canvas, [_to_int32(room_local)], FLOOR_ID)

    # ── 2. Overlay furniture footprints
    # 顶部重要类(34/35/46/54)采用“非覆盖写入”：只在 floor 像素落笔，避免盖住其他家具。
    sorted_furniture = sorted(
        room_furniture,
        key=lambda item: _render_priority(lookup_furniture_category(item["name"]))
    )
    for item in sorted_furniture:
        cat_id = lookup_furniture_category(item["name"])

        # Ignore items mapped to VOID_ID, ceiling_lamp (35), and cove_ceiling (54)
        if cat_id == VOID_ID or cat_id in (35, 54):
            continue

        fp_pts = item["points_2d"]
        hull = _footprint_hull(tx(fp_pts)) if len(fp_pts) >= 3 else None

        if hull is None:
            # Fallback: axis-aligned bounding box from size / pos
            px, py, _ = item["pos"]
            sx = item["size"][0] / 2 if len(item["size"]) > 0 else 20.0
            sy = item["size"][1] / 2 if len(item["size"]) > 1 else 20.0
            fallback_pts = [
                (px - sx, py - sy),
                (px + sx, py - sy),
                (px + sx, py + sy),
                (px - sx, py + sy),
            ]
            hull = _to_int32(tx(fallback_pts))

        if cat_id in _TOP_IMPORTANT_CAT_IDS:
            obj_mask = np.zeros_like(canvas, dtype=np.uint8)
            cv2.fillPoly(obj_mask, [hull], 1)
            write_mask = (obj_mask == 1) & (canvas == FLOOR_ID)
            canvas[write_mask] = cat_id
        else:
            cv2.fillPoly(canvas, [hull], cat_id)

    # ── 3. Overlay holes (doors / windows) — always on top ────────────────
    for hole in holes.values():
        if room["id"] not in hole.get("room_ids", []):
            continue
        hole_pts = hole["points"]
        if len(hole_pts) < 3:
            continue
        cat = HOLE_TYPE_MAP.get(hole["type"], DOOR_ID)
        cv2.fillPoly(canvas, [_to_int32(tx(hole_pts))], cat)

    # ── 4. Pad to target_size (centered, void=0 padding) ─────────────────
    pad_h = SH - canvas_h
    pad_w = SW - canvas_w
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    padded_canvas = np.pad(
        canvas,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=VOID_ID,
    )

    room_polygon_padded = [(float(x + pad_left), float(y + pad_top)) for (x, y) in room_local]
    edge_lengths_cm: List[float] = []
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        edge_lengths_cm.append(float(np.hypot(x2 - x1, y2 - y1)))
    
    # Return semantic map + room dimensions + per-edge geometry for visualization.
    return padded_canvas, W_room, H_room, room_polygon_padded, edge_lengths_cm


# ──────────────────────────────────────────────────────────────────────────────
# Name Collection (First Pass)
# ──────────────────────────────────────────────────────────────────────────────

def collect_all_names(
    json_paths: List[str],
) -> Tuple[Counter, Counter]:
    """
    Read every JSON and collect room / furniture name frequencies.
    Returns (room_name_counter, furniture_name_counter).
    """
    room_cnt: Counter = Counter()
    furn_cnt: Counter = Counter()

    for path in tqdm(json_paths, desc="Pass 1 – collecting names"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("roomData", []):
                name = r.get("name", "").strip()
                if name and not should_drop_room(name):
                    room_cnt[name] += 1
            for item in data.get("furnitureData", []):
                name = item.get("name", "").strip()
                if name:
                    furn_cnt[name] += 1
        except Exception as exc:
            print(f"  [WARN] {path}: {exc}")

    return room_cnt, furn_cnt


# ──────────────────────────────────────────────────────────────────────────────
# Mapping Builders
# ──────────────────────────────────────────────────────────────────────────────

def build_room_type_mapping(room_cnt: Counter) -> Dict[str, int]:
    """Chinese room name → integer type-id."""
    return {name: lookup_room_type(name) for name in sorted(room_cnt)}


def build_furniture_category_mapping(furn_cnt: Counter) -> Dict[str, Dict]:
    """Chinese furniture name → {category_id, category_name, count}."""
    result = {}
    for name in sorted(furn_cnt, key=lambda x: -furn_cnt[x]):
        cat_id = lookup_furniture_category(name)
        result[name] = {
            "category_id": cat_id,
            "category_name": CATEGORIES.get(cat_id, "unknown"),
            "count": furn_cnt[name],
        }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation Helper
# ──────────────────────────────────────────────────────────────────────────────

_VIZ_COLORMAP: Dict[int, Tuple[int, int, int]] = {
    VOID_ID:   (255, 255, 255),
    FLOOR_ID:  (200, 200, 200),
    DOOR_ID:   (255, 120,  0),
    WINDOW_ID: (0,  180, 255),
    # Sample furniture colours
    2: (200,  80,  80), 3: (200,  80,  80), 4: (200,  80,  80),
    10: (80, 140, 200), 13: (100, 200, 100), 23: (200, 130,  50),
    33: (160,  60, 160), 39: (80, 200, 200), 42: (200, 200,  60),
}

def _cat_to_color(cat_id: int) -> Tuple[int, int, int]:
    if cat_id in _VIZ_COLORMAP:
        return _VIZ_COLORMAP[cat_id]
    rng = np.random.RandomState(cat_id * 17 + 3)
    return tuple(int(x) for x in rng.randint(60, 230, 3))


def save_visualization(
    semantic_map: np.ndarray,
    out_path: str,
) -> None:
    """Save a colour-coded PNG of a semantic map for visual inspection.

    Uses PIL instead of cv2 to correctly handle Unicode file paths on Windows.
    """
    from PIL import Image as PILImage

    h, w = semantic_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id in np.unique(semantic_map):
        mask = semantic_map == cat_id
        rgb[mask] = _cat_to_color(int(cat_id))
    PILImage.fromarray(rgb, mode="RGB").save(out_path)


def save_visualization_with_dimensions(
    semantic_map: np.ndarray,
    room_width_cm: float,
    room_height_cm: float,
    room_polygon_px: List[Tuple[float, float]],
    edge_lengths_cm: List[float],
    out_path: str,
    add_border: bool = False,
    border_size: int = 60,
) -> None:
    """Save a colour-coded PNG with per-edge room dimensions.
    
    This function annotates each room-edge length (in cm) near the corresponding edge.
    Optionally adds white margins around the semantic map.
    
    Parameters
    ----------
    semantic_map : np.ndarray
        Shape (H, W) uint8 semantic map
    room_width_cm : float
        Room width in centimeters (from JSON)
    room_height_cm : float
        Room height in centimeters (from JSON)
    room_polygon_px : List[Tuple[float, float]]
        Room polygon vertices in semantic-map pixel coordinates.
    edge_lengths_cm : List[float]
        Per-edge lengths in centimeters, aligned with polygon vertex order.
    out_path : str
        Output PNG file path
    add_border : bool
        If True, add white border around the image. If False, keep original size.
    border_size : int
        Size of the border in pixels (default: 60, only used if add_border=True)
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont
    
    h, w = semantic_map.shape
    
    # Convert semantic map to RGB
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id in np.unique(semantic_map):
        mask = semantic_map == cat_id
        rgb[mask] = _cat_to_color(int(cat_id))
    
    # Create image with optional white border
    if add_border:
        img_h = h + 2 * border_size
        img_w = w + 2 * border_size
        bordered = PILImage.new("RGB", (img_w, img_h), color=(255, 255, 255))
        bordered.paste(PILImage.fromarray(rgb, mode="RGB"), (border_size, border_size))
        offset = border_size
    else:
        bordered = PILImage.fromarray(rgb, mode="RGB")
        offset = 0
    
    # Draw dimension annotations
    draw = ImageDraw.Draw(bordered)
    
    # Try to use a reasonable font size
    try:
        font_size = 14
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text_color = (60, 60, 60)
    poly = [(x + offset, y + offset) for (x, y) in room_polygon_px]
    if len(poly) >= 3 and len(edge_lengths_cm) == len(poly):
        # Merge consecutive collinear edges so one straight side gets one total length.
        merged_edges: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
        collinear_eps = 0.015
        for i in range(len(poly)):
            s = poly[i]
            e = poly[(i + 1) % len(poly)]
            seg_len = float(edge_lengths_cm[i])
            if not merged_edges:
                merged_edges.append((s, e, seg_len))
                continue

            ps, pe, plen = merged_edges[-1]
            v1x, v1y = pe[0] - ps[0], pe[1] - ps[1]
            v2x, v2y = e[0] - s[0], e[1] - s[1]
            n1 = float(np.hypot(v1x, v1y))
            n2 = float(np.hypot(v2x, v2y))

            # Merge if contiguous and nearly collinear with same direction.
            contiguous = abs(pe[0] - s[0]) <= 1e-3 and abs(pe[1] - s[1]) <= 1e-3
            if contiguous and n1 > 1e-6 and n2 > 1e-6:
                cross = abs(v1x * v2y - v1y * v2x) / (n1 * n2)
                dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
                if cross <= collinear_eps and dot > 0.98:
                    merged_edges[-1] = (ps, e, plen + seg_len)
                    continue

            merged_edges.append((s, e, seg_len))

        # First and last may still be mergeable because polygon is closed.
        if len(merged_edges) >= 2:
            fs, fe, flen = merged_edges[0]
            ls, le, llen = merged_edges[-1]
            v1x, v1y = le[0] - ls[0], le[1] - ls[1]
            v2x, v2y = fe[0] - fs[0], fe[1] - fs[1]
            n1 = float(np.hypot(v1x, v1y))
            n2 = float(np.hypot(v2x, v2y))
            contiguous = abs(le[0] - fs[0]) <= 1e-3 and abs(le[1] - fs[1]) <= 1e-3
            if contiguous and n1 > 1e-6 and n2 > 1e-6:
                cross = abs(v1x * v2y - v1y * v2x) / (n1 * n2)
                dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
                if cross <= collinear_eps and dot > 0.98:
                    merged_edges[0] = (ls, fe, llen + flen)
                    merged_edges.pop()

        cx = float(sum(x for x, _ in poly) / len(poly))
        cy = float(sum(y for _, y in poly) / len(poly))
        offset_px = max(10, border_size // 6)

        for s, e, seg_len in merged_edges:
            x1, y1 = s
            x2, y2 = e
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0

            dx = x2 - x1
            dy = y2 - y1
            norm = float(np.hypot(dx, dy))
            if norm < 1e-6:
                continue

            nx, ny = -dy / norm, dx / norm
            # Choose inward normal: towards polygon centroid (inside the room).
            to_center_x = cx - mx
            to_center_y = cy - my
            if nx * to_center_x + ny * to_center_y < 0:
                nx, ny = -nx, -ny

            lx = mx + nx * offset_px
            ly = my + ny * offset_px
            label = f"{seg_len:.0f}cm"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = int(round(lx - tw / 2))
            ty = int(round(ly - th / 2))
            
            # Clamp text position to keep it fully within image bounds
            img_width, img_height = bordered.size
            tx = max(0, min(tx, img_width - tw))
            ty = max(0, min(ty, img_height - th))
            
            draw.text((tx, ty), label, fill=text_color, font=font)
    
    bordered.save(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────




def process_all(
    input_dir: str,
    output_dir: str,
    room_size:   Tuple[int, int] = (128, 128),
    global_size: Tuple[int, int] = (64, 64),
    train_ratio: float = 0.70,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
    visualize:   bool  = False,
    show_dimensions: bool = False,
    add_border:  bool  = False,
    max_viz:     int   = 10,
) -> None:
    """End-to-end Phase 0 pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # Locate metadata dir relative to this script
    script_dir   = Path(__file__).parent
    meta_dir     = script_dir.parent / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = Path(output_dir) / "viz_samples"
    if visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover JSON files ─────────────────────────────────────────────
    json_paths = sorted(glob.glob(os.path.join(input_dir, "room_*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No room_*.json files found in '{input_dir}'")
    print(f"Found {len(json_paths)} floor plan JSON files.")

    room_cnt, furn_cnt = collect_all_names(json_paths)

    print(f"\n{'='*60}")
    print(f"Room type names found  ({len(room_cnt)} unique):")
    for name, cnt in room_cnt.most_common():
        print(f"    '{name}': {cnt} occurrences  → type_id {lookup_room_type(name)} "
              f"({ROOM_TYPES.get(lookup_room_type(name), '?')})")

    print(f"\nFurniture names found  ({len(furn_cnt)} unique, top 30):")
    for name, cnt in furn_cnt.most_common(30):
        cid = lookup_furniture_category(name)
        print(f"    [{cnt:3d}x] '{name[:40]}' → cat {cid} ({CATEGORIES.get(cid, '?')})")
    print(f"{'='*60}\n")

    # ── 3. Build and save mappings ─────────────────────────────────────────
    room_type_map  = build_room_type_mapping(room_cnt)
    furn_cat_map   = build_furniture_category_mapping(furn_cnt)

    (meta_dir / "chinese_room_types.json").write_text(
        json.dumps(room_type_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (meta_dir / "chinese_furniture_categories.json").write_text(
        json.dumps(furn_cat_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (meta_dir / "chinese_idx_to_label.json").write_text(
        json.dumps({str(k): v for k, v in CATEGORIES.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (meta_dir / "chinese_room_types_idx.json").write_text(
        json.dumps({str(k): v for k, v in ROOM_TYPES.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved mappings to {meta_dir}")

    # ── 4. Split floor plans by plan ID ───────────────────────────────────
    plan_ids = [
        os.path.basename(p).replace("room_", "").replace(".json", "")
        for p in json_paths
    ]
    
    # Enforce specific floor plans to be in test set
    force_test_ids = {
        "2158503", "2177259", "2339563", "2473337", "2669661",
        "3044927", "3707839", "4090210", "4142540", "4358876",
        "4654871", "4655690", "4729472", "4773513", "4773532",
    }
    
    # Separate forced-test and remaining floor plans
    forced_test = [pid for pid in plan_ids if pid in force_test_ids]
    remaining = [pid for pid in plan_ids if pid not in force_test_ids]
    
    # Randomly shuffle and split remaining floor plans
    rng = random.Random(seed)
    rng.shuffle(remaining)

    n = len(remaining)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    split_map: Dict[str, str] = {}
    
    # Assign remaining floor plans according to train/val/test ratios
    for pid in remaining[:n_train]:
        split_map[pid] = "train"
    for pid in remaining[n_train: n_train + n_val]:
        split_map[pid] = "val"
    for pid in remaining[n_train + n_val:]:
        split_map[pid] = "test"
    
    # Force specified floor plans to test set
    for pid in forced_test:
        split_map[pid] = "test"

    n_train_actual = sum(1 for v in split_map.values() if v == "train")
    n_val_actual = sum(1 for v in split_map.values() if v == "val")
    n_test_actual = sum(1 for v in split_map.values() if v == "test")
    print(f"Dataset split: {n_train_actual} train / {n_val_actual} val / "
          f"{n_test_actual} test  (by floor plan ID, {len(forced_test)} forced to test)")

    # ── 4. Second pass: parse + render ────────────────────────────────────
    SH, SW  = room_size
    GH, GW  = global_size

    collected: Dict[str, dict] = {s: {"layout": [], "layout_cond": [], "global": [], "meta": []}
                                   for s in ("train", "val", "test")}

    viz_count = 0
    errors    = 0
    skipped   = 0
    total_rooms = 0

    for path in tqdm(json_paths, desc="Pass 2 – rendering"):
        plan_id = os.path.basename(path).replace("room_", "").replace(".json", "")
        split   = split_map.get(plan_id, "train")

        try:
            fp_data = parse_floorplan(path)
        except Exception as exc:
            print(f"  [ERROR] Parsing {path}: {exc}")
            errors += 1
            continue

        # Assign furniture to rooms
        room_to_furn = assign_furniture_to_rooms(fp_data["furniture"], fp_data["rooms"])

        # Global context mask (once per floor plan)
        global_mask = render_global_mask(fp_data, target_size=(GW, GH))

        has_furniture = bool(fp_data["furniture"])

        for room in fp_data["rooms"]:
            room_name = room["name"].strip()
            if should_drop_room(room_name):
                skipped += 1
                continue

            room_type_id = lookup_room_type(room_name)
            room_furn    = room_to_furn.get(room["id"], [])

            # Generate two versions: with furniture (target) and without (condition)
            result_with_furn = render_room_semantic_map(
                room        = room,
                holes       = fp_data["holes"],
                room_furniture = room_furn,
                target_size = (SH, SW),
                include_furniture = True,
            )
            result_no_furn = render_room_semantic_map(
                room        = room,
                holes       = fp_data["holes"],
                room_furniture = [],  # No furniture for condition map
                target_size = (SH, SW),
                include_furniture = False,
            )
            
            # Both results should return semantic map + room geometry info or None.
            if result_with_furn is None or result_no_furn is None:
                skipped += 1
                continue
            
            sem_map_with_furn, room_width_cm, room_height_cm, room_polygon_px, edge_lengths_cm = result_with_furn
            sem_map_no_furn, _, _, _, _ = result_no_furn

            # Build (2, H, W) tensors for both versions
            room_type_layer = np.full((SH, SW), room_type_id, dtype=np.uint8)
            sample_with_furn = np.stack([room_type_layer, sem_map_with_furn], axis=0)
            sample_no_furn = np.stack([room_type_layer, sem_map_no_furn], axis=0)

            bucket = collected[split]
            # Store both versions (target with furniture, condition without)
            bucket["layout"].append(sample_with_furn)
            bucket["layout_cond"].append(sample_no_furn)
            bucket["global"].append(global_mask)
            bucket["meta"].append({
                "plan_id":      plan_id,
                "room_id":      room["id"],
                "room_name":    room_name,
                "room_type_id": room_type_id,
                "room_type":    ROOM_TYPES.get(room_type_id, "unknown"),
                "has_furniture":has_furniture,
                "n_furniture":  len(room_furn),
                "room_width_cm":  float(room_width_cm),
                "room_height_cm": float(room_height_cm),
            })
            total_rooms += 1
            
            # 为条件图（无家具）生成带尺寸标注的PNG，用于推理使用
            condition_viz_dir = Path(output_dir) / f"condition_with_dimensions_{split_name}"
            condition_viz_dir.mkdir(parents=True, exist_ok=True)
            sample_idx = total_rooms - 1  # 当前样本在该split中的索引
            condition_png_path = str(condition_viz_dir / f"sample_{sample_idx}_condition.png")
            save_visualization_with_dimensions(
                sem_map_no_furn,
                room_width_cm,
                room_height_cm,
                room_polygon_px,
                edge_lengths_cm,
                condition_png_path,
                add_border=False,
            )

            # Optional visualisation (visualize both versions with and without furniture)
            if visualize and viz_count < max_viz:
                if show_dimensions:
                    # Visualize version WITH furniture (with dimension annotations)
                    viz_path_with = str(viz_dir / f"{plan_id}_{room['id']}_{room_name}_with_furn.png")
                    save_visualization_with_dimensions(
                        sem_map_with_furn,
                        room_width_cm,
                        room_height_cm,
                        room_polygon_px,
                        edge_lengths_cm,
                        viz_path_with,
                        add_border=add_border,
                    )
                    
                    # Visualize version WITHOUT furniture (condition, with dimension annotations)
                    viz_path_no_furn = str(viz_dir / f"{plan_id}_{room['id']}_{room_name}_no_furn.png")
                    save_visualization_with_dimensions(
                        sem_map_no_furn,
                        room_width_cm,
                        room_height_cm,
                        room_polygon_px,
                        edge_lengths_cm,
                        viz_path_no_furn,
                        add_border=add_border,
                    )
                else:
                    # Simple visualization without dimension annotations
                    viz_path_with = str(viz_dir / f"{plan_id}_{room['id']}_{room_name}_with_furn.png")
                    save_visualization(sem_map_with_furn, viz_path_with)
                    
                    viz_path_no_furn = str(viz_dir / f"{plan_id}_{room['id']}_{room_name}_no_furn.png")
                    save_visualization(sem_map_no_furn, viz_path_no_furn)
                
                viz_count += 1

    # ── 6. Save outputs ────────────────────────────────────────────────────
    print(f"\nTotal rooms rendered: {total_rooms}  (errors={errors}, skipped={skipped})")
    
    # 显示生成的条件图目录
    print(f"\n已生成带尺寸标注的条件图到：")
    for split_name in ["train", "val", "test"]:
        cond_dir = Path(output_dir) / f"condition_with_dimensions_{split_name}"
        if cond_dir.exists():
            n_files = len(list(cond_dir.glob("*.png")))
            if n_files > 0:
                print(f"  {cond_dir.name}/  ({n_files} 个PNG文件)")

    for split_name, bucket in collected.items():
        samples = bucket["layout"]
        samples_cond = bucket["layout_cond"]
        if not samples:
            print(f"  [{split_name}] No samples — skipping.")
            continue

        layout_arr = np.stack(samples, axis=0)          # (N, 2, H, W) uint8 - target with furniture
        layout_cond_arr = np.stack(samples_cond, axis=0)  # (N, 2, H, W) uint8 - condition without furniture
        global_arr = np.array(bucket["global"], dtype=np.uint8)  # (N, GH, GW)

        n_s = len(samples)
        out_layout = os.path.join(output_dir, f"chinese_{split_name}_{SH}x{SW}.npy")
        out_layout_cond = os.path.join(output_dir, f"chinese_{split_name}_condition_{SH}x{SW}.npy")
        out_global = os.path.join(output_dir, f"chinese_global_{split_name}_{GH}x{GW}.npy")
        out_meta   = os.path.join(output_dir, f"chinese_meta_{split_name}.json")
        out_csv    = os.path.join(output_dir, f"chinese_{split_name}_splits.csv")

        np.save(out_layout, layout_arr)
        np.save(out_layout_cond, layout_cond_arr)
        np.save(out_global, global_arr)

        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(bucket["meta"], f, ensure_ascii=False, indent=2)

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["plan_id", "room_id", "room_name", "room_type_id",
                             "room_type", "has_furniture", "n_furniture",
                             "room_width_cm", "room_height_cm"],
            )
            writer.writeheader()
            writer.writerows(bucket["meta"])

        # Summary statistics
        sem_maps  = layout_arr[:, 1, :, :]
        cat_ids   = np.unique(sem_maps)
        room_type_ids = np.unique(layout_arr[:, 0, 0, 0])
        pct_furniture = (sem_maps > 1).any(axis=(1, 2)).mean() * 100

        print(f"\n[{split_name}]  {n_s} rooms")
        print(f"  Layout (with furniture)     → {out_layout}  shape={layout_arr.shape}")
        print(f"  Layout (condition, no furn) → {out_layout_cond}  shape={layout_cond_arr.shape}")
        print(f"  Global  → {out_global}  shape={global_arr.shape}")
        print(f"  Unique semantic cat IDs : {cat_ids.tolist()}")
        print(f"  Unique room type IDs    : {room_type_ids.tolist()}")
        print(f"  Rooms with any furniture: {pct_furniture:.1f}%")

    if visualize and viz_count:
        print(f"\nSaved {viz_count} visualisation PNGs to {viz_dir}")

    print("\n[OK]  Phase 0 complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: parse floor-plan JSONs → training-ready .npy files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        default="/share/home/202230550120/diffusers/plan_json",
        help="Directory containing room_*.json files",
    )
    parser.add_argument(
        "--output_dir",
        default="/share/home/202230550120/diffusers/output_2026.3.27",
        help="Output directory for .npy and metadata files",
    )
    parser.add_argument("--room_h",     type=int,   default=1024)
    parser.add_argument("--room_w",     type=int,   default=1024)
    parser.add_argument("--global_h",   type=int,   default=64)
    parser.add_argument("--global_w",   type=int,   default=64)
    parser.add_argument("--train_ratio",type=float, default=0.70)
    parser.add_argument("--val_ratio",  type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save colour-coded PNG samples for visual inspection",
    )
    parser.add_argument(
        "--show_dimensions",
        action="store_true",
        help="Add per-edge room dimension annotations to visualization (requires --visualize)",
    )
    parser.add_argument(
        "--add_border",
        action="store_true",
        help="Add white border around visualizations (only with --show_dimensions, increases image size)",
    )
    parser.add_argument(
        "--max_viz",
        type=int, default=100000000,
        help="Maximum number of visualisation PNGs to save",
    )

    args = parser.parse_args()

    process_all(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        room_size   = (args.room_h, args.room_w),
        global_size = (args.global_h, args.global_w),
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
        visualize   = args.visualize,
        show_dimensions = args.show_dimensions,
        add_border   = args.add_border,
        max_viz     = args.max_viz,
    )


if __name__ == "__main__":
    main()
