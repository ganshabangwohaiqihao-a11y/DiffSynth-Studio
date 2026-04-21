#!/usr/bin/env python3
"""
根据户型ID配对condition和GT图像，生成新的metadata.json
condition: /share/home/202230550120/diffusers/scripts_custom/full_floorplans_no_furniture/floor_plans_png/<ID>_floorplan.png
GT: /share/home/202230550120/extracted_images_col20/edited_image_<ID>.png
"""

import json
import os
from pathlib import Path

# 路径定义
METADATA_SRC = "/share/home/202230550120/DiffSynth-Studio/data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata.json"
METADATA_OUT = "/share/home/202230550120/diffusers/scripts_custom/metadata_paired.json"

CONDITION_DIR = "/share/home/202230550120/diffusers/scripts_custom/full_floorplans_no_furniture/floor_plans_png"
GT_DIR = "/share/home/202230550120/extracted_images_col20"

def extract_plan_id(filename):
    """从文件名提取户型ID"""
    # edited_image_1540780.png -> 1540780
    # 1540780_floorplan.png -> 1540780
    return filename.replace("edited_image_", "").replace("_floorplan.png", "").replace(".png", "")

def main():
    # 读取原始metadata
    with open(METADATA_SRC, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"读取原始metadata: {len(metadata)} 条记录")
    
    # 构建condition图像映射
    condition_files = os.listdir(CONDITION_DIR)
    condition_map = {}
    for fname in condition_files:
        plan_id = extract_plan_id(fname)
        condition_map[plan_id] = os.path.join(CONDITION_DIR, fname)
    
    print(f"Condition图像: {len(condition_map)} 个")
    
    # 构建GT图像映射
    gt_files = os.listdir(GT_DIR)
    gt_map = {}
    for fname in gt_files:
        plan_id = extract_plan_id(fname)
        gt_map[plan_id] = os.path.join(GT_DIR, fname)
    
    print(f"GT图像: {len(gt_map)} 个")
    
    # 配对并生成新的metadata
    paired_metadata = []
    paired_count = 0
    missing_condition = 0
    
    for i, record in enumerate(metadata):
        # 从GT图像路径提取ID
        gt_filename = os.path.basename(record["image"])
        plan_id = extract_plan_id(gt_filename)
        
        # 检查是否有对应的condition图像
        if plan_id in condition_map:
            new_record = {
                "image": record["image"],  # GT图像保持不变
                "edit_image": condition_map[plan_id],  # 替换为condition图像
                "prompt": record.get("prompt", ""),
                "plan_id": plan_id
            }
            paired_metadata.append(new_record)
            paired_count += 1
        else:
            missing_condition += 1
            print(f"警告: 图像 {plan_id} 无对应condition图像 (GT: {record['image']})")
    
    # 保存新的metadata
    with open(METADATA_OUT, 'w', encoding='utf-8') as f:
        json.dump(paired_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n配对结果:")
    print(f"  总记录数: {len(metadata)}")
    print(f"  成功配对: {paired_count}")
    print(f"  缺少condition: {missing_condition}")
    print(f"输出文件: {METADATA_OUT}")
    print(f"输出记录数: {len(paired_metadata)}")

if __name__ == "__main__":
    main()
