#!/usr/bin/env python3
"""
从metadata_paired.json生成CSV格式的训练数据清单
"""

import json
import csv

METADATA_JSON = "/share/home/202230550120/diffusers/scripts_custom/metadata_paired.json"
METADATA_CSV = "/share/home/202230550120/diffusers/scripts_custom/metadata_paired.csv"

def main():
    # 读取JSON metadata
    with open(METADATA_JSON, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 写入CSV
    with open(METADATA_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['plan_id', 'condition_image', 'gt_image', 'prompt'])
        writer.writeheader()
        
        for record in metadata:
            writer.writerow({
                'plan_id': record['plan_id'],
                'condition_image': record['edit_image'],  # condition是edit_image字段
                'gt_image': record['image'],  # GT是image字段
                'prompt': record['prompt']
            })
    
    print(f"生成CSV文件: {METADATA_CSV}")
    print(f"总记录数: {len(metadata)}")

if __name__ == "__main__":
    main()
