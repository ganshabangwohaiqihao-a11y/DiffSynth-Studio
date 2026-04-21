#!/usr/bin/env python3
"""
为推理结果图片重命名，添加户型ID
用法:
  python rename_with_scheme_id.py \
    --output-dir /share/home/202230550120/DiffSynth-Studio/examples/qwen_image/model_inference/infer_results \
    --metadata-csv /share/home/202230550120/DiffSynth-Studio/data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata1.csv
"""

import argparse
import csv
import os
import re
from pathlib import Path


def extract_scheme_id(image_path):
    """从图像路径中提取户型ID (scheme_id)"""
    # 假设文件名格式是 edited_image_XXXX.png，其中XXXX是ID
    match = re.search(r'edited_image_(\d+)\.png', image_path)
    if match:
        return match.group(1)
    return None


def main():
    parser = argparse.ArgumentParser(description="为推理结果图片添加户型ID重命名")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--metadata-csv", type=str, required=True, help="元数据CSV文件")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 读取metadata.csv
    metadata_list = []
    try:
        with open(args.metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_list.append(row)
    except Exception as e:
        print(f"错误: 无法读取CSV文件: {e}")
        return
    
    if not metadata_list:
        print("CSV文件中没有数据")
        return
    
    print(f"\n{'='*70}")
    print(f"重命名推理结果图片")
    print(f"{'='*70}")
    print(f"输出目录:         {output_dir}")
    print(f"元数据条数:       {len(metadata_list)}")
    print(f"{'='*70}\n")
    
    # 建立索引号到户型ID的映射
    sample_to_scheme_id = {}
    for idx, sample in enumerate(metadata_list):
        edit_image_path = sample.get('edit_image', '').strip()
        scheme_id = extract_scheme_id(edit_image_path)
        if scheme_id:
            sample_to_scheme_id[idx] = scheme_id
        else:
            print(f"⚠ 样本 {idx}: 无法从 {edit_image_path} 提取户型ID")
    
    # 遍历输出目录中的文件，进行重命名
    renamed_count = 0
    failed_count = 0
    
    # 获取所有sample_XXXX_YY_*.jpg文件
    pattern = r'sample_(\d+)_(\d+)_(.*?)\.(jpg|png)'
    
    for file in sorted(output_dir.glob('sample_*')):
        if file.is_dir():
            continue
        
        match = re.match(pattern, file.name)
        if not match:
            continue
        
        sample_id = int(match.group(1))
        stage_id = match.group(2)  # 00, 01, 02, 03
        stage_name = match.group(3)  # condition, generated, gt, combined
        ext = match.group(4)
        
        # 查找对应的户型ID
        if sample_id not in sample_to_scheme_id:
            print(f"⚠ {file.name}: 找不到对应的样本 {sample_id}")
            failed_count += 1
            continue
        
        scheme_id = sample_to_scheme_id[sample_id]
        
        # 生成新文件名
        new_name = f"scheme_{scheme_id}_sample_{sample_id:04d}_{stage_id}_{stage_name}.{ext}"
        new_path = file.parent / new_name
        
        # 重命名
        try:
            file.rename(new_path)
            print(f"✓ {file.name} → {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"✗ {file.name}: 重命名失败 - {e}")
            failed_count += 1
    
    print(f"\n{'='*70}")
    print(f"重命名完成:")
    print(f"  ✓ 成功: {renamed_count}")
    if failed_count > 0:
        print(f"  ✗ 失败: {failed_count}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
