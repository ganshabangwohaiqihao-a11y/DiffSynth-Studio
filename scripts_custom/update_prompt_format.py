#!/usr/bin/env python3
"""
更新JSON文件中的prompt_en格式

原格式: "Adequate bedroom collection system..."
新格式: "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: {room_type}. feature: Adequate bedroom collection system..."
"""

import json
from pathlib import Path

def update_prompt_format(json_path: str, output_path: str = None) -> None:
    """
    更新JSON文件中所有房间的prompt_en格式
    
    Args:
        json_path: 输入JSON文件路径
        output_path: 输出JSON文件路径（默认覆盖原文件）
    """
    
    json_path = Path(json_path)
    output_path = Path(output_path) if output_path else json_path
    
    print(f"[*] 加载JSON文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[*] 处理 {len(data)} 个房间")
    
    updated_count = 0
    skipped_count = 0
    
    for idx, item in enumerate(data):
        room_type = item.get('room_type', 'unknown')
        prompt_en = item.get('prompt_en', '')
        
        # 只处理有prompt_en的项
        if prompt_en and prompt_en.strip():
            # 检查是否已经是新格式（避免重复转换）
            if prompt_en.startswith('2D semantic segmentation map'):
                skipped_count += 1
                continue
            
            # 转换为新格式
            new_prompt_en = f"2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: {room_type}. feature: {prompt_en}"
            item['prompt_en'] = new_prompt_en
            updated_count += 1
            
            if idx < 3:  # 打印前3个示例
                print(f"\n  [样本 {idx}] {item.get('room_name', 'N/A')} ({room_type})")
                print(f"    原: {prompt_en[:80]}...")
                print(f"    新: {new_prompt_en[:100]}...")
        else:
            skipped_count += 1
    
    print(f"\n[*] 处理结果:")
    print(f"    更新: {updated_count} 个")
    print(f"    跳过: {skipped_count} 个（无prompt_en或已是新格式）")
    
    print(f"\n[*] 保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[✓] 完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="更新prompt_en格式")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出JSON文件路径（默认覆盖原文件）"
    )
    
    args = parser.parse_args()
    
    update_prompt_format(args.input, args.output)
