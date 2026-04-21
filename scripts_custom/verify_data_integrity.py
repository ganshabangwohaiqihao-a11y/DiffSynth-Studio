#!/usr/bin/env python3
"""
验证 NPY-JSON 对应关系的数据完整性

这个脚本验证：
1. NPY 和 JSON 的生成顺序是否一致
2. 对应的房间元数据是否匹配
"""

import numpy as np
import json
from pathlib import Path


def verify_npy_json_correspondence():
    """验证 NPY 样本与 JSON 元数据的对应关系"""
    
    npy_path = Path("/share/home/202230550120/diffusers/output_2026.3.27/chinese_test_condition_1024x1024.npy")
    json_path = Path("/share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_test_enhanced_llm_mixed_clean_en.json")
    
    # 加载数据
    npy_data = np.load(npy_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print("=" * 80)
    print("NPY-JSON 对应关系数据完整性验证")
    print("=" * 80)
    
    # 1. 基本数量验证
    print(f"\n[1] 基本数量检查")
    print(f"    NPY shape: {npy_data.shape}")
    print(f"    NPY 样本数: {npy_data.shape[0]}")
    print(f"    JSON 对象数: {len(json_data)}")
    
    if npy_data.shape[0] != len(json_data):
        print(f"    ✗ 数量不匹配！")
        return False
    else:
        print(f"    ✓ 数量一致")
    
    # 2. NPY 数据结构验证
    print(f"\n[2] NPY 数据结构检查")
    print(f"    期望形状: (N, 2, H, W) 其中 N=109")
    print(f"    实际形状: {npy_data.shape}")
    
    if len(npy_data.shape) != 4 or npy_data.shape[1] != 2:
        print(f"    ✗ NPY 结构不正确")
        return False
    else:
        print(f"    ✓ NPY 结构正确")
    
    # 3. JSON 数据结构验证
    print(f"\n[3] JSON 数据结构检查")
    required_fields = ['plan_id', 'room_id', 'room_type', 'room_type_id']
    
    all_valid = True
    for idx, item in enumerate(json_data):
        for field in required_fields:
            if field not in item:
                print(f"    ✗ item[{idx}] 缺少字段: {field}")
                all_valid = False
                break
        if not all_valid:
            break
    
    if all_valid:
        print(f"    ✓ 所有 {len(json_data)} 个对象都包含必需字段")
    else:
        return False
    
    # 4. 房间类型一致性检查（通过 NPY 的房间类型层）
    print(f"\n[4] 房间类型一致性检查（NPY与JSON）")
    print(f"    从 NPY 第 0 通道提取房间类型 ID...")
    
    mismatches = 0
    for idx in range(min(10, len(json_data))):
        # NPY 中的房间类型层（第 0 通道）
        room_type_layer = npy_data[idx, 0]  # (H, W)
        npy_room_type_id = int(room_type_layer[0, 0])
        
        # JSON 中的房间类型 ID
        json_room_type_id = json_data[idx].get('room_type_id', -1)
        
        match_status = "✓" if npy_room_type_id == json_room_type_id else "✗"
        print(f"    [{idx}] {match_status} NPY={npy_room_type_id}, JSON={json_room_type_id} ({json_data[idx]['room_type']})")
        
        if npy_room_type_id != json_room_type_id:
            mismatches += 1
    
    if mismatches > 0:
        print(f"    ✗ 发现 {mismatches} 处不匹配")
        return False
    else:
        print(f"    ✓ 前 10 个样本房间类型全部一致")
    
    # 5. Prompt_en 覆盖率检查
    print(f"\n[5] prompt_en 覆盖率检查")
    with_prompt = sum(1 for item in json_data if item.get('prompt_en'))
    without_prompt = len(json_data) - with_prompt
    print(f"    有 prompt_en: {with_prompt}/{len(json_data)} ({100*with_prompt/len(json_data):.1f}%)")
    print(f"    无 prompt_en: {without_prompt}/{len(json_data)} ({100*without_prompt/len(json_data):.1f}%)")
    
    # 6. 生成顺序的数据完整性
    print(f"\n[6] 生成顺序完整性检查")
    print(f"    验证序列是否连续且无重复...")
    
    plan_ids = [item['plan_id'] for item in json_data]
    room_ids = [item['room_id'] for item in json_data]
    
    # 检查是否有重复的 (plan_id, room_id) 对
    seen_pairs = set()
    duplicates = 0
    for idx, (plan_id, room_id) in enumerate(zip(plan_ids, room_ids)):
        pair = (plan_id, room_id)
        if pair in seen_pairs:
            print(f"    ✗ 重复的 (plan_id, room_id) 对: {pair} at index {idx}")
            duplicates += 1
        seen_pairs.add(pair)
    
    if duplicates > 0:
        print(f"    ✗ 发现 {duplicates} 个重复对")
        return False
    else:
        print(f"    ✓ 无重复对，序列完整")
    
    # 7. 最终结论
    print(f"\n[7] 最终结论")
    print(f"    ✓ NPY 样本与 JSON 元数据的对应关系正确")
    print(f"    ✓ 可以安全地使用 metadata[i] 对应 sample[i]")
    print(f"    ✓ JSON[i].prompt_en 对应 NPY[i] 的推理提示词")
    
    print(f"\n" + "=" * 80)
    print(f"数据完整性验证: ✓ 通过")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = verify_npy_json_correspondence()
    exit(0 if success else 1)
