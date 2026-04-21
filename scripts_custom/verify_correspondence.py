#!/usr/bin/env python3
"""
验证 NPY 样本与 JSON prompt_en 的对应关系

用法:
    python verify_correspondence.py \
        --npy /path/to/chinese_test_condition_1024x1024.npy \
        --json /path/to/chinese_meta_test_enhanced_llm_mixed_clean_en.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="验证 NPY-JSON 对应关系")
    parser.add_argument("--npy", type=str, required=True, help="条件 NPY 文件路径")
    parser.add_argument("--json", type=str, required=True, help="元数据 JSON 文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    args = parser.parse_args()

    npy_path = Path(args.npy)
    json_path = Path(args.json)

    # 验证文件存在
    if not npy_path.exists():
        print(f"✗ NPY 文件不存在: {npy_path}")
        return
    if not json_path.exists():
        print(f"✗ JSON 文件不存在: {json_path}")
        return

    print("=" * 70)
    print("NPY 与 JSON 对应关系验证")
    print("=" * 70)

    # 加载 NPY
    print(f"\n[1] 加载 NPY 文件: {npy_path.name}")
    try:
        npy_data = np.load(npy_path)
        npy_count = npy_data.shape[0]
        npy_shape = npy_data.shape
        print(f"    ✓ 形状: {npy_shape}")
        print(f"    ✓ 样本数: {npy_count}")
    except Exception as e:
        print(f"    ✗ 加载失败: {e}")
        return

    # 加载 JSON
    print(f"\n[2] 加载 JSON 文件: {json_path.name}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        json_count = len(json_data) if isinstance(json_data, list) else None
        print(f"    ✓ 对象数: {json_count}")
    except Exception as e:
        print(f"    ✗ 加载失败: {e}")
        return

    # 验证数量匹配
    print(f"\n[3] 验证数量一致性")
    if npy_count == json_count:
        print(f"    ✓ NPY 样本数 ({npy_count}) == JSON 对象数 ({json_count})")
    else:
        print(f"    ✗ NPY 样本数 ({npy_count}) != JSON 对象数 ({json_count})")
        print(f"    建议: 检查数据是否对齐")

    # 统计 prompt_en 分布
    print(f"\n[4] 统计 prompt_en 分布")
    with_prompt = 0
    without_prompt = 0
    room_types = {}

    for idx, item in enumerate(json_data):
        has_prompt = bool(item.get("prompt_en"))
        if has_prompt:
            with_prompt += 1
        else:
            without_prompt += 1

        room_type = item.get("room_type", "unknown")
        if room_type not in room_types:
            room_types[room_type] = {"with": 0, "without": 0}
        room_types[room_type]["with" if has_prompt else "without"] += 1

    print(f"    ✓ 有 prompt_en: {with_prompt} ({100*with_prompt/json_count:.1f}%)")
    print(f"    ✓ 无 prompt_en: {without_prompt} ({100*without_prompt/json_count:.1f}%)")

    print(f"\n[5] 按房间类型统计")
    for room_type in sorted(room_types.keys()):
        stats = room_types[room_type]
        total = stats["with"] + stats["without"]
        pct = 100 * stats["with"] / total
        print(f"    {room_type:20s}: {stats['with']:3d}/{total:3d} ({pct:5.1f}%)")

    # 显示样本详情
    if args.verbose or npy_count <= 20:
        print(f"\n[6] 样本对应关系详情")
        print("-" * 70)
        for idx in range(min(10, npy_count)):  # 显示前 10 个或所有
            item = json_data[idx]
            room_id = item.get("room_id", "?")
            room_type = item.get("room_type", "?")
            prompt = item.get("prompt_en", "")
            prompt_status = "✓" if prompt else "✗"

            print(f"\n  [样本 {idx}]")
            print(f"    NPY 索引: {idx}")
            print(f"    JSON 项目: item[{idx}]")
            print(f"    room_id: {room_id}")
            print(f"    room_type: {room_type}")
            print(f"    prompt_en: {prompt_status} {prompt[:50]}" + (
                "..." if len(prompt) > 50 else ""
            ))

        if npy_count > 10:
            print(f"\n  ... (还有 {npy_count - 10} 个样本)")

    # 总结
    print(f"\n[7] 总结")
    print("-" * 70)
    print(f"  NPY 文件:     {npy_count} 个样本")
    print(f"  JSON 文件:    {json_count} 个对象")
    print(f"  对应关系:     item[i] <-> sample[i]")
    print(f"  prompt_en:    {with_prompt}/{json_count} 可用")
    print(f"  状态:         {'✓ 对应关系正确' if npy_count == json_count else '✗ 数量不匹配'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
