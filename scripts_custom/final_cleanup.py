#!/usr/bin/env python3
"""Comprehensive cleanup of LLM output - detect AND REMOVE explanation text"""
import json
import re

input_file = "/share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_test_enhanced_llm_mixed.json"
output_file = "/share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_test_enhanced_llm_mixed_clean.json"

print("🧹 COMPREHENSIVE CLEANUP\n")

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Patterns that should result in EMPTY string
SHOULD_BE_EMPTY_PATTERNS = [
    r"输出为空字符串",
    r"输出:\s*\"\"",
    r"相关需求描述为空字符串",
    r"与.+相关的需求描述为空字符串",
    r"没有直接提到",
    r"无法从中提取",
    r"因此返回空字符串",
    r"因此，输出结果应为空",
    r"因此，输出为空",
    r"由于全局描述.*无法提取",
    r"未提及.*的具体需求",
    r"没有找到与.*直接相关",
    r"与.+相关的需求描述为空",
    r"^\"\"$",  # Just empty quotes
]

# Patterns where we should extract content from quotes
EXTRACT_PATTERNS = [
    r'^输出:\s*["{1,2}]?(.+?)["{1,2}]?$',  # 输出: "content"
    r'^\"(.+?)\"$',  # "content"
]

cleaned_count = 0
extracted_count = 0
already_clean = 0
details_cleaned = {}

for entry in data:
    prompt = entry.get("prompt", "").strip()
    original = prompt
    
    # Check if should be empty
    should_be_empty = False
    for pattern in SHOULD_BE_EMPTY_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            should_be_empty = True
            break
    
    if should_be_empty:
        entry["prompt"] = ""
        cleaned_count += 1
        room_type = entry.get("room_type", "unknown")
        if room_type not in details_cleaned:
            details_cleaned[room_type] = 0
        details_cleaned[room_type] += 1
        continue
    
    # Try to extract content from quotes
    extracted_something = False
    for pattern in EXTRACT_PATTERNS:
        match = re.search(pattern, prompt)
        if match:
            content = match.group(1).strip()
            if content and len(content) > 2 and not any(
                word in content for word in 
                ["输出", "空字符串", "无法", "没有", "因此", "需求描述为"]
            ):
                entry["prompt"] = content
                extracted_count += 1
                extracted_something = True
                break
    
    if not extracted_something and prompt:
        already_clean += 1

print(f"📊 STATISTICS:\n")
print(f"  ✅ Cleaned (set to empty): {cleaned_count}")
print(f"  ✂️  Extracted (removed quotes/prefix): {extracted_count}")
print(f"  ✓ Already clean: {already_clean}")
print(f"  Total entries: {len(data)}\n")

if cleaned_count > 0:
    print(f"📋 Breakdown (cleaned entries):")
    for room_type, count in sorted(details_cleaned.items(), key=lambda x: x[1], reverse=True):
        print(f"   {room_type:20s}: {count:3d}")

# Final verification
empty_count = sum(1 for entry in data if not entry.get("prompt", "").strip())
non_empty = len(data) - empty_count

print(f"\n🔍 FINAL STATE:")
print(f"   Empty prompts: {empty_count} ({empty_count/len(data)*100:.1f}%)")
print(f"   Non-empty prompts: {non_empty} ({non_empty/len(data)*100:.1f}%)")

# Check for any remaining problematic entries
problematic = []
for i, entry in enumerate(data):
    prompt = entry.get("prompt", "")
    if any(word in prompt for word in ["输出", "空字符串", "因此", "无法从中"]):
        problematic.append((i, entry.get("room_name"), prompt[:50]))

if problematic:
    print(f"\n⚠️  FOUND {len(problematic)} REMAINING PROBLEMATIC ENTRIES:")
    for idx, room_name, text in problematic[:10]:  # Show first 10
        print(f"   Line {idx}: {room_name} -> {text}...")
else:
    print(f"\n✅ NO PROBLEMATIC ENTRIES FOUND!")

# Save
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved to: {output_file}")
