#!/usr/bin/env python3
"""More comprehensive cleanup of LLM output"""
import json
import re

input_file = "/home/chengjiajia/diffusers/output/chinese_meta_test_enhanced_llm.json"
output_file = "/home/chengjiajia/diffusers/output/chinese_meta_test_enhanced_llm_cleaned.json"

print("🧹 Advanced cleaning of LLM output...\n")

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

cleaned_count = 0
simplified_count = 0

for entry in data:
    prompt = entry.get("prompt", "").strip()
    original = prompt
    
    # 1. Patterns that should result in EMPTY string
    if any(pattern in prompt for pattern in [
        "相关需求描述为空字符串",
        "没有直接提到",
        "无法从中提取",
        "因此，输出结果应为空字符串",
        "因此，输出为空字符串",
        "输出: \"\"",
    ]):
        entry["prompt"] = ""
        cleaned_count += 1
        continue
    
    # 2. Clean up "输出:" prefix and trailing explanation
    if prompt.startswith("输出:"):
        # Extract content after "输出:" and before any explanation
        match = re.search(r'输出:\s*["{1,2}]?([^"]*)["{1,2}]?(?:\s|$)', prompt)
        if match:
            content = match.group(1).strip()
            if content and content != "":
                entry["prompt"] = content
                simplified_count += 1
            else:
                entry["prompt"] = ""
                cleaned_count += 1
    
    # 3. Remove outer quotes if present (but preserve inner content)
    elif prompt.startswith('"') and prompt.endswith('"') and len(prompt) > 2:
        # Remove outer quotes
        unquoted = prompt[1:-1]
        if unquoted and unquoted not in ['', '""']:
            entry["prompt"] = unquoted
            simplified_count += 1

# Save cleaned output
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Cleaned (removed): {cleaned_count} entries")
print(f"✅ Simplified (formatted): {simplified_count} entries")
print(f"\n📊 Final stats:")

# Count empty vs non-empty
empty = sum(1 for entry in data if not entry.get("prompt", "").strip())
non_empty = len(data) - empty
print(f"   Total entries: {len(data)}")
print(f"   Non-empty prompts: {non_empty} ({non_empty/len(data)*100:.1f}%)")
print(f"   Empty prompts: {empty} ({empty/len(data)*100:.1f}%)")

print(f"\n✅ Saved to: {output_file}")
