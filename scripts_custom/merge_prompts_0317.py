#!/usr/bin/env python3
"""
Merge prompt_en from JSON into metadata_kontext_add_train_v8.csv
"""
import json
import csv
import os
import re
from typing import Dict, Tuple

def parse_image_basename(image_path: str) -> Tuple[str, str]:
    """Extract plan_id and room_id from image filename."""
    basename = os.path.basename(image_path)
    # Pattern: plan_id_room_id_...
    match = re.match(r"(\d+)_(\d+)_", basename)
    if match:
        return match.group(1), match.group(2)
    return "", ""


def main():
    json_path = "/share/home/202230550120/diffusers/output_add/chinese_meta_train_enhanced_llm_xlsx_3.25_clean_en.json"
    csv_path = "/share/home/202230550120/diffusers/output_add/metadata_kontext_add_train_v8.csv"
    output_path = "/share/home/202230550120/diffusers/output_add/metadata_kontext_add_train_v8_enhanced.csv"
    
    # Load JSON
    print("[*] Loading JSON metadata...")
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Build prompt_en map: (plan_id, room_id) -> prompt_en
    prompt_en_map: Dict[Tuple[str, str], str] = {}
    for item in json_data:
        plan_id = str(item.get("plan_id", ""))
        room_id = str(item.get("room_id", ""))
        prompt_en = item.get("prompt_en", "")
        
        if plan_id and room_id and prompt_en:
            key = (plan_id, room_id)
            prompt_en_map[key] = prompt_en
    
    print(f"[*] Loaded {len(prompt_en_map)} prompt_en entries from JSON")
    
    # Process CSV
    print("[*] Processing CSV...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"[*] Read {len(rows)} rows from CSV")
    
    # Merge prompts
    updated_count = 0
    for row in rows:
        image = row.get("image", "")
        plan_id, room_id = parse_image_basename(image)
        
        if plan_id and room_id:
            key = (plan_id, room_id)
            if key in prompt_en_map:
                prompt_en = prompt_en_map[key]
                # Append feature to prompt
                original_prompt = row.get("prompt", "")
                if original_prompt and not original_prompt.endswith("."):
                    original_prompt += "."
                row["prompt"] = f"{original_prompt} feature: {prompt_en}"
                updated_count += 1
    
    print(f"[*] Updated {updated_count} rows with prompt_en")
    
    # Write output
    print(f"[*] Writing output to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = rows[0].keys() if rows else ["image", "kontext_images", "prompt"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[OK] Done! Output saved to {output_path}")
    
    # Print sample
    print("\nSample merged prompts:")
    for i, row in enumerate(rows[:3]):
        if row.get("prompt"):
            image = row.get("image", "")
            prompt = row.get("prompt", "")
            print(f"  [{i}] {image}")
            print(f"      {prompt[:120]}...")


if __name__ == "__main__":
    main()
