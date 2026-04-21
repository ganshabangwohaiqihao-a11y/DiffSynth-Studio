#!/usr/bin/env python3
"""Translate Chinese prompts to English using MarianMT"""
import json
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

input_file = "/share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_test_enhanced_llm_mixed_clean.json"
output_file = "/share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_test_enhanced_llm_mixed_clean_en.json"

print("🌍 Loading MarianMT model for ZH→EN translation...")
model_name = "/share/home/202230550120/.cache/modelscope/hub/models/Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to("cuda")

print("✅ Model loaded on GPU!\n")
print("📖 Translating prompts...\n")

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

translated_count = 0
skipped_count = 0

for entry in tqdm(data, desc="Translating", total=len(data)):
    prompt = entry.get("prompt", "").strip()
    
    if prompt:  # Only translate non-empty prompts
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        
        # Generate translation
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Update entry
        entry["prompt_en"] = translated_text
        entry["prompt_zh"] = prompt  # Keep original
        translated_count += 1
    else:
        skipped_count += 1

# Save output
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ TRANSLATION COMPLETE:")
print(f"   Translated: {translated_count}")
print(f"   Skipped (empty): {skipped_count}")
print(f"   Total: {len(data)}")
print(f"\n📁 Saved to: {output_file}")
print(f"\n💡 Each entry now has:")
print(f"   - prompt_zh: Original Chinese")
print(f"   - prompt_en: English translation")
