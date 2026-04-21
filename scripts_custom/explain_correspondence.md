# NPY 样本与 prompt_en 的对应机制说明

## 整体流程

```
chinese_test_condition_1024x1024.npy (109 个样本)
        |
        | sample_idx = 0, 1, 2, ..., 108
        |
        ▼
inference_npy.py - run_inference()
        |
        ├─► 加载条件: load_condition_npy(path, sample_idx)  --> condition (2, H, W)
        |
        ├─► 加载元数据: metadata_json --> metadata dict
        |       └─► metadata[0] = item[0].prompt_en
        |       └─► metadata[1] = item[1].prompt_en
        |       └─► ...
        |       └─► metadata[108] = item[108].prompt_en
        |
        ├─► 查找提示词: prompt = metadata.get(sample_idx, default_prompt)
        |
        └─► 推理并保存: sample_{idx}_pred.npy
```

## 数据对应关系

### 1. NPY 文件结构
- **文件**: `chinese_test_condition_1024x1024.npy`
- **形状**: `(109, 2, 1024, 1024)` uint8
- **索引**: `0 ~ 108`

### 2. JSON 元数据结构
- **文件**: `chinese_meta_test_enhanced_llm_mixed_clean_en.json`
- **形状**: 列表，共 109 个对象
- **对应规则**: `JSON[i]` 对应 `NPY[i]`

### 3. 逐步对应示例

```python
# NPY 中的样本 0
sample_idx = 0
condition_0 = numpy.load("chinese_test_condition_1024x1024.npy")[0]  # (2, 1024, 1024)

# JSON 中的元数据 0
metadata_0 = json.load("chinese_meta_test_enhanced_llm_mixed_clean_en.json")[0]
# {
#     "plan_id": "2158503",
#     "room_id": "9697",
#     "room_type": "balcony",
#     "prompt_en": ""  # 可能为空或有值
# }

# 查找该样本的提示词
prompt = metadata_0.get("prompt_en", "默认提示词")  # "" 或 "默认提示词"

# 进行推理
prediction = model.infer(condition_0, prompt)
# 保存为: sample_0_pred.npy
```

## 代码中的对应实现

### 加载阶段（line 244-260）
```python
metadata = {}
if metadata_json_path and metadata_json_path.exists():
    with open(metadata_json_path, 'r') as f:
        metadata_list = json.load(f)  # 加载 JSON 列表，长度 109
        
        # ★ 关键：按照枚举顺序建立对应
        for idx, item in enumerate(metadata_list):
            # idx 从 0 到 108，与 NPY 索引一致
            # 如果 item 有显式的 sample_index，则优先使用
            if "sample_index" in item:
                idx = item["sample_index"]
            
            # 从 item 中提取 prompt_en
            metadata[idx] = item.get("prompt_en", "")
```

结果: `metadata = {0: "...", 1: "...", ..., 108: "..."}`

### 查询阶段（line 281-282）
```python
for idx in range(num_samples):  # idx: 0 ~ 108
    condition = load_condition_npy(npy_path, idx)
    
    # ★ 查询对应的 prompt_en
    prompt = metadata.get(idx, "2D semantic segmentation map, ...")
    
    # 推理
    prediction = model(prompt, condition)
    
    # 保存为 sample_{idx}_pred.npy
```

## 关键点总结

| 位置 | NPY 索引 | JSON 数组索引 | room_id | room_type | prompt_en |
|-----|---------|-----------|---------|-----------|-----------|
| 第 1 个 | 0 | [0] | 9697 | balcony | "" |
| 第 2 个 | 1 | [1] | 11655 | bathroom | "" |
| 第 3 个 | 2 | [2] | 11958 | bedroom | "充足的卧室..." |
| ... | ... | ... | ... | ... | ... |
| 第 109 个 | 108 | [108] | ? | ? | ? |

## 验证对应关系

运行 [verify_correspondence.py](verify_correspondence.py) 来验证：
```bash
python verify_correspondence.py \
    --npy /share/home/202230550120/diffusers/output_2026.3.27/chinese_test_condition_1024x1024.npy \
    --json /share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_test_enhanced_llm_mixed_clean_en.json
```

输出示例：
```
[验证] NPY 样本数: 109
[验证] JSON 对象数: 109

[样本 0]
    NPY 索引: 0 ✓
    JSON 项目: room_id=9697, room_type=balcony
    prompt_en: "" ✗ (无提示词)

[样本 2]
    NPY 索引: 2 ✓
    JSON 项目: room_id=11958, room_type=bedroom
    prompt_en: "充足的卧室收纳..." ✓

... (共 109 个样本)
```

## 调试技巧

如果对应关系出现问题，检查以下几点：

1. **NPY 文件大小**: `chinese_test_condition_1024x1024.npy` 应该有 109 个样本
   ```bash
   python -c "import numpy as np; d=np.load('chinese_test_condition_1024x1024.npy'); print(f'Shape: {d.shape}, Count: {d.shape[0]}')"
   ```

2. **JSON 数组长度**: `chinese_meta_test_enhanced_llm_mixed_clean_en.json` 应该有 109 个对象
   ```bash
   python -c "import json; d=json.load(open('chinese_meta_test_enhanced_llm_mixed_clean_en.json')); print(f'Count: {len(d)}')"
   ```

3. **prompt_en 分布**: 查看有多少对象有 prompt_en 字段
   ```bash
   python -c "import json; d=json.load(open('chinese_meta_test_enhanced_llm_mixed_clean_en.json')); print(f'With prompt_en: {sum(1 for x in d if x.get(\"prompt_en\"))}, Without: {sum(1 for x in d if not x.get(\"prompt_en\"))}')"
   ```

## 推理时的对应关系

执行推理时：
```bash
python inference_npy.py \
    --condition-npy-dir output_2026.3.27 \
    --condition-split test \
    --lora-weights epoch-8.safetensors \
    --output-dir predictions_test \
    --metadata-json output_2026.3.27/chinese_meta_test_enhanced_llm_mixed_clean_en.json
```

脚本会：
1. 加载 NPY: 109 个样本，索引 0-108
2. 加载 JSON: 109 个对象，按顺序对应
3. 对每个 idx (0-108)：
   - 加载 condition[idx]
   - 查询 metadata[idx].prompt_en
   - 推理并保存 sample_{idx}_pred.npy
