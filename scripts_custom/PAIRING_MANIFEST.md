# 户型ID配对数据清单

## 📊 统计摘要

| 指标 | 数量 |
|------|------|
| Condition 图像总数 | 119 个 |
| GT 图像总数 | 92 个 |
| **成功配对** | **89 对** |
| 缺少 GT 的 Condition | 27 个 |

## 📁 数据路径

### Condition 图像（清晰的结构化图像，无家具）
- 位置：`/share/home/202230550120/diffusers/scripts_custom/full_floorplans_no_furniture/floor_plans_png/`
- 文件格式：`<plan_id>_floorplan.png`
- 分辨率：1024×1024 像素（推测）
- 描述：从 JSON 户型数据渲染的干净结构图，包含房间类型和面积标注

### GT 图像（风格化的渲染图像，包含家具等）
- 位置：`/share/home/202230550120/extracted_images_col20/`
- 文件格式：`edited_image_<plan_id>.png`
- 分辨率：800×848 像素（推测）
- 描述：通过 Qwen-Image-Edit 模型生成的风格化改造效果图，包含家具、贴图等

## 📋 输出文件

### 1. metadata_paired.json（JSON 格式）
路径：`/share/home/202230550120/diffusers/scripts_custom/metadata_paired.json`

结构示例：
```json
[
  {
    "plan_id": "1540780",
    "image": "/share/home/202230550120/extracted_images_col20/edited_image_1540780.png",
    "edit_image": "/share/home/202230550120/diffusers/scripts_custom/full_floorplans_no_furniture/floor_plans_png/1540780_floorplan.png",
    "prompt": "改造描述文本..."
  },
  ...
]
```

字段说明：
- `plan_id`: 户型 ID（唯一标识符）
- `image`: GT 图像路径（目标域渲染图）
- `edit_image`: Condition 图像路径（条件/输入图）
- `prompt`: 改造设计文本描述

### 2. metadata_paired.csv（CSV 格式）
路径：`/share/home/202230550120/diffusers/scripts_custom/metadata_paired.csv`

字段：`plan_id, condition_image, gt_image, prompt`

CSV 更易于数据加载、分割、验证等操作。

## 🎯 使用建议

### 用于模型训练：
```python
import csv
import torch
from PIL import Image

# 加载配对数据
with open('metadata_paired.csv', 'r') as f:
    reader = csv.DictReader(f)
    samples = list(reader)

# 准备训练样本
for sample in samples:
    plan_id = sample['plan_id']
    condition_img = Image.open(sample['condition_image'])
    gt_img = Image.open(sample['gt_image'])
    
    # 预处理、归一化等
    # 送入模型训练...
```

### 数据分割（训练/验证/测试）：
```python
import random
from sklearn.model_selection import train_test_split

random.shuffle(samples)
train, test = train_test_split(samples, test_size=0.2, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)

print(f"Training: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
```

## ⚠️ 已知问题

- 119 个 condition 图像中有 27 个缺少对应的 GT 编辑图像
- 以下户型 ID 无法配对（共 27 个）：
  ```
  1711526, 2153099, 2329060, 2683605, 2722644, 2748655, 2774751, 2798185, 
  2894243, 2894247, 3016111, 3022191, 3045259, 3073068, 3148547, 3156654, 
  3164955, 3179195, 3237679, 3259047, 3977039, 4111984, 4123439, 4129959, 
  4232962, 4295142, 4358876
  ```

## 🔄 后续工作

1. ✅ **数据配对完成** - 89 对有效的 condition↔GT 配对
2. ⏳ **分辨率对齐** - 考虑是否需要将 GT 图像缩放到 1024×1024
3. ⏳ **数据增强** - 包括旋转、翻转、色彩抖动等
4. ⏳ **模型训练** - 使用配对数据训练 pix2pix 或类似的条件生成模型
