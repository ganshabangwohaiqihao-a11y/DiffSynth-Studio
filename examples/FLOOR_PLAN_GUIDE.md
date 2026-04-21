# 户型改造数据准备指南

## 问题总结

### ❌ 问题1：GT图包含家具
- **影响**：模型会学习家具可变特征，而非纯结构
- **表现**：训练效果不稳定，生成时可能出现不必要的家具重排

### ❌ 问题2：条件输入缺少房间标注  
- **影响**：模型不知道图中哪个区域是"主卧"、"客厅"等
- **表现**：可能在错误的房间执行改造

---

## ✅ 完整解决方案

### 步骤1：去除GT图中的家具

#### 方式A：形态学运算（推荐，快速）
```bash
python furniture_removal.py \
  --input-dir ./gt_images \
  --output-dir ./gt_images_clean \
  --method morphology
```

**特点**：
- ⚡ 速度快，不需要深度学习
- 🎯 保留主要线条，移除细节
- ✓ 适合清晰的黑白户型图

#### 方式B：轮廓分析（精确度高）
```bash
python furniture_removal.py \
  --input-dir ./gt_images \
  --output-dir ./gt_images_clean \
  --method contour \
  --area-threshold 1000  # 调整阈值
```

**特点**：
- 🎯 基于面积过滤，移除小对象
- ✓ 更好地保留房间结构
- ⏱️ 速度中等

#### 方式C：Inpainting修复（效果逼真，慢）
```bash
python furniture_removal.py \
  --input-image ./gt.png \
  --output-image ./gt_clean.png \
  --method inpainting
```

**特点**：
- 🎨 AI修复，效果最逼真
- ⏱️ 耗时较长
- 💾 需要较多显存

#### 处理单张图像
```bash
python furniture_removal.py \
  --input-image ./sample_gt.png \
  --output-image ./sample_gt_clean.png \
  --method morphology
```

---

### 步骤2：为条件输入图添加房间标注

这是**关键一步**！

#### 方式A：自动标注（从JSON）

```bash
python floor_plan_annotator.py \
  --image floor_plan.png \
  --json floor_plan.json \
  --output floor_plan_annotated.png
```

**前置条件**：你的JSON包含房间坐标信息
```json
[
  [],  // [0] 未使用
  [    // [1] 房间标签数组
    ["room_001", 500, 300, 25, 30],  // room_name, x, y, length, width
    ["room_002", 750, 400, 15, 20],
    ...
  ],
  ...
]
```

#### 方式B：手动指定房间

```bash
python floor_plan_annotator.py \
  --image floor_plan.png \
  --rooms '[
    {"name": "主卧", "x": 500, "y": 300},
    {"name": "次卧", "x": 750, "y": 400},
    {"name": "客厅", "x": 800, "y": 700},
    {"name": "厨房", "x": 400, "y": 800}
  ]' \
  --output floor_plan_annotated.png
```

#### 方式C：处理单个房间

```bash
python floor_plan_annotator.py \
  --image floor_plan.png \
  --rooms '[{"name": "拆除区域", "x": 600, "y": 400}]' \
  --output floor_plan_with_mask.png
```

---

## 📊 数据流示意

### 原始流程（有问题）
```
GT图（含家具）     条件输入图（无标注）
        ↓                    ↓
    [训练集]            [条件输入]
        ↓                    ↓
   Qwen-Image-Edit ←────────┘
        ↓
    ❌ 效果不稳定，可能改错地方
```

### 改进流程（推荐）
```
GT图（含家具）                条件输入图
    ↓                             ↓
[furniture_removal.py]    [floor_plan_annotator.py]
    ↓                             ↓
GT图（纯结构）         条件输入图（标注房间）
    ↓                             ↓
    └──────┬───────────────────────┘
           ↓
      [训练集]    [条件输入]
           ↓           ↓
      Qwen-Image-Edit ⚡
           ↓
      ✓ 稳定、可控、精确
```

---

## 🎯 条件输入图的标注方式

### 核心要素

对于文本"拆除主卧与客餐厅之间的隔墙"，模型需要这样理解：

```
条件输入图：
┌─────────────────────────────┐
│  【主卧】  【隔墙】【客厅】 │  ← 房间标注
│                             │
│  [结构边界清晰]             │  ← 视觉信息
└─────────────────────────────┘

文本提示：
"拆除主卧与客餐厅之间的隔墙"

模型推理：
✓ 定位"主卧" → 图中左侧蓝色区域
✓ 定位"客厅" → 图中右侧绿色区域  
✓ 定位隔墙 → 两者之间的分割线
✓ 动作"拆除" → 移除该线，连接两个区域
```

### 标注内容建议

| 要素 | 是否需要 | 说明 |
|------|--------|------|
| **房间名称** | ✅ **必须** | 主卧、次卧、客厅、餐厅、厨房等 |
| **房间分类** | ✅ **推荐** | 用颜色或符号区分，如"□卧室 ◇客厅" |
| **尺寸标注** | ⚙️ 可选 | 如需精确改造，可标注尺寸 |
| **门窗符号** | ⚙️ 可选 | 有助于模型理解流线 |
| **改造区域掩码** | ⚙️ 可选 | 明确哪些区域允许修改 |

---

## 🔧 完整工作流示例

### 假设你有：
- `original_floor_plan.png` - 原始户型图（无标注家具）
- `gt_images/` - GT目录（包含家具）
- `floor_plan.json` - 房间坐标JSON

### 执行步骤：

```bash
# 1️⃣ 清理GT图中的家具
python furniture_removal.py \
  --input-dir ./gt_images \
  --output-dir ./gt_images_clean \
  --method morphology

# 2️⃣ 为条件输入图添加房间标注
python floor_plan_annotator.py \
  --image ./original_floor_plan.png \
  --json ./floor_plan.json \
  --output ./original_floor_plan_annotated.png

# 3️⃣ 现在可以准备训练数据
# 条件输入：original_floor_plan_annotated.png（带房间标注）
# GT目标： gt_images_clean/sample1.png（无家具）
# 文本提示：拆除主卧...

# ✓ 准备完成，可以开始训练！
```

---

## 📝 最佳实践

### ✅ DO

- **DO** 标注所有重要房间
- **DO** 使用清晰的房间名称（不要用"R1"，用"主卧"）
- **DO** 确保文本提示与标注对应
- **DO** 定期检查生成样本，验证改造是否在正确位置

### ❌ DON'T

- **DON'T** 让GT图包含移动的家具
- **DON'T** 用模糊的房间标记（如"区域1"）
- **DON'T** 文本说"主卧"，但图中没有标注
- **DON'T** 标注太密集或文字太小

---

## 📎 调试技巧

### 如果生成结果改错地方：

```python
# 检查清单
1. ✓ 条件输入图中房间是否正确标注？
2. ✓ 文本提示是否与标注名称一致？
3. ✓ GT图是否已移除所有干扰家具？
4. ✓ 改造区域是否清晰明确？
```

### 如果某些样本标注失败：

```bash
# 手动调整该样本的坐标
python floor_plan_annotator.py \
  --image problematic_floor_plan.png \
  --rooms '[
    {"name": "主卧", "x": 520, "y": 290},  # 手动调整位置
    {"name": "客厅", "x": 780, "y": 400}
  ]' \
  --output corrected_floor_plan.png
```

---

## 📊 效果对比示例

### 案例：拆除主卧隔墙

**不标注房间的后果**：
```
文本："拆除主卧隔墙"
模型：🤔 主卧在哪？可能错误识别
结果：❌ 拆除了次卧的墙
```

**标注房间的效果**：
```
文本："拆除主卧隔墙"
条件图：【主卧】标注清晰
模型：✓ 明确位置就是这个
结果：✅ 精确拆除主卧隔墙
```

---

## 总结

| 工具 | 用途 | 命令 |
|------|------|------|
| `furniture_removal.py` | 清理GT图家具 | `--method morphology` |
| `floor_plan_annotator.py` | 标注条件输入图 | `--json floor_plan.json` |

**简而言之**：
1. **训练数据**（GT）：必须去掉家具 🏠
2. **条件输入**：必须标注房间 🏷️

这两个改动会显著提升模型的训练效果和推理准确度！
