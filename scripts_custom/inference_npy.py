#!/usr/bin/env python3
"""
推理脚本：将条件语义图通过 FLUX.1-Kontext 模型生成语义布局预测

输入：
  - condition_1024x1024.npy: 条件语义图 (N, 2, 1024, 1024)
  - epoch-8.safetensors: LoRA 权重

输出：
  - sample_{idx}_pred.npy: 预测语义图 (1024, 1024) uint8

使用方式：
  python inference_npy.py \
    --condition-npy-dir output_2026.3.27 \
    --condition-split test \
    --lora-weights models/train10.3.27.23:37/FLUX.1-Kontext-dev_lora_learning_rate_constant_nochidu/epoch-8.safetensors \
    --output-dir predictions_epoch8 \
    --metadata-json output_2026.3.27/chinese_meta_test.json
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 导入 DiffSynth-Studio 框架
try:
    sys.path.insert(0, str(Path("/share/home/202230550120/DiffSynth-Studio")))
    from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
    DIFFSYNTH_AVAILABLE = True
except ImportError as e:
    DIFFSYNTH_AVAILABLE = False
    print(f"警告: DiffSynth-Studio 框架未安装: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 颜色映射（与 parse_json_floorplan.py 保持一致）
# ──────────────────────────────────────────────────────────────────────────────

VOID_ID   = 0
FLOOR_ID  = 1
DOOR_ID   = 36
WINDOW_ID = 37

# 预定义的可视化颜色映射
_VIZ_COLORMAP: Dict[int, Tuple[int, int, int]] = {
    VOID_ID:   (255, 255, 255),
    FLOOR_ID:  (200, 200, 200),
    DOOR_ID:   (255, 120,  0),
    WINDOW_ID: (0,  180, 255),
    # Sample furniture colours
    2: (200,  80,  80), 3: (200,  80,  80), 4: (200,  80,  80),
    10: (80, 140, 200), 13: (100, 200, 100), 23: (200, 130,  50),
    33: (160,  60, 160), 39: (80, 200, 200), 42: (200, 200,  60),
}


def _cat_to_color(cat_id: int) -> Tuple[int, int, int]:
    """从颜色映射表获取颜色，或使用确定性生成（与 parse_json_floorplan.py 一致）"""
    if cat_id in _VIZ_COLORMAP:
        return _VIZ_COLORMAP[cat_id]
    # 对于未在映射表中的 ID，使用确定性随机生成
    rng = np.random.RandomState(cat_id * 17 + 3)
    return tuple(int(x) for x in rng.randint(60, 230, 3))


def save_visualization_with_dimensions(
    semantic_map: np.ndarray,
    room_width_cm: float,
    room_height_cm: float,
    room_polygon_px: List[Tuple[float, float]],
    edge_lengths_cm: List[float],
    out_path: str,
) -> None:
    """保存带房间尺寸标注的可视化图像（与 parse_json_floorplan.py 逻辑一致）"""
    from PIL import ImageFont
    
    h, w = semantic_map.shape
    
    # 转换语义图为RGB
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id in np.unique(semantic_map):
        mask = semantic_map == cat_id
        rgb[mask] = _cat_to_color(int(cat_id))
    
    bordered = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(bordered)
    
    # 使用字体
    try:
        font_size = 14
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text_color = (60, 60, 60)
    poly = room_polygon_px
    
    if len(poly) >= 3 and len(edge_lengths_cm) == len(poly):
        # 简化：不进行边合并，直接标注每条边
        cx = float(sum(x for x, _ in poly) / len(poly))
        cy = float(sum(y for _, y in poly) / len(poly))
        offset_px = 15
        
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            
            dx = x2 - x1
            dy = y2 - y1
            norm = float(np.hypot(dx, dy))
            if norm < 1e-6:
                continue
            
            # 法向量
            nx, ny = -dy / norm, dx / norm
            # 指向中心
            to_center_x = cx - mx
            to_center_y = cy - my
            if nx * to_center_x + ny * to_center_y < 0:
                nx, ny = -nx, -ny
            
            lx = mx + nx * offset_px
            ly = my + ny * offset_px
            label = f"{edge_lengths_cm[i]:.0f}cm"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = int(round(lx - tw / 2))
            ty = int(round(ly - th / 2))
            
            # 保证文字在图像范围内
            img_width, img_height = bordered.size
            tx = max(0, min(tx, img_width - tw))
            ty = max(0, min(ty, img_height - th))
            
            draw.text((tx, ty), label, fill=text_color, font=font)
    
    bordered.save(out_path)


def load_condition_npy(npy_path: Path, sample_index: int) -> np.ndarray:
    """
    加载条件 NPY 文件中的单个样本
    
    Args:
        npy_path: 条件 NPY 文件路径 (N, 2, H, W)
        sample_index: 样本索引
    
    Returns:
        condition_map: 形状 (2, H, W) 的语义图
            - Channel 0: 房间类型 (0-15)
            - Channel 1: 语义分割图（无家具，0-54）
    """
    data = np.load(npy_path)  # (N, 2, H, W) uint8
    assert data.ndim == 4, f"Expected 4D array, got {data.ndim}D: {data.shape}"
    assert sample_index < data.shape[0], f"Sample index {sample_index} out of range {data.shape[0]}"
    
    condition = data[sample_index]  # (2, H, W)
    return condition


def condition_to_image(
    condition: np.ndarray,
    palette: Optional[dict] = None,
    id2c: Optional[Dict] = None,
    c2rgb: Optional[Dict] = None,
) -> Image.Image:
    """
    将条件语义图 (2, H, W) 转换为可视化 PNG 图像
    
    遵循 parse_json_floorplan.py 的可视化格式，使用彩色映射。
    
    Args:
        condition: (2, H, W) uint8 数据，或 (1024, 1024) 语义图
        palette: {label_id -> (R, G, B)} 的颜色映射（已弃用，使用 _VIZ_COLORMAP）
        id2c: {category_id -> category_name} 映射（用于调试）
        c2rgb: {category_name -> [R, G, B]} 颜色映射（已弃用）
    
    Returns:
        Image: RGB 模式的 PIL 图像，颜色与 parse_json_floorplan.py 一致
    """
    # 提取语义通道
    if condition.ndim == 3:
        # (2, H, W) -> 取 Channel 1（语义图）
        semantic_map = condition[1]  # (H, W)
    else:
        # 已经是 (H, W)
        semantic_map = condition
    
    h, w = semantic_map.shape
    rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 使用统一的颜色映射表
    for label_id in np.unique(semantic_map):
        color = _cat_to_color(int(label_id))
        mask = semantic_map == label_id
        rgb_array[mask] = color
    
    return Image.fromarray(rgb_array, mode='RGB')


def image_to_semantic_map(
    image: Image.Image,
    id2c: Optional[Dict] = None,
    c2rgb: Optional[Dict] = None,
) -> np.ndarray:
    """
    将推理生成的 PIL Image (RGB) 转回语义标签 NPY (H, W)
    
    反向过程：颜色 → 语义 ID
    
    Args:
        image: PIL Image (RGB)
        id2c: {category_id -> category_name} 映射（已弃用）
        c2rgb: {category_name -> [R, G, B]} 颜色映射（已弃用）
    
    Returns:
        semantic_map: (H, W) uint8，值范围 0-54
    """
    rgb = np.asarray(image, dtype=np.uint8)
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"Expected RGB image, got shape {rgb.shape}"
    
    h, w = rgb.shape[:2]
    semantic_map = np.zeros((h, w), dtype=np.uint8)
    
    # 构建反向色表（使用统一的颜色映射）
    color_to_label = {}
    for label_id in range(55):
        color = _cat_to_color(label_id)
        color_to_label[color] = label_id
    
    # 文字标注色（来自 parse_json_floorplan.py 的 save_visualization_with_dimensions）
    # RGB(60, 60, 60) 映射为 VOID
    color_to_label[(60, 60, 60)] = VOID_ID
    
    # 映射像素
    rgb_flat = rgb.reshape(-1, 3)
    semantic_flat = semantic_map.reshape(-1)
    
    for idx, (r, g, b) in enumerate(rgb_flat):
        color = (int(r), int(g), int(b))
        if color in color_to_label:
            semantic_flat[idx] = color_to_label[color]
        else:
            # 找最近的颜色（欧氏距离）
            min_dist = float('inf')
            best_label = 0
            for known_color, known_label in color_to_label.items():
                dist = (r - known_color[0])**2 + (g - known_color[1])**2 + (b - known_color[2])**2
                if dist < min_dist:
                    min_dist = dist
                    best_label = known_label
            semantic_flat[idx] = best_label
    
    return semantic_map


def run_inference(
    condition_npy_path: Path,
    lora_weights_path: Path,
    output_dir: Path,
    metadata_json_path: Optional[Path] = None,
    gt_npy_path: Optional[Path] = None,
    condition_png_dir: Optional[Path] = None,
    max_samples: int = 0,
    embedded_guidance: float = 2.5,
    seed_start: int = 42,
    sample_start: int = 0,
    sample_end: int = -1,
) -> None:
    """
    运行 FLUX.1-Kontext 推理（DiffSynth-Studio 框架）
    
    Args:
        condition_npy_path: 条件 NPY 文件路径 (N, 2, H, W)，无家具
        lora_weights_path: LoRA 权重文件路径 (.safetensors)
        output_dir: 输出目录
        metadata_json_path: 可选的元数据 JSON（包含提示词）
        gt_npy_path: 可选的GT NPY文件路径 (N, 2, H, W)，带家具
        condition_png_dir: 可选的条件PNG目录，包含带尺寸标注的条件图（来自 parse_json_floorplan.py）
        max_samples: 最大样本数，0 表示全部
        embedded_guidance: 嵌入制导强度
        seed_start: 初始种子
        sample_start: 起始样本索引（用于分布式推理）
        sample_end: 结束样本索引（用于分布式推理），-1表示到最后
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子文件夹
    npy_dir = output_dir / "npy"
    img_dir = output_dir / "images"
    npy_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    if not DIFFSYNTH_AVAILABLE:
        print("错误: DiffSynth-Studio 框架不可用，无法加载 FLUX 管道")
        return
    
    print(f"[1] 加载条件数据: {condition_npy_path}")
    condition_data = np.load(condition_npy_path)
    num_samples = condition_data.shape[0]
    print(f"    条件数据形状: {condition_data.shape}")
    
    # 加载 GT 数据（可选）
    gt_data = None
    if gt_npy_path and gt_npy_path.exists():
        print(f"[1b] 加载 GT 数据: {gt_npy_path}")
        gt_data = np.load(gt_npy_path)
        print(f"    GT 数据形状: {gt_data.shape}")
        assert gt_data.shape[0] == num_samples, f"GT样本数 {gt_data.shape[0]} 与条件样本数 {num_samples} 不匹配"
    else:
        print(f"[1b] 跳过 GT 数据（未提供或文件不存在）")
    
    print(f"[2] 加载 LoRA 权重: {lora_weights_path}")
    lora_config = ModelConfig(path=str(lora_weights_path))
    print(f"    LoRA 配置已准备")
    
    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"[2.5] 检测到 {num_gpus} 个GPU")
    if num_gpus > 1:
        print(f"    将使用所有 {num_gpus} 个GPU进行推理（通过轮转调度）")
    
    print(f"[3] 初始化 FLUX.1-Kontext 管道")
    try:
        # DiffSynth-Studio 的 FluxImagePipeline 只支持 device="cuda"
        # 多GPU支持通过显式轮转实现
        pipe = FluxImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path="/share/home/202230550120/models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors"),
                ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/text_encoder/model.safetensors"),
                ModelConfig(path=[
                    "/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors",
                    "/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors"
                ]),
                ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/ae.safetensors"),
            ],
            tokenizer_1_config=ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/tokenizer/"),
            tokenizer_2_config=ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/tokenizer_2/"),
        )
        
        # 加载 LoRA 权重
        pipe.load_lora(pipe.dit, lora_config, alpha=1)
        print(f"    ✓ 管道加载成功，LoRA 权重已融合（使用 {num_gpus} 个GPU）")
    except Exception as e:
        print(f"    ✗ 管道加载失败: {e}")
        print("    提示: 检查模型路径是否正确")
        return
    
    # 加载元数据（提示词和房间信息）
    metadata = {}
    metadata_list = []
    if metadata_json_path and metadata_json_path.exists():
        with open(metadata_json_path, 'r') as f:
            metadata_list = json.load(f)
            # 支持两种格式：
            # 1. 如果 item 有 sample_index/idx 字段，则使用该字段
            # 2. 否则按照列表顺序对应（item[i] -> sample_i）
            for idx, item in enumerate(metadata_list):
                # 保存完整的元数据对象（包括尺寸和多边形信息）
                actual_idx = idx
                if "sample_index" in item:
                    actual_idx = item["sample_index"]
                elif "idx" in item:
                    actual_idx = item["idx"]
                metadata[actual_idx] = item
            
            print(f"    ✓ 加载元数据成功，共 {len(metadata)} 条记录")
    else:
        metadata = {}
    
    print(f"[4] 开始推理 (max_samples={max_samples})")
    
    # 确定要推理的样本索引
    sample_indices = list(range(num_samples))
    
    # 应用样本分片（用于分布式推理）
    if sample_end == -1:
        sample_end = num_samples
    sample_indices = sample_indices[sample_start:sample_end]
    
    if max_samples > 0:
        sample_indices = sample_indices[:max_samples]
    
    if sample_start > 0 or sample_end < num_samples:
        print(f"    分布式推理模式: 处理样本 {sample_start}-{min(sample_end, num_samples)-1}")
    
    successful = 0
    failed = 0
    failed_samples = []
    
    # 统计每个GPU的处理样本数
    gpu_stats = {i: 0 for i in range(num_gpus)}
    current_gpu = 0
    
    for idx in tqdm(sample_indices, desc="推理进度"):
        try:
            # ===== 多GPU轮转推理 =====
            # 在推理前，切换到当前GPU并清空缓存
            if num_gpus > 1:
                torch.cuda.set_device(current_gpu)
                torch.cuda.empty_cache()
                gpu_stats[current_gpu] += 1
                current_gpu = (current_gpu + 1) % num_gpus
            else:
                torch.cuda.empty_cache()
            
            # ===== 加载条件图像 =====
            # 从metadata获取房间信息，用于查找条件PNG
            condition_image = None
            condition_png_used = False
            
            # 首先尝试从metadata中获取plan_id和room_id来定位PNG文件
            if condition_png_dir and idx in metadata:
                room_meta = metadata[idx]
                plan_id = room_meta.get("plan_id", "")
                room_id = room_meta.get("room_id", "")
                room_name = room_meta.get("room_name", "")
                
                # 构建文件名：{plan_id}_{room_id}_{room_name}_no_furn.png
                if plan_id and room_id and room_name:
                    condition_png_filename = f"{plan_id}_{room_id}_{room_name}_no_furn.png"
                    condition_png_path = Path(condition_png_dir) / condition_png_filename
                    
                    if condition_png_path.exists():
                        try:
                            condition_image = Image.open(condition_png_path)
                            condition_png_used = True
                        except Exception as e:
                            pass  # 静默失败，继续降级
            
            # 降级：如果没找到PNG，尝试 sample_{idx}_condition.png 格式
            if condition_image is None and condition_png_dir:
                condition_png_path = Path(condition_png_dir) / f"sample_{idx}_condition.png"
                if condition_png_path.exists():
                    try:
                        condition_image = Image.open(condition_png_path)
                        condition_png_used = True
                    except Exception as e:
                        pass
            
            # 最后降级：从NPY生成（无文字）
            if condition_image is None:
                condition = load_condition_npy(condition_npy_path, idx)  # (2, H, W)
                condition_image = condition_to_image(condition)  # RGB PIL Image
            
            # 获取提示词和房间信息
            prompt = "2D semantic segmentation map, top-down floorplan, simple solid color blocks."
            room_meta = None
            if idx in metadata:
                room_meta = metadata[idx]
                prompt = room_meta.get("prompt_en", prompt)
            
            # 推理（在轮转的GPU上执行）
            with torch.no_grad():
                # 多GPU模式下转移管道到当前GPU
                if num_gpus > 1:
                    gpu_id = current_gpu - 1 if current_gpu > 0 else num_gpus - 1
                    try:
                        # 尝试将管道移到指定GPU
                        pipe.to(f"cuda:{gpu_id}")
                    except Exception as e:
                        # 如果不支持直接移动，使用 torch.cuda.set_device()
                        torch.cuda.set_device(gpu_id)
                
                generated_image = pipe(
                    prompt=prompt,
                    kontext_images=condition_image,
                    embedded_guidance=embedded_guidance,
                    seed=seed_start + idx,
                )
            
            # 转回语义图
            pred_semantic_map = image_to_semantic_map(generated_image)
            assert pred_semantic_map.shape == (1024, 1024), f"Expected (1024, 1024), got {pred_semantic_map.shape}"
            assert pred_semantic_map.dtype == np.uint8, f"Expected uint8, got {pred_semantic_map.dtype}"
            
            # 保存 NPY 文件
            output_npy_path = npy_dir / f"sample_{idx}_pred.npy"
            np.save(output_npy_path, pred_semantic_map)
            
            # 保存条件输入图（已在推理时使用，可能包含尺寸文字）
            output_cond_path = img_dir / f"sample_{idx}_00_condition.png"
            condition_image.save(output_cond_path)
            
            # 保存推理结果图（改为 PNG 无损保存）
            output_gen_path = img_dir / f"sample_{idx}_01_generated.png"
            generated_image.save(output_gen_path)
            
            # 加载GT图像（优先从PNG，否则从NPY生成）
            gt_image = None
            
            # 首先尝试从PNG加载GT图（带家具）
            if condition_png_dir and idx in metadata:
                room_meta = metadata[idx]
                plan_id = room_meta.get("plan_id", "")
                room_id = room_meta.get("room_id", "")
                room_name = room_meta.get("room_name", "")
                
                if plan_id and room_id and room_name:
                    gt_png_filename = f"{plan_id}_{room_id}_{room_name}_with_furn.png"
                    gt_png_path = Path(condition_png_dir) / gt_png_filename
                    
                    if gt_png_path.exists():
                        try:
                            gt_image = Image.open(gt_png_path)
                        except Exception as e:
                            pass  # 静默失败，继续降级
            
            # 降级：如果没有PNG或加载失败，从NPY生成GT
            if gt_image is None and gt_data is not None:
                gt_semantic = gt_data[idx, 1]  # 取Channel 1（语义图，包含家具）
                gt_semantic_rgb = np.zeros((*gt_semantic.shape, 3), dtype=np.uint8)
                for label_id in np.unique(gt_semantic):
                    color = _cat_to_color(int(label_id))
                    mask = gt_semantic == label_id
                    gt_semantic_rgb[mask] = color
                gt_image = Image.fromarray(gt_semantic_rgb, mode='RGB')
            
            # 如果有GT数据，保存GT图并拼接
            if gt_image is not None:
                
                # 保存GT图
                output_gt_path = img_dir / f"sample_{idx}_02_gt_with_furn.png"
                gt_image.save(output_gt_path)
                
                # 拼接：[GT | 间隙 | 推理结果]
                gap_width = 20
                label_height = 40
                
                combined_width = gt_image.width + gap_width + generated_image.width
                combined_height = generated_image.height + label_height
                combined_image = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
                
                combined_image.paste(gt_image, (0, label_height))
                combined_image.paste(generated_image, (gt_image.width + gap_width, label_height))
                
                # 添加标签
                draw = ImageDraw.Draw(combined_image)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
                except:
                    font = ImageFont.load_default()
                
                draw.text((gt_image.width // 2 - 50, 10), "Ground Truth", fill=(0, 0, 0), font=font)
                draw.text((gt_image.width + gap_width + generated_image.width // 2 - 50, 10), "Prediction", fill=(0, 0, 0), font=font)
                
                output_combined_path = img_dir / f"sample_{idx}_03_combined.png"
                combined_image.save(output_combined_path)
            else:
                # 如果没有GT数据，仍然保存条件和推理结果的拼接（用于参考）
                gap_width = 20
                label_height = 40
                
                combined_width = condition_image.width + gap_width + generated_image.width
                combined_height = generated_image.height + label_height
                combined_image = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
                
                combined_image.paste(condition_image, (0, label_height))
                combined_image.paste(generated_image, (condition_image.width + gap_width, label_height))
                
                # 添加标签
                draw = ImageDraw.Draw(combined_image)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
                except:
                    font = ImageFont.load_default()
                
                draw.text((condition_image.width // 2 - 50, 10), "Condition", fill=(0, 0, 0), font=font)
                draw.text((condition_image.width + gap_width + generated_image.width // 2 - 50, 10), "Prediction", fill=(0, 0, 0), font=font)
                
                output_combined_path = img_dir / f"sample_{idx}_03_combined.png"
                combined_image.save(output_combined_path)
            
            successful += 1
            
        except Exception as e:
            print(f"\n[样本 {idx}] 推理失败: {e}")
            failed += 1
            failed_samples.append(idx)
    
    print(f"\n[5] 推理完成")
    print(f"    ✓ 成功: {successful}/{len(sample_indices)}")
    if failed > 0:
        print(f"    ✗ 失败: {failed}")
        print(f"    失败样本: {failed_samples[:10]}")  # 显示前10个
    
    # 打印GPU使用统计
    if num_gpus > 1:
        print(f"    GPU 使用统计:")
        for gpu_id, count in gpu_stats.items():
            if count > 0:
                print(f"      GPU {gpu_id}: {count} 个样本")
    
    print(f"    输出目录: {output_dir}")
    print(f"    输出文件结构:")
    print(f"      {output_dir}/npy/")
    print(f"        - sample_{{idx}}_pred.npy: 预测语义图")
    print(f"      {output_dir}/images/")
    print(f"        - sample_{{idx}}_00_condition.png: 条件图（无家具）")
    print(f"        - sample_{{idx}}_01_generated.png: 推理结果")
    if gt_data is not None:
        print(f"        - sample_{{idx}}_02_gt_with_furn.png: GT图（有家具）")
        print(f"        - sample_{{idx}}_03_combined.png: 对比图（GT | 推理结果）")
    else:
        print(f"        - sample_{{idx}}_03_combined.png: 对比图（条件 | 推理结果）")


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="FLUX.1-Kontext NPY 推理脚本")
    
    parser.add_argument(
        "--condition-npy-dir",
        type=str,
        default="output_2026.3.27",
        help="条件 NPY 文件所在目录"
    )
    parser.add_argument(
        "--condition-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="数据集分割"
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        required=True,
        help="LoRA 权重文件路径 (.safetensors)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录（保存 *_pred.npy 文件）"
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="可选的元数据 JSON 文件（包含提示词）"
    )
    parser.add_argument(
        "--gt-npy-path",
        type=str,
        default=None,
        help="可选的GT NPY文件路径（带家具），用于拼接对比"
    )
    parser.add_argument(
        "--condition-png-dir",
        type=str,
        default=None,
        help="可选的条件PNG目录，包含带尺寸标注的条件图（来自 parse_json_floorplan.py）"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最大样本数，0 表示全部"
    )
    parser.add_argument(
        "--embedded-guidance",
        type=float,
        default=2.5,
        help="嵌入制导强度"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="初始种子"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="指定要使用的GPU ID（逗号分隔），例如 0,1,2,3。默认使用所有可用GPU"
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="起始样本索引（用于分布式推理）"
    )
    parser.add_argument(
        "--sample-end",
        type=int,
        default=-1,
        help="结束样本索引（用于分布式推理），-1表示到最后"
    )
    
    args = parser.parse_args()
    
    # 设置环境变量来指定GPU
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # 构建完整路径
    condition_npy_path = Path(args.condition_npy_dir) / f"chinese_{args.condition_split}_condition_1024x1024.npy"
    metadata_json_path = Path(args.metadata_json) if args.metadata_json else None
    gt_npy_path = Path(args.gt_npy_path) if args.gt_npy_path else None
    condition_png_dir = Path(args.condition_png_dir) if args.condition_png_dir else None
    
    if not condition_npy_path.exists():
        print(f"错误: 条件 NPY 文件不存在: {condition_npy_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"FLUX.1-Kontext NPY 推理脚本")
    print(f"{'='*60}")
    print(f"条件 NPY:     {condition_npy_path}")
    print(f"LoRA 权重:     {args.lora_weights}")
    print(f"输出目录:     {args.output_dir}")
    print(f"指导强度:     {args.embedded_guidance}")
    print(f"{'='*60}\n")
    
    run_inference(
        condition_npy_path=condition_npy_path,
        lora_weights_path=Path(args.lora_weights),
        output_dir=args.output_dir,
        metadata_json_path=metadata_json_path,
        gt_npy_path=gt_npy_path,
        condition_png_dir=condition_png_dir,
        max_samples=args.max_samples,
        embedded_guidance=args.embedded_guidance,
        seed_start=args.seed_start,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
    )


if __name__ == "__main__":
    main()
