#!/usr/bin/env python3
"""
户型图家具移除脚本
从GT图中自动去除家具，保留纯结构信息
"""

import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path


def remove_furniture_morphology(image_path, output_path, kernel_size=5):
    """
    方法1：基于形态学操作的家具移除
    保留主要结构（墙体、门窗），移除非结构元素
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"⚠️  无法读取图像: {image_path}")
        return False
    
    # 图像反转（墙体是黑色时）
    # 可根据实际图像调整
    # img = cv2.bitwise_not(img)
    
    # 应用闭运算（先膨胀后腐蚀）- 连接断裂的线，填充小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 应用开运算（先腐蚀后膨胀）- 移除小对象（家具细节）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 保存结果
    cv2.imwrite(str(output_path), opened)
    print(f"✓ 已保存: {output_path}")
    return True


def remove_furniture_contour(image_path, output_path, area_threshold=500):
    """
    方法2：基于轮廓分析的家具移除
    检测到的小轮廓认为是家具，予以移除
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"⚠️  无法读取图像: {image_path}")
        return False
    
    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建输出图
    output_img = np.ones_like(binary) * 255  # 白色背景
    
    # 只绘制面积大的轮廓（主要结构）
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            cv2.drawContours(output_img, [contour], 0, 0, -1)
    
    cv2.imwrite(str(output_path), output_img)
    print(f"✓ 已保存: {output_path}")
    return True


def remove_furniture_inpainting(image_path, output_path, 
                                mask_path=None, dilate_kernel=15):
    """
    方法3：使用inpainting修复
    如果提供mask，使用mask标记家具区域，然后用inpainting修复
    需要OpenCV的inpainting功能
    """
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"⚠️  无法读取图像: {image_path}")
        return False
    
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        # 自动生成mask - 检测非黑色、非白色的中间灰度（可能是家具）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 家具通常在128附近的灰度值
        mask = cv2.inRange(gray, 80, 180)
        # 膨胀mask以包含周边
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 使用Telea's或Navier-Stokes inpainting
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    cv2.imwrite(str(output_path), inpainted)
    print(f"✓ 已保存: {output_path}")
    return True


def batch_process(input_dir, output_dir, method='morphology'):
    """
    批量处理目录中的G图像
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 找到所有GT图像（假设名称包含'gt'或'GT'）
    image_files = []
    for pattern in ['*gt*.png', '*GT*.png', '*gt*.jpg', '*GT*.jpg']:
        image_files.extend(input_path.glob(pattern))
    
    print(f"\n找到 {len(image_files)} 个GT图像\n")
    
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] 处理: {img_file.name}")
        
        output_file = output_path / img_file.name
        
        if method == 'morphology':
            remove_furniture_morphology(img_file, output_file)
        elif method == 'contour':
            remove_furniture_contour(img_file, output_file)
        elif method == 'inpainting':
            remove_furniture_inpainting(img_file, output_file)
    
    print(f"\n✓ 全部处理完成，输出目录: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="户型图家具移除工具")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./",
        help="输入GT图像目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gt_no_furniture",
        help="输出目录"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['morphology', 'contour', 'inpainting'],
        default='morphology',
        help="家具移除方法: morphology(形态学), contour(轮廓), inpainting(修复)"
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="单个图像处理（处理单个文件而非整个目录）"
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default=None,
        help="单个图像输出路径"
    )
    
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              户型图家具移除工具 v1.0                         ║
║              Furniture Removal for Floor Plans               ║
╚═══════════════════════════════════════════════════════════════╝

当前方法: {args.method}

方法说明:
  - morphology: 形态学运算，快速简洁 ✓ 推荐首选
  - contour:    轮廓分析，精确度高
  - inpainting: 修复算法，效果逼真（需更多显存）

""")
    
    if args.input_image:
        # 处理单个图像
        output_img = args.output_image or Path(args.input_image).stem + "_no_furniture.png"
        
        if args.method == 'morphology':
            remove_furniture_morphology(args.input_image, output_img)
        elif args.method == 'contour':
            remove_furniture_contour(args.input_image, output_img)
        elif args.method == 'inpainting':
            remove_furniture_inpainting(args.input_image, output_img)
    else:
        # 批量处理目录
        batch_process(args.input_dir, args.output_dir, args.method)


if __name__ == "__main__":
    main()
