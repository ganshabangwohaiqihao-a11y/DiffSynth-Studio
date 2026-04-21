#!/usr/bin/env python3
"""
Generate a visualization table of furniture categories and their colors.
Creates a PNG image showing all category IDs with their names and color swatches.
"""
# 生成家具类别可视化参考表，创建一个 PNG 图像，展示所有 55 个家具类别的ID、名称和颜色
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ──────────────────────────────────────────────────────────────────────────────
# Category and Color Definitions (from parse_json_floorplan.py)
# ──────────────────────────────────────────────────────────────────────────────

CATEGORIES = {
    0:  "void",
    1:  "floor",
    2:  "kids_bed",
    3:  "single_bed",
    4:  "double_bed",
    5:  "corner_side_table",
    6:  "round_end_table",
    7:  "coffee_table",
    8:  "console_table",
    9:  "tv_stand",
    10: "desk",
    11: "dressing_table",
    12: "table",
    13: "dining_table",
    14: "stool",
    15: "dressing_chair",
    16: "dining_chair",
    17: "chinese_chair",
    18: "armchair",
    19: "chair",
    20: "lounge_chair",
    21: "loveseat_sofa",
    22: "lazy_sofa",
    23: "sofa",
    24: "multi_seat_sofa",
    25: "chaise_longue_sofa",
    26: "l_shaped_sofa",
    27: "nightstand",
    28: "shelf",
    29: "bookshelf",
    30: "children_cabinet",
    31: "wine_cabinet",
    32: "cabinet",
    33: "wardrobe",
    34: "pendant_lamp",
    35: "ceiling_lamp",
    36: "door",
    37: "window",
    38: "bathtub",
    39: "toilet",
    40: "wash_basin",
    41: "shower",
    42: "refrigerator",
    43: "cooking_appliance",
    44: "piano",
    45: "washing_machine",
    46: "air_conditioner",
    47: "display_screen",
    48: "kitchen_cabinet",
    49: "bathroom_cabinet",
    50: "fitness_equipment",
    51: "shoe_cabinet",
    52: "tatami",
    53: "kitchen_island",
    54: "cove_ceiling",
}

_VIZ_COLORMAP = {
    0:   (0,   0,   0),        # void
    1:   (200, 200, 200),      # floor
    36:  (255, 120,  0),       # door
    37:  (0,  180, 255),       # window
    2:   (200,  80,  80),      # kids_bed
    3:   (200,  80,  80),      # single_bed
    4:   (200,  80,  80),      # double_bed
    10:  (80, 140, 200),       # desk
    13:  (100, 200, 100),      # dining_table
    23:  (200, 130,  50),      # sofa
    33:  (160,  60, 160),      # wardrobe
    39:  (80, 200, 200),       # toilet
    42:  (200, 200,  60),      # refrigerator
}

def _cat_to_color(cat_id: int) -> tuple:
    """Get color for a category, using predefined map or generating randomly."""
    if cat_id in _VIZ_COLORMAP:
        return _VIZ_COLORMAP[cat_id]
    # Generate deterministic color from category ID
    rng = np.random.RandomState(cat_id * 17 + 3)
    return tuple(int(x) for x in rng.randint(60, 230, 3))


# ──────────────────────────────────────────────────────────────────────────────
# Generate Visualization
# ──────────────────────────────────────────────────────────────────────────────

def generate_colormap_table(output_path: str, swatch_width: int = 18, swatch_height: int = 8):
    """
    Generate a compact visualization table of all categories with colors.
    Each entry: [ID] name + small rectangular color swatch, tightly arranged.
    
    Parameters
    ----------
    output_path : str
        Path to save the PNG image
    swatch_width : int
        Width of the rectangular color swatch
    swatch_height : int
        Height of the rectangular color swatch
    """
    
    num_cats = len(CATEGORIES)
    cols = 8  # 8 categories per row for compact layout
    rows = (num_cats + cols - 1) // cols  # Ceiling division
    
    # Try to load a nice font, fall back to default
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font_label = ImageFont.load_default()
    
    # First pass: measure text dimensions to calculate cell size
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    
    text_heights = []
    text_widths = []
    for cat_id in range(num_cats):
        cat_name = CATEGORIES.get(cat_id, "unknown")
        label = f"[{cat_id:2d}] {cat_name}"
        bbox = temp_draw.textbbox((0, 0), label, font=font_label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_widths.append(text_width)
        text_heights.append(text_height)
    
    max_text_height = max(text_heights) if text_heights else 12
    max_text_width = max(text_widths) if text_widths else 100
    
    # Cell dimensions - tight packing
    gap_after_text = 4  # Small gap between swatch and text
    cell_width = swatch_width + gap_after_text + max_text_width + 6  # swatch + gap + text + margins
    cell_height = max(max_text_height, swatch_height) + 4  # Minimal vertical padding
    
    horizontal_gap = 2   # Gap between cells horizontally
    vertical_gap = 2     # Gap between rows vertically
    
    img_width = cols * cell_width + (cols - 1) * horizontal_gap + 20
    img_height = rows * cell_height + (rows - 1) * vertical_gap + 35
    
    # Create image with white background
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw title
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        title_font = ImageFont.load_default()
    
    title = "Furniture Category Reference"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (img_width - title_width) // 2
    draw.text((title_x, 6), title, fill=(0, 0, 0), font=title_font)
    
    # Draw each category
    for cat_id in sorted(CATEGORIES.keys()):
        row = cat_id // cols
        col = cat_id % cols
        
        # Calculate position with minimal gaps
        x = col * (cell_width + horizontal_gap) + 10
        y = row * (cell_height + vertical_gap) + 28
        
        # Get color
        color = _cat_to_color(cat_id)
        
        # Draw color swatch first (on the left), vertically centered
        swatch_x = x
        swatch_y = y + (max_text_height - swatch_height) // 2
        draw.rectangle(
            [swatch_x, swatch_y, swatch_x + swatch_width, swatch_y + swatch_height],
            fill=color,
            outline=(0, 0, 0),
            width=1
        )
        
        # Draw ID and name right after swatch
        cat_name = CATEGORIES.get(cat_id, "unknown")
        label = f"[{cat_id:2d}] {cat_name}"
        text_x = x + swatch_width + gap_after_text
        draw.text((text_x, y + 1), label, fill=(0, 0, 0), font=font_label)
    
    # Save image
    img.save(output_path)
    print(f"✅ Saved category colormap visualization to: {output_path}")
    print(f"   Image size: {img_width} x {img_height} pixels")
    print(f"   Layout: {cols} columns × {rows} rows")
    print(f"   Swatch size: {swatch_width}×{swatch_height} (rectangle)")
    print(f"   Total categories: {num_cats}")


if __name__ == "__main__":
    output_dir = "/share/home/202230550120/diffusers/output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "category_colormap_visualization.png")
    generate_colormap_table(output_path, swatch_width=18, swatch_height=8)
    
    print("\n📊 Category Colormap Summary:")
    print(f"    Total categories: {len(CATEGORIES)}")
    print(f"    Explicitly mapped colors: {len(_VIZ_COLORMAP)}")
    print(f"    Auto-generated colors: {len(CATEGORIES) - len(_VIZ_COLORMAP)}")
