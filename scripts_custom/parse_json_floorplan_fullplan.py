#!/usr/bin/env python3
"""
Full Floor Plan Generator (No Room Segmentation, No Furniture)
================================================================

Based on parse_json_floorplan.py, but generates:
  - ONE complete floor plan per JSON (full canvas, no room-by-room segmentation)
  - ONLY structural elements (walls, doors, windows)
  - NO furniture
  - High-resolution PNG output (1024×1024 or custom size)

Usage
-----
  python parse_json_floorplan_fullplan.py \
      --input_dir /share/home/202230550120/diffusers/plan_json_0228 \
      --output_dir ./full_floorplans_no_furniture \
      --output_size 1024 1024 \
      --visualize
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Minimal Category Definitions (structural elements only)
# ──────────────────────────────────────────────────────────────────────────────

# Semantic IDs for architectural elements (no furniture)
VOID_ID    = 0
FLOOR_ID   = 1
WALL_ID    = 2
DOOR_ID    = 3
WINDOW_ID  = 4

# Hole type string → semantic ID
HOLE_TYPE_MAP: Dict[str, int] = {
    "门": DOOR_ID,
    "推拉门": DOOR_ID,
    "折叠门": DOOR_ID,
    "洞": DOOR_ID,
    "窗": WINDOW_ID,
    "窗户": WINDOW_ID,
    "飘窗": WINDOW_ID,
}


def _extract_points(raw_points: List[Dict[str, float]]) -> List[Tuple[float, float]]:
    """Convert JSON point objects [{x, y}, ...] to [(x, y), ...]."""
    pts: List[Tuple[float, float]] = []
    for p in raw_points or []:
        if isinstance(p, dict) and "x" in p and "y" in p:
            pts.append((float(p["x"]), float(p["y"])))
    return pts


def _polygon_area_cm2(points: List[Tuple[float, float]]) -> float:
    """Shoelace polygon area in cm^2 (JSON coordinate unit)."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Polygon centroid in source coordinates. Falls back to vertex mean for degenerate polygons."""
    if not points:
        return 0.0, 0.0
    if len(points) < 3:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    signed = 0.0
    cx = 0.0
    cy = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        a = x0 * y1 - x1 * y0
        signed += a
        cx += (x0 + x1) * a
        cy += (y0 + y1) * a

    if abs(signed) < 1e-6:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    signed *= 0.5
    cx /= (6.0 * signed)
    cy /= (6.0 * signed)
    return float(cx), float(cy)

# ──────────────────────────────────────────────────────────────────────────────
# JSON Parsing (reused from original)
# ──────────────────────────────────────────────────────────────────────────────

def parse_floorplan(json_path: str) -> Dict[str, Any]:
    """Parse a floor plan JSON file using the same field conventions as parse_json_floorplan.py."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    canvas_size: List[float] = data.get("size", [1200, 1200])
    level_height: int = data.get("levelHeight", 280)

    # Holes (doors, windows)
    holes: Dict[str, Dict] = {}
    for h in data.get("holeData", []):
        hole_id = h.get("id", f"hole_{len(holes)}")
        hole_type = h.get("type", "门")
        holes[hole_id] = {
            "id": hole_id,
            "type": HOLE_TYPE_MAP.get(hole_type, DOOR_ID),
            "raw_type": hole_type,
            "points": _extract_points(h.get("points", [])),
        }

    # Walls
    walls: Dict[str, Dict] = {}
    for w in data.get("wallData", []):
        wall_id = w.get("id", f"wall_{len(walls)}")
        walls[wall_id] = {
            "id": wall_id,
            "points": _extract_points(w.get("points", [])),
            "center_line": _extract_points(w.get("centerLinePoints", [])),
            "thickness": w.get("thickness", 20),
        }

    # Rooms (kept only for metadata; we do not split by room)
    rooms: List[Dict] = []
    for r in data.get("roomData", []):
        rooms.append({
            "id": r.get("id", f"room_{len(rooms)}"),
            "name": r.get("name", ""),
            "points": _extract_points(r.get("points", [])),
        })

    return {
        "canvas_size": canvas_size,
        "level_height": level_height,
        "rooms": rooms,
        "walls": walls,
        "holes": holes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full Floor Plan Rendering (entire canvas, no segmentation)
# ──────────────────────────────────────────────────────────────────────────────

def render_full_floorplan(
    data: Dict[str, Any],
    output_size: Tuple[int, int] = (1024, 1024),
) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Render the complete floor plan at full canvas scale.
    
    Returns
    -------
    semantic_map : np.ndarray
        Shape (H, W) uint8 semantic map (WALL_ID, DOOR_ID, WINDOW_ID, VOID_ID)
    scale_x, scale_y : float
        Pixel-per-unit scale factors
    """
    OH, OW = output_size
    
    canvas_size = data["canvas_size"]
    canvas_w_cm, canvas_h_cm = canvas_size[0], canvas_size[1]
    
    # Compute scale: fit canvas to output size
    scale_x = OW / canvas_w_cm if canvas_w_cm > 0 else 1.0
    scale_y = OH / canvas_h_cm if canvas_h_cm > 0 else 1.0
    
    # Initialize with VOID (white background)
    semantic_map = np.full((OH, OW), VOID_ID, dtype=np.uint8)

    # Draw rooms (fill room polygons with FLOOR_ID)
    for room in data["rooms"]:
        pts = room.get("points", [])
        if len(pts) >= 3:
            room_poly = []
            for x, y in pts:
                px = int(round(x * scale_x))
                py = int(round(y * scale_y))
                room_poly.append([
                    int(np.clip(px, 0, OW - 1)),
                    int(np.clip(py, 0, OH - 1)),
                ])
            cv2.fillPoly(semantic_map, [np.array(room_poly, dtype=np.int32)], FLOOR_ID)

    # Draw walls first (filled polygons from wallData.points)
    for wall in data["walls"].values():
        pts = wall.get("points", [])
        if len(pts) >= 3:
            wall_poly = []
            for x, y in pts:
                px = int(round(x * scale_x))
                py = int(round(y * scale_y))
                wall_poly.append([
                    int(np.clip(px, 0, OW - 1)),
                    int(np.clip(py, 0, OH - 1)),
                ])
            cv2.fillPoly(semantic_map, [np.array(wall_poly, dtype=np.int32)], WALL_ID)
        elif len(pts) == 2:
            # Fallback for line-style walls
            (x1, y1), (x2, y2) = pts
            thickness_px = max(1, int(round(float(wall.get("thickness", 20)) * scale_x)))
            cv2.line(
                semantic_map,
                (int(round(x1 * scale_x)), int(round(y1 * scale_y))),
                (int(round(x2 * scale_x)), int(round(y2 * scale_y))),
                WALL_ID,
                thickness_px,
            )
    
    # Draw holes (doors/windows) on top of walls
    for hole in data["holes"].values():
        hole_pts = hole.get("points", [])
        hole_type = int(hole.get("type", DOOR_ID))
        if len(hole_pts) >= 3:
            poly = []
            for x, y in hole_pts:
                px = int(round(x * scale_x))
                py = int(round(y * scale_y))
                poly.append([
                    int(np.clip(px, 0, OW - 1)),
                    int(np.clip(py, 0, OH - 1)),
                ])
            cv2.fillPoly(semantic_map, [np.array(poly, dtype=np.int32)], hole_type)
        elif len(hole_pts) == 2:
            (x1, y1), (x2, y2) = hole_pts
            cv2.line(
                semantic_map,
                (int(round(x1 * scale_x)), int(round(y1 * scale_y))),
                (int(round(x2 * scale_x)), int(round(y2 * scale_y))),
                hole_type,
                max(1, int(round(4 * scale_x))),
            )
    
    return semantic_map, scale_x, scale_y


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

_VIZ_COLORMAP: Dict[int, Tuple[int, int, int]] = {
    VOID_ID:    (255, 255, 255),  # White - empty space
    FLOOR_ID:   (200, 200, 200),  # Light gray - floor/room interior
    WALL_ID:    (30, 30, 30),     # Black-ish - walls
    DOOR_ID:    (255, 120, 0),    # Orange - doors
    WINDOW_ID:  (0, 180, 255),    # Light blue - windows
}

def _cat_to_color(cat_id: int) -> Tuple[int, int, int]:
    """Get RGB color for a semantic category ID."""
    if cat_id in _VIZ_COLORMAP:
        return _VIZ_COLORMAP[cat_id]
    return (200, 200, 200)  # Default gray


def _find_cjk_font_path() -> Optional[str]:
    """Find an available CJK font path on the current machine."""
    # 先直接尝试fc-list，这是最可靠的方式
    try:
        out = subprocess.check_output(
            ["fc-list", ":lang=zh", "file"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        for line in out.splitlines():
            path = line.split(":", 1)[0].strip()
            if path:
                # 尝试通过尝试读取来判断是否可用（比os.path.isfile更可靠）
                try:
                    with open(path, 'rb') as f:
                        f.read(1)
                    return path
                except Exception:
                    continue
    except Exception:
        pass
    
    # 备选方案：直接尝试已知路径
    candidates = [
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/google-droid/DroidSansFallback.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        try:
            with open(p, 'rb') as f:
                f.read(1)
            return p
        except Exception:
            pass

    return None


def save_visualization(
    semantic_map: np.ndarray,
    rooms: List[Dict[str, Any]],
    scale_x: float,
    scale_y: float,
    out_path: str,
    annotate_room_name: bool = True,
    annotate_room_area: bool = True,
) -> None:
    """Save a color-coded PNG of the semantic map with optional room annotations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    
    h, w = semantic_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cat_id in np.unique(semantic_map):
        mask = semantic_map == cat_id
        color = _cat_to_color(cat_id)
        rgb[mask] = color

    # 使用matplotlib保存基础图像
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.imshow(rgb)
    ax.axis('off')
    
    # 设置中文字体（自动探测，避免固定路径在不同机器失效）
    cjk_font_path = _find_cjk_font_path()
    if cjk_font_path:
        zhfont = fm.FontProperties(fname=cjk_font_path, size=10)
    else:
        zhfont = fm.FontProperties(size=10)
        if not getattr(save_visualization, "_warned_no_cjk", False):
            # 输出诊断信息
            print("[WARN] No CJK font found. Attempting manual discovery...")
            manual_candidates = [
                "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
                "/usr/share/fonts/google-droid/DroidSansFallback.ttf",
            ]
            for candidate in manual_candidates:
                exists = os.path.isfile(candidate)
                print(f"  {candidate}: {exists}")
            print("[WARN] Chinese room names will be skipped to avoid garbled text.")
            save_visualization._warned_no_cjk = True

    # 添加房间标注
    if annotate_room_name or annotate_room_area:
        for room in rooms:
            pts = room.get("points", [])
            if len(pts) < 3:
                continue

            cx, cy = _polygon_centroid(pts)
            px = cx * scale_x
            py = cy * scale_y

            # Keep label anchor within image bounds
            px = np.clip(px, 0, w - 1)
            py = np.clip(py, 0, h - 1)

            label_parts: List[str] = []
            if annotate_room_name:
                room_name = str(room.get("name", "")).strip()
                if room_name and cjk_font_path:
                    label_parts.append(room_name)
            if annotate_room_area:
                area_m2 = _polygon_area_cm2(pts) / 10000.0
                # 若当前环境找不到CJK字体，使用ASCII单位避免乱码
                if cjk_font_path:
                    label_parts.append(f"{area_m2:.2f}㎡")
                else:
                    label_parts.append(f"{area_m2:.2f} m2")

            if not label_parts:
                continue

            label_text = "\n".join(label_parts)
            
            # 在matplotlib中绘制文本，去掉背景框
            ax.text(
                px, py, label_text,
                fontproperties=zhfont,
                fontsize=12,
                color='black',
                ha='center', va='center',
                weight='bold'
            )
    
    # 保存图像
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_semantic_map_npy(
    semantic_map: np.ndarray,
    out_path: str,
) -> None:
    """Save the semantic map as a NumPy file."""
    np.save(out_path, semantic_map.astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_all(
    input_dir: str,
    output_dir: str,
    output_size: Tuple[int, int] = (1024, 1024),
    visualize: bool = False,
    annotate_room_name: bool = True,
    annotate_room_area: bool = True,
) -> None:
    """
    Process all JSON floor plans in input_dir.
    Generate one complete floor plan per JSON file.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    png_dir = Path(output_dir) / "floor_plans_png"
    npy_dir = Path(output_dir) / "floor_plans_npy"
    viz_dir = Path(output_dir) / "visualizations"
    
    png_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover JSON files
    json_paths = sorted(glob.glob(os.path.join(input_dir, "room_*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No room_*.json files found in '{input_dir}'")
    
    print(f"\nFound {len(json_paths)} floor plan JSON files.")
    print(f"Output size: {output_size[0]}×{output_size[1]} pixels")
    print(f"Output directory: {output_dir}\n")
    
    results = []
    errors = 0
    
    for path in tqdm(json_paths, desc="Processing"):
        plan_id = os.path.basename(path).replace("room_", "").replace(".json", "")
        
        try:
            # Parse JSON
            fp_data = parse_floorplan(path)
            
            # Render full floor plan (no furniture, only walls/doors/windows)
            semantic_map, scale_x, scale_y = render_full_floorplan(
                fp_data,
                output_size=output_size,
            )
            
            if semantic_map is None:
                print(f"  ⚠️  {plan_id}: Failed to render")
                errors += 1
                continue
            
            # Save outputs
            png_path = str(png_dir / f"{plan_id}_floorplan.png")
            npy_path = str(npy_dir / f"{plan_id}_floorplan.npy")
            
            save_semantic_map_npy(semantic_map, npy_path)
            save_visualization(
                semantic_map,
                rooms=fp_data["rooms"],
                scale_x=scale_x,
                scale_y=scale_y,
                out_path=png_path,
                annotate_room_name=annotate_room_name,
                annotate_room_area=annotate_room_area,
            )
            
            # Optional visualization with filename
            if visualize:
                viz_path = str(viz_dir / f"{plan_id}_viz.png")
                save_visualization(
                    semantic_map,
                    rooms=fp_data["rooms"],
                    scale_x=scale_x,
                    scale_y=scale_y,
                    out_path=viz_path,
                    annotate_room_name=annotate_room_name,
                    annotate_room_area=annotate_room_area,
                )
            
            results.append({
                "plan_id": plan_id,
                "canvas_size": fp_data["canvas_size"],
                "num_rooms": len(fp_data["rooms"]),
                "num_walls": len(fp_data["walls"]),
                "num_holes": len(fp_data["holes"]),
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "output_png": png_path,
                "output_npy": npy_path,
            })
            
        except Exception as exc:
            print(f"  ✗ {plan_id}: {exc}")
            errors += 1
            continue
    
    # Save metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✓ Processed: {len(results)}")
    print(f"✗ Errors: {errors}")
    print(f"{'='*70}")
    print(f"\n📁 Output directories:")
    print(f"  PNG files:  {png_dir}")
    print(f"  NPY files:  {npy_dir}")
    if visualize:
        print(f"  Visualizations: {viz_dir}")
    print(f"  Metadata:   {meta_path}")
    print(f"\n✓ Done!")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate full floor plans (no room segmentation, no furniture)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/share/home/202230550120/diffusers/plan_json_0228",
        help="Directory containing room_*.json files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./full_floorplans_no_furniture",
        help="Output directory for PNG/NPY files",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="Output image size (height width)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization PNGs",
    )
    parser.add_argument(
        "--annotate_room_name",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to annotate room type/name text (1=yes, 0=no)",
    )
    parser.add_argument(
        "--annotate_room_area",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to annotate room area text in square meters (1=yes, 0=no)",
    )
    
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║    Full Floor Plan Generator (No Room Segmentation)          ║
║    基于JSON生成完整户型图（无房间分割、无家具）              ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    output_size = tuple(args.output_size)
    
    process_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_size=output_size,
        visualize=args.visualize,
        annotate_room_name=bool(args.annotate_room_name),
        annotate_room_area=bool(args.annotate_room_area),
    )


if __name__ == "__main__":
    main()
