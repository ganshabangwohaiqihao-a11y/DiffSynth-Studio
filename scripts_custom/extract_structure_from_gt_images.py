#!/usr/bin/env python3
"""
Extract structure-only maps (floor/wall/door/window) from styled GT floorplan images.

This script does NOT use condition maps or JSON. It works directly from GT rasters,
so it can handle cases where GT structure differs from condition input.

Semantic IDs:
  0 = void
  1 = floor
  2 = wall
  3 = door
  4 = window

Output:
  - floor_plans_png/*.png  (color visualization)
  - floor_plans_npy/*.npy  (semantic IDs)

Usage:
python extract_structure_from_gt_images.py \
  --input_dir /share/home/202230550120/extracted_images_col20 \
  --output_dir /share/home/202230550120/diffusers/scripts_custom/gt_no_furniture_from_gt \
  --wall_thresh 75 \
  --wall_min_thickness 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm


VOID_ID = 0
FLOOR_ID = 1
WALL_ID = 2
DOOR_ID = 3
WINDOW_ID = 4

ID_PATTERN = re.compile(r"edited_image_(\d+)\.png$")

VIZ_COLORS: Dict[int, Tuple[int, int, int]] = {
    VOID_ID: (255, 255, 255),
    FLOOR_ID: (200, 200, 200),
    WALL_ID: (30, 30, 30),
    DOOR_ID: (255, 120, 0),
    WINDOW_ID: (0, 180, 255),
}


def _to_viz_rgb(semantic: np.ndarray) -> np.ndarray:
    h, w = semantic.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in VIZ_COLORS.items():
        rgb[semantic == cid] = color
    return rgb


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out


def _extract_wall_mask(img_bgr: np.ndarray, wall_thresh: int, wall_min_thickness: int) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Absolute threshold branch.
    dark_abs = (gray <= wall_thresh).astype(np.uint8)

    # 2) Adaptive threshold branch for style-variant renders.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_otsu = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    dark = ((dark_abs | dark_otsu) > 0).astype(np.uint8)

    # Keep thick strokes and suppress tiny text/furniture lines.
    dist = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    thick = (dist >= float(wall_min_thickness)).astype(np.uint8)

    # Restore continuous wall regions.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, k, iterations=1)
    wall = cv2.dilate(wall, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Remove very small components.
    wall = _remove_small_components(wall, min_area=80)
    return wall


def _extract_floor_and_outside(wall_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = wall_mask.shape

    # Seal wall gaps before flood-fill; otherwise outside leaks into rooms.
    sealed = cv2.dilate(wall_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)

    # Areas not occupied by walls.
    inv = (1 - sealed).astype(np.uint8) * 255

    # Flood fill from border to mark outside region.
    flood = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 128)

    outside = (flood == 128).astype(np.uint8)

    # Interior floor is non-wall and non-outside.
    floor = ((inv > 0).astype(np.uint8) & (1 - outside)).astype(np.uint8)
    floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    floor = cv2.erode(floor, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    floor = _remove_small_components(floor, min_area=300)
    return floor, outside


def _detect_window_by_color(img_bgr: np.ndarray) -> np.ndarray:
    """Detect cyan/light-blue window strokes commonly used in floorplan renders."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Cyan/light-blue range.
    lower = np.array([80, 30, 80], dtype=np.uint8)
    upper = np.array([120, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    m = (mask > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    m = _remove_small_components(m, min_area=20)
    return m


def _estimate_openings(
    img_bgr: np.ndarray,
    wall_mask: np.ndarray,
    floor_mask: np.ndarray,
    outside_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    window_color = _detect_window_by_color(img_bgr)

    # Candidate pixels near walls where openings likely exist.
    wall_band = cv2.dilate(wall_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    # Floor pixels very close to walls are potential gaps/openings.
    floor_near_wall = (wall_band & floor_mask).astype(np.uint8)
    floor_near_wall = _remove_small_components(floor_near_wall, min_area=20)

    # Openings close to outside are mostly windows / entrance doors.
    outside_band = cv2.dilate(outside_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    window_by_geom = (floor_near_wall & outside_band).astype(np.uint8)

    # Prefer color window detection, then geometric fallback.
    window_mask = ((window_color | window_by_geom) > 0).astype(np.uint8)

    # Remaining opening candidates treated as doors (interior transitions).
    door_mask = (floor_near_wall & (1 - window_mask)).astype(np.uint8)

    window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    door_mask = cv2.morphologyEx(door_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    window_mask = _remove_small_components(window_mask, min_area=15)
    door_mask = _remove_small_components(door_mask, min_area=15)
    return door_mask, window_mask


def extract_structure_semantic(
    img_bgr: np.ndarray,
    wall_thresh: int,
    wall_min_thickness: int,
) -> np.ndarray:
    wall = _extract_wall_mask(img_bgr, wall_thresh=wall_thresh, wall_min_thickness=wall_min_thickness)
    floor, outside = _extract_floor_and_outside(wall)
    door, window = _estimate_openings(img_bgr, wall, floor, outside)

    sem = np.full(wall.shape, VOID_ID, dtype=np.uint8)
    sem[floor == 1] = FLOOR_ID
    sem[wall == 1] = WALL_ID
    sem[window == 1] = WINDOW_ID
    sem[door == 1] = DOOR_ID
    return sem


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract wall/door/window/floor from GT raster floorplans.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with edited_image_*.png")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--wall_thresh", type=int, default=75, help="Gray threshold for dark wall detection")
    parser.add_argument("--wall_min_thickness", type=int, default=2, help="Min stroke half-thickness in distance-transform pixels")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    png_dir = output_dir / "floor_plans_png"
    npy_dir = output_dir / "floor_plans_npy"
    png_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("edited_image_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No edited_image_*.png found in {input_dir}")

    done = 0
    failed = 0
    for p in tqdm(image_paths, desc="Extracting structure"):
        m = ID_PATTERN.search(p.name)
        if not m:
            continue
        plan_id = m.group(1)

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            failed += 1
            continue

        try:
            sem = extract_structure_semantic(
                img,
                wall_thresh=args.wall_thresh,
                wall_min_thickness=args.wall_min_thickness,
            )

            np.save(str(npy_dir / f"{plan_id}_floorplan.npy"), sem)
            rgb = _to_viz_rgb(sem)
            cv2.imwrite(str(png_dir / f"{plan_id}_floorplan.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            done += 1
        except Exception:
            failed += 1

    print("\n" + "=" * 68)
    print(f"Input images: {len(image_paths)}")
    print(f"Processed: {done}")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
