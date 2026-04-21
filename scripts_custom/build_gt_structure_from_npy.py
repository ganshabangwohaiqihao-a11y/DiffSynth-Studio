#!/usr/bin/env python3
"""
Build furniture-free GT images from precomputed structural NPY maps.

Why this script
---------------
When GT images have mixed visual styles and no JSON, robustly removing furniture
from raw raster images is unreliable. If matching structural maps already exist
for each plan ID, this script uses those maps directly to generate consistent
GT targets containing only floor, wall, door, window.

Input naming convention
-----------------------
GT image: edited_image_<plan_id>.png
NPY map:  <plan_id>_floorplan.npy

NPY semantic IDs expected (from parse_json_floorplan_fullplan.py):
  0 = void, 1 = floor, 2 = wall, 3 = door, 4 = window

Usage
-----
python build_gt_structure_from_npy.py \
  --gt_dir /share/home/202230550120/extracted_images_col20 \
  --npy_dir /share/home/202230550120/diffusers/scripts_custom/full_floorplans_no_furniture/floor_plans_npy \
  --out_dir /share/home/202230550120/diffusers/scripts_custom/gt_no_furniture
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm


ID_PATTERN = re.compile(r"edited_image_(\d+)\.png$")

# RGB colors for output visualization
COLORMAP: Dict[int, Tuple[int, int, int]] = {
    0: (255, 255, 255),  # void
    1: (200, 200, 200),  # floor
    2: (30, 30, 30),     # wall
    3: (255, 120, 0),    # door
    4: (0, 180, 255),    # window
}


def to_rgb(semantic_map: np.ndarray) -> np.ndarray:
    h, w = semantic_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id, color in COLORMAP.items():
        rgb[semantic_map == cat_id] = color
    return rgb


def extract_plan_id(file_name: str) -> str | None:
    m = ID_PATTERN.search(file_name)
    if not m:
        return None
    return m.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clean GT images (wall/door/window/floor only) from NPY maps."
    )
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with edited_image_*.png")
    parser.add_argument("--npy_dir", type=str, required=True, help="Directory with *_floorplan.npy")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for clean GT PNGs")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob("edited_image_*.png"))
    if not gt_files:
        raise FileNotFoundError(f"No edited_image_*.png found in {gt_dir}")

    matched = 0
    missing_npy = 0
    bad_npy = 0

    for gt_path in tqdm(gt_files, desc="Building clean GT"):
        plan_id = extract_plan_id(gt_path.name)
        if plan_id is None:
            continue

        npy_path = npy_dir / f"{plan_id}_floorplan.npy"
        if not npy_path.exists():
            missing_npy += 1
            continue

        try:
            sem = np.load(str(npy_path))
            if sem.ndim != 2:
                bad_npy += 1
                continue

            # Match GT resolution for 1:1 pairing in later training.
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            if gt_img is None:
                bad_npy += 1
                continue
            gh, gw = gt_img.shape[:2]

            sem_resized = cv2.resize(sem.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST)
            rgb = to_rgb(sem_resized)

            out_path = out_dir / gt_path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            matched += 1
        except Exception:
            bad_npy += 1
            continue

    print("\n" + "=" * 68)
    print(f"GT images found: {len(gt_files)}")
    print(f"Successfully generated: {matched}")
    print(f"Missing NPY: {missing_npy}")
    print(f"Invalid/failed NPY: {bad_npy}")
    print(f"Output dir: {out_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
