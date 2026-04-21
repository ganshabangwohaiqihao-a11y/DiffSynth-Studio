#!/usr/bin/env python3
"""
Convert .npy semantic maps into individual PNG images for visualization/training.

Usage:
    python npy_to_images.py \
        --npy_file chinese_train_128x128.npy \
        --output_dir ./train_images \
        --channel layout   # or 'condition', or 'both'
        --split_size 100   # save every 100th image
        --metadata_file chinese_meta_train.json  # for naming
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import os


# Color mapping (same as in parse_json_floorplan.py)
_VIZ_COLORMAP = {
    0:   (0,   0,   0),      # void
    1:   (200, 200, 200),    # floor
    36:  (255, 120,  0),     # door
    37:  (0,  180, 255),     # window
    # Sample furniture colours
    2: (200,  80,  80), 3: (200,  80,  80), 4: (200,  80,  80),      # beds
    10: (80, 140, 200), 13: (100, 200, 100), 23: (200, 130,  50),   # desk, table, sofa
    33: (160,  60, 160), 39: (80, 200, 200), 42: (200, 200,  60),   # wardrobe, toilet, fridge
}

def _cat_to_color(cat_id: int) -> tuple:
    """Convert category ID to RGB color."""
    if cat_id in _VIZ_COLORMAP:
        return _VIZ_COLORMAP[cat_id]
    # Generate random color for unknown IDs
    rng = np.random.RandomState(cat_id * 17 + 3)
    return tuple(int(x) for x in rng.randint(60, 230, 3))


def semantic_map_to_rgb(semantic_map: np.ndarray) -> np.ndarray:
    """Convert semantic category map to RGB image."""
    h, w = semantic_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cat_id in np.unique(semantic_map):
        mask = semantic_map == cat_id
        rgb[mask] = _cat_to_color(int(cat_id))
    
    return rgb


def npy_to_images(
    npy_file: str,
    output_dir: str,
    channel: str = "both",  # 'layout', 'condition', or 'both'
    metadata_file: str = None,
    split_size: int = 1,  # save every Nth image
    prefix: str = "room",
):
    """Convert .npy file to PNG images.
    
    Parameters
    ----------
    npy_file : str
        Path to .npy file (shape: N, 2, H, W)
    output_dir : str
        Output directory to save PNG files
    channel : str
        Which channel to save: 'layout', 'condition', or 'both'
    metadata_file : str, optional
        Path to .json metadata file for naming samples
    split_size : int
        Only save every Nth image (default 1 = save all)
    prefix : str
        Prefix for output filenames
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load .npy data
    print(f"Loading {npy_file}...")
    data = np.load(npy_file)
    print(f"  Shape: {data.shape}")
    n_samples, n_channels, h, w = data.shape
    assert n_channels == 2, "Expected 2 channels (room_type + semantic)"
    
    # Load metadata if provided
    metadata = []
    if metadata_file and os.path.exists(metadata_file):
        print(f"Loading metadata from {metadata_file}...")
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  Loaded {len(metadata)} metadata entries")
    
    # Create subdirectories
    if channel in ("layout", "both"):
        layout_dir = Path(output_dir) / "layout"
        layout_dir.mkdir(exist_ok=True)
    if channel in ("condition", "both"):
        condition_dir = Path(output_dir) / "condition"
        condition_dir.mkdir(exist_ok=True)
    
    # Convert and save images
    saved_count = 0
    print(f"\nConverting {n_samples} samples (saving every {split_size})...")
    
    for i in range(0, n_samples, split_size):
        # Get filename
        if i < len(metadata):
            room_name = metadata[i].get("room_name", "unknown")
            plan_id = metadata[i].get("plan_id", "")
            filename_base = f"{plan_id}_{room_name}_{i:06d}"
        else:
            filename_base = f"{prefix}_{i:06d}"
        
        # Extract channels
        room_type_map = data[i, 0, :, :]      # Channel 0: room type (uniform values)
        semantic_map = data[i, 1, :, :]       # Channel 1: furniture/doors/windows
        
        # Convert to RGB
        semantic_rgb = semantic_map_to_rgb(semantic_map)
        
        # Optional: create composite with room type info
        # (you could overlay room_type_map as a separate layer)
        
        # Save layout (semantic map with furniture)
        if channel in ("layout", "both"):
            layout_path = Path(output_dir) / "layout" / f"{filename_base}_layout.png"
            Image.fromarray(semantic_rgb, mode="RGB").save(layout_path)
        
        # Save condition (we need to know which pixels are "void" in furniture)
        # For now, just save the same as layout
        # (The difference is in how they were generated, not in the pixel values)
        if channel in ("condition", "both"):
            condition_path = Path(output_dir) / "condition" / f"{filename_base}_condition.png"
            Image.fromarray(semantic_rgb, mode="RGB").save(condition_path)
        
        saved_count += 1
        if saved_count % 100 == 0:
            print(f"  Saved {saved_count} images...")
    
    print(f"\n✅ Conversion complete!")
    print(f"Total saved: {saved_count} images")
    print(f"Output directories:")
    if channel in ("layout", "both"):
        print(f"  - {Path(output_dir) / 'layout'}")
    if channel in ("condition", "both"):
        print(f"  - {Path(output_dir) / 'condition'}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .npy semantic maps to PNG images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "npy_file",
        help="Path to .npy file (shape: N, 2, H, W)",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="./npy_images",
        help="Output directory for PNG files",
    )
    parser.add_argument(
        "-c", "--channel",
        choices=["layout", "condition", "both"],
        default="layout",
        help="Which channel(s) to save",
    )
    parser.add_argument(
        "-m", "--metadata_file",
        default=None,
        help="Optional .json metadata file for better naming",
    )
    parser.add_argument(
        "-s", "--split_size",
        type=int,
        default=1,
        help="Only save every Nth image (default 1 = save all)",
    )
    parser.add_argument(
        "-p", "--prefix",
        default="room",
        help="Filename prefix",
    )
    
    args = parser.parse_args()
    
    npy_to_images(
        npy_file=args.npy_file,
        output_dir=args.output_dir,
        channel=args.channel,
        metadata_file=args.metadata_file,
        split_size=args.split_size,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
