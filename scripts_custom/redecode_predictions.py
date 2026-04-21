#!/usr/bin/env python3
"""
Re-decode generated RGB prediction images into label-map NPY files

Usage:
python3 scripts_custom/redecode_predictions.py \
  --images-dir /share/home/202230550120/diffusers/output_2026.3.27_nolength/predictions_nochidu_v4/images \
  --out-npy-dir /share/home/202230550120/diffusers/output_2026.3.27_nolength/predictions_nochidu_v4/npy_new \
  --metadata /share/home/202230550120/diffusers/metadata \
  --min-area 0 \
  --nearest-threshold 1000 \
  --pattern 'sample_*_01_generated.*' \
  --out-template 'sample_{idx}_pred.npy'

This script maps image colors -> label ids using `c2rgb.json` (exact match first,
then nearest neighbor within squared-distance threshold). Optionally removes small
connected components smaller than `--min-area` pixels (sets them to VOID=0).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None


VOID_ID = 0


def load_palette(metadata_dir: Path) -> Tuple[Dict[Tuple[int,int,int], int], Dict[int, Tuple[int,int,int]]]:
    id2c = json.load(open(metadata_dir / 'id2c.json', 'r', encoding='utf-8'))
    c2rgb = json.load(open(metadata_dir / 'c2rgb.json', 'r', encoding='utf-8'))

    eval_map = {}
    rev_map = {}
    for id_str, class_name in id2c.items():
        label_id = int(id_str)
        rgb = tuple(int(x) for x in c2rgb.get(class_name, [0,0,0]))
        eval_map[rgb] = label_id
        rev_map[label_id] = rgb

    # ensure text color maps to VOID
    eval_map[(60,60,60)] = VOID_ID
    rev_map.setdefault(VOID_ID, (255,255,255))
    return eval_map, rev_map


def map_image_to_labels(img_arr: np.ndarray, color_map: Dict[Tuple[int,int,int], int], nearest_threshold: int) -> np.ndarray:
    h, w, _ = img_arr.shape
    flat = img_arr.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    # build palette arrays
    palette_colors = np.array(list(color_map.keys()), dtype=np.float32)
    palette_labels = np.array([color_map[c] for c in color_map.keys()], dtype=np.uint8)

    mapped = np.zeros((unique_colors.shape[0],), dtype=np.uint8)

    # exact match dict for speed
    exact = {c: color_map[c] for c in color_map.keys()}

    for i, uc in enumerate(unique_colors):
        t = (int(uc[0]), int(uc[1]), int(uc[2]))
        if t in exact:
            mapped[i] = exact[t]
            continue
        # nearest neighbor
        if palette_colors.size == 0:
            mapped[i] = VOID_ID
            continue
        dif = palette_colors - uc.astype(np.float32)
        d2 = np.einsum('ij,ij->i', dif, dif)
        idx = int(np.argmin(d2))
        if d2[idx] <= nearest_threshold:
            mapped[i] = palette_labels[idx]
        else:
            mapped[i] = VOID_ID

    labels_flat = mapped[inverse].reshape(h, w).astype(np.uint8)
    return labels_flat


def remove_small_components(label_map: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0 or cv2 is None:
        return label_map

    out = label_map.copy()
    h, w = out.shape
    for lab in np.unique(label_map):
        if lab == VOID_ID:
            continue
        mask = (label_map == lab).astype(np.uint8)
        if mask.max() == 0:
            continue
        num_labels, comp = cv2.connectedComponents(mask, connectivity=8)
        for cid in range(1, num_labels):
            area = int((comp == cid).sum())
            if area < min_area:
                out[comp == cid] = VOID_ID
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--out-npy-dir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--min-area', type=int, default=8, help='min connected component area to keep')
    parser.add_argument('--nearest-threshold', type=int, default=1000, help='squared color distance threshold for nearest-match')
    parser.add_argument('--pattern', type=str, default='sample_*_01_generated.*')
    parser.add_argument('--out-template', type=str, default='sample_{idx}_pred_redecoded.npy', help='output filename template, use {idx} for sample index')
    args = parser.parse_args()

    args.out_npy_dir.mkdir(parents=True, exist_ok=True)

    color_map, rev_map = load_palette(args.metadata)

    imgs = sorted(args.images_dir.glob(args.pattern))
    if not imgs:
        # fallback broader pattern
        imgs = sorted(args.images_dir.glob('sample_*_generated.*'))

    print(f'Found {len(imgs)} generated images in {args.images_dir}')

    for p in imgs:
        try:
            name = p.name
            # extract index
            import re
            m = re.search(r'sample_(\d+)', name)
            if m:
                idx = int(m.group(1))
            else:
                # fallback to numeric part
                idx = int(''.join([c for c in name if c.isdigit()]) or 0)

            img = Image.open(p).convert('RGB')
            arr = np.asarray(img, dtype=np.uint8)

            labels = map_image_to_labels(arr, color_map, args.nearest_threshold)

            if args.min_area > 0 and cv2 is not None:
                labels = remove_small_components(labels, args.min_area)

            out_path = args.out_npy_dir / args.out_template.format(idx=idx)
            np.save(out_path, labels)

            # quick stats
            uniq = np.unique(labels)
            print(f'{p.name} -> {out_path.name}: unique_labels={len(uniq)} top3={uniq[:3].tolist()}')
        except Exception as e:
            print(f'Error processing {p}: {e}')


if __name__ == '__main__':
    main()
