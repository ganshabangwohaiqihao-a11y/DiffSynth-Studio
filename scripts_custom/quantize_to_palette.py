#!/usr/bin/env python3
"""
Quantize generated RGB images to the authoritative palette (c2rgb.json).

For each image this script replaces every pixel with the nearest palette
color (squared Euclidean distance) and writes a PNG to the output directory
preserving the original filename.

Usage:
  python3 diffusers/scripts_custom/quantize_to_palette.py \
   --images-dir /share/home/202230550120/diffusers/output_2026.3.27_nolength/predictions_nochidu_v4/images \
   --metadata /share/home/202230550120/diffusers/metadata \
   --out-dir /share/home/202230550120/diffusers/output_2026.3.27_nolength/predictions_nochidu_v4/quantized_slic_c10_n1000_morph5 \
   --pattern 'sample_*_generated.png' \
   --slic-n-segments 1000 --slic-compactness 10 --morph closing --morph-kernel 5
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import warnings

# optional dependencies
try:
    import cv2
except Exception:
    cv2 = None
try:
    from skimage.segmentation import slic
except Exception:
    slic = None


def load_palette(metadata_dir: Path):
    c2rgb = json.load(open(metadata_dir / 'c2rgb.json', 'r', encoding='utf-8'))
    # c2rgb maps class_name -> [R,G,B]
    palette = []
    for class_name, rgb in c2rgb.items():
        palette.append(tuple(int(x) for x in rgb))
    # ensure common VOID color exists
    if (60, 60, 60) not in palette:
        palette.append((60, 60, 60))
    palette_arr = np.array(palette, dtype=np.uint8)
    return palette_arr


def quantize_image(
    img_path: Path,
    palette_arr: np.ndarray,
    out_path: Path,
    chunk: int = 200000,
    morph: str = "none",
    morph_kernel: int = 3,
    slic_n_segments: int = 0,
    slic_compactness: float = 10.0,
):
    img = Image.open(img_path).convert('RGB')
    arr = np.asarray(img, dtype=np.uint8)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3).astype(np.float32)

    palette_f = palette_arr.astype(np.float32)
    N = flat.shape[0]
    idx_all = np.empty((N,), dtype=np.int32)

    for i in range(0, N, chunk):
        block = flat[i:i+chunk]
        d2 = ((block[:, None, :] - palette_f[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1).astype(np.int32)
        idx_all[i:i+chunk] = idx

    label_idx = idx_all.reshape(h, w)

    # Optional SLIC superpixel majority voting
    if slic_n_segments and slic is not None:
        try:
            segments = slic(arr, n_segments=int(slic_n_segments), compactness=float(slic_compactness), start_label=0)
            # for each superpixel, compute majority label and assign
            seg_ids = np.unique(segments)
            for sid in seg_ids:
                mask = segments == sid
                seg_labels = label_idx[mask]
                if seg_labels.size == 0:
                    continue
                vals, counts = np.unique(seg_labels, return_counts=True)
                maj = vals[np.argmax(counts)]
                label_idx[mask] = int(maj)
        except Exception as e:
            warnings.warn(f"SLIC majority voting failed for {img_path.name}: {e}")
    elif slic_n_segments and slic is None:
        warnings.warn("scikit-image not available: skipping SLIC superpixel majority voting")

    # Optional morphology per-label (opening/closing)
    if morph != "none":
        if cv2 is None:
            warnings.warn("cv2 not available: skipping morphology")
        else:
            op = None
            if morph == "opening":
                op = cv2.MORPH_OPEN
            elif morph == "closing":
                op = cv2.MORPH_CLOSE
            if op is not None and morph_kernel > 0:
                k = int(morph_kernel)
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                # apply morphology per label
                unique_labels = np.unique(label_idx)
                out_idx = np.zeros_like(label_idx, dtype=np.int32)
                for lab in unique_labels:
                    mask = (label_idx == lab).astype(np.uint8)
                    try:
                        cleaned = cv2.morphologyEx(mask, op, kernel)
                    except Exception:
                        cleaned = mask
                    out_idx[cleaned.astype(bool)] = int(lab)
                label_idx = out_idx

    out_arr = palette_arr[label_idx]
    out_img = Image.fromarray(out_arr.astype(np.uint8), mode='RGB')
    out_img.save(out_path, format='PNG')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--pattern', type=str, default='sample_*_generated.*')
    parser.add_argument('--morph', type=str, default='none', choices=['none', 'opening', 'closing'], help='Per-label morphology to apply after quantize')
    parser.add_argument('--morph-kernel', type=int, default=3, help='Kernel size for morphology (odd integer)')
    parser.add_argument('--slic-n-segments', type=int, default=0, help='If >0, run SLIC superpixel majority-vote with this many segments')
    parser.add_argument('--slic-compactness', type=float, default=10.0, help='Compactness for SLIC')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    palette = load_palette(args.metadata)

    imgs = sorted(args.images_dir.glob(args.pattern))
    if not imgs:
        imgs = sorted(args.images_dir.glob('*'))

    print(f'Quantizing {len(imgs)} images from {args.images_dir} -> {args.out_dir}')

    for p in imgs:
        try:
            out_p = args.out_dir / p.name
            quantize_image(
                p,
                palette,
                out_p,
                morph=args.morph,
                morph_kernel=int(args.morph_kernel),
                slic_n_segments=int(args.slic_n_segments),
                slic_compactness=float(args.slic_compactness),
            )
            print('Wrote', out_p)
        except Exception as e:
            print(f'Error quantizing {p}: {e}')


if __name__ == '__main__':
    main()
