#!/usr/bin/env python3
"""
Evaluate SLDN semantic layouts without APM/retrieval.

Inputs:
- GT semantic layout npy from parse_json_floorplan.py
- Predicted semantic layout npy/png/jpg/jpeg from SLDN inference

Outputs:
- Rendered semantic maps for GT and predictions
- FID/KID (image-level)
- SCA (image-level classifier)
- Semantic CKL (instance-count distribution on semantic maps)
"""

# python DiffSynth-Studio/评估/scripts/evaluate_sldn_layouts.py \
#   --gt-data-dir /share/home/202230550120/diffusers \
#   --gt-template output_2026.3.27/chinese_{split}_1024x1024.npy \
#   --split test \
#   --pred-dir /share/home/202230550120/diffusers/output_2026.3.27/predictions_chidu_v4/npy_slic_c10_n1000_morph5 \
#   --pred-glob "*_pred_slic_c10.npy" \
#   --pred-index-regex "sample_(\\d+)" \
#   --output-dir DiffSynth-Studio/评估/evaluation_results/chidu_v7 \
#   --channel-index 1 \
#   --id2c-path /share/home/202230550120/diffusers/metadata/id2c.json \
#   --c2rgb-path /share/home/202230550120/diffusers/metadata/c2rgb.json \
#   --metrics iou count_2d oar_2d nav_2d oob_2d ckl

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG = {
    "output_dir": "evaluation_results/sldn_layout_eval",
    "gt_data_dir": "datasets/_tmp_parse_floorplan_map_full",
    "gt_template": "chinese_{split}_128x128.npy",
    "split": "test",
    "pred_dir": "",
    "pred_glob": "*",
    "pred_index_regex": r"sample_(\d+)",
    "name_filter": "",
    "max_samples": 0,
    "channel_index": 1,
    "id2c_path": "my_tools/id2c.json",
    "c2rgb_path": "my_tools/c2rgb.json",
    "ignore_label_names": ["void", "floor", "door", "window"],
    "metrics": ["iou", "oob_2d", "count_2d", "oar_2d", "nav_2d", "fid_kid", "sca", "ckl"],
    "min_instance_pixels": 4,
    "count_2d": {
        "connectivity": 8,
    },
    "oar_2d": {
        "connectivity": 8,
        "wall_band_pixels": 3,
        "near_opening_pixels": 6,
        "middle_threshold_ratio": 0.18,
        "supported_relations": ["against_wall", "near_door", "near_window", "middle_of_room"],
    },
    "nav_2d": {
        "connectivity": 8,
    },
    "fid": {
        "device": None,
        "num_iterations": 10,
        "num_workers": 0,
        "use_dataparallel": False,
        "temp_dir_base": None,
    },
    "sca": {
        "batch_size": 256,
        "num_workers": 0,
        "epochs": 10,
        "device": None,
    },
}


@dataclass
class PredictionSample:
    sample_index: int
    source_path: Path
    source_kind: str  # npy/png/jpg/jpeg
    stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SLDN semantic layouts.")
    parser.add_argument("--config", type=str, default="", help="Optional YAML config path.")
    parser.add_argument("--gt-data-dir", type=str, default=None, help="Directory containing GT npy.")
    parser.add_argument("--gt-template", type=str, default=None, help="Template like chinese_{split}_128x128.npy")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--pred-dir", type=str, default=None, help="Directory containing predicted npy/png files.")
    parser.add_argument("--pred-glob", type=str, default=None, help="Glob pattern inside pred dir.")
    parser.add_argument("--pred-index-regex", type=str, default=None, help="Regex with one capture group for sample idx.")
    parser.add_argument("--name-filter", type=str, default=None, help="Only include predicted files whose name contains this substring.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit after matching.")
    parser.add_argument("--channel-index", type=int, default=None, help="Channel index of semantic map in GT npy.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save rendered maps and metrics.")
    parser.add_argument("--id2c-path", type=str, default=None, help="Path to id2c.json.")
    parser.add_argument("--c2rgb-path", type=str, default=None, help="Path to c2rgb.json.")
    parser.add_argument("--metrics", nargs="+", default=None, choices=["iou", "oob_2d", "count_2d", "oar_2d", "nav_2d", "fid_kid", "sca", "ckl"], help="Metrics to run.")
    parser.add_argument("--min-instance-pixels", type=int, default=None, help="Min connected-component size for semantic CKL.")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(args: argparse.Namespace) -> Dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config:
        yaml_cfg = load_yaml(Path(args.config))
        config = deep_update(config, yaml_cfg)

    cli_overrides = {
        "gt_data_dir": args.gt_data_dir,
        "gt_template": args.gt_template,
        "split": args.split,
        "pred_dir": args.pred_dir,
        "pred_glob": args.pred_glob,
        "pred_index_regex": args.pred_index_regex,
        "name_filter": args.name_filter,
        "max_samples": args.max_samples,
        "channel_index": args.channel_index,
        "output_dir": args.output_dir,
        "id2c_path": args.id2c_path,
        "c2rgb_path": args.c2rgb_path,
        "metrics": args.metrics,
        "min_instance_pixels": args.min_instance_pixels,
    }
    for key, value in cli_overrides.items():
        if value is not None and value != "":
            config[key] = value

    if not config.get("pred_dir"):
        raise ValueError("pred_dir is required.")
    return config


def deep_update(base: Dict, new: Dict) -> Dict:
    result = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_module_from_project(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_map_array(sample: np.ndarray, channel_index: int) -> np.ndarray:
    arr = np.asarray(sample)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3:
        if channel_index >= arr.shape[0]:
            raise ValueError(f"channel_index={channel_index} out of range for sample shape {arr.shape}")
        return arr[channel_index].astype(np.uint8)
    raise ValueError(f"Unsupported GT sample shape: {arr.shape}")


def deterministic_color(label_id: int) -> Tuple[int, int, int]:
    return (
        int((37 * label_id + 53) % 256),
        int((97 * label_id + 29) % 256),
        int((17 * label_id + 191) % 256),
    )


def build_palette(id2c: Dict[str, str], c2rgb: Dict[str, List[int]]) -> Dict[int, Tuple[int, int, int]]:
    palette = {}
    for id_str, class_name in id2c.items():
        label_id = int(id_str)
        if class_name in c2rgb:
            palette[label_id] = tuple(int(v) for v in c2rgb[class_name])
        else:
            palette[label_id] = deterministic_color(label_id)
    return palette


def render_label_map(label_map: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    height, width = label_map.shape
    rendered = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(label_map)
    for label_id in unique_labels:
        rendered[label_map == label_id] = palette.get(int(label_id), deterministic_color(int(label_id)))
    return rendered


def save_rendered_png(label_map: np.ndarray, palette: Dict[int, Tuple[int, int, int]], path: Path) -> None:
    rgb = render_label_map(label_map, palette)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def decode_image_to_label_map(
    image_path: Path,
    id2c: Dict[str, str],
    c2rgb: Dict[str, List[int]],
    use_nearest_color: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    reverse = {}
    ambiguous_colors = []
    for id_str, class_name in id2c.items():
        if class_name not in c2rgb:
            continue
        color = tuple(int(v) for v in c2rgb[class_name])
        if color in reverse and reverse[color] != int(id_str):
            ambiguous_colors.append(f"{color}: {reverse[color]} vs {id_str}")
        reverse.setdefault(color, int(id_str))

    label_map = np.zeros(rgb.shape[:2], dtype=np.uint8)
    flat = rgb.reshape(-1, 3)
    unique_colors = np.unique(flat, axis=0)
    unknown_colors = 0

    if use_nearest_color:
        valid_colors = np.array(list(reverse.keys()), dtype=np.float32)
        valid_label_ids = np.array([reverse[tuple(c)] for c in valid_colors.astype(np.uint8)], dtype=np.uint8)
        for color_arr in unique_colors:
            color = tuple(int(v) for v in color_arr)
            if color in reverse:
                label_id = reverse[color]
            else:
                unknown_colors += 1
                deltas = valid_colors - color_arr.astype(np.float32)
                distances = np.einsum("ij,ij->i", deltas, deltas)
                nearest_idx = int(np.argmin(distances))
                label_id = int(valid_label_ids[nearest_idx])
            label_map[(rgb == color_arr).all(axis=-1)] = label_id
    else:
        for color_arr in unique_colors:
            color = tuple(int(v) for v in color_arr)
            label_id = reverse.get(color, 0)
            if color not in reverse:
                unknown_colors += 1
            label_map[(rgb == color_arr).all(axis=-1)] = label_id

    if unknown_colors > 0:
        ambiguous_colors.append(
            f"{image_path.name}: {unknown_colors} unknown colors encountered; "
            + ("mapped to nearest class colors" if use_nearest_color else "mapped to label 0")
        )

    return label_map, ambiguous_colors


def extract_index(filename: str, pattern: re.Pattern[str]) -> Optional[int]:
    match = pattern.search(filename)
    if not match:
        return None
    return int(match.group(1))


def discover_prediction_samples(config: Dict) -> List[PredictionSample]:
    pred_dir = Path(config["pred_dir"])
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    name_filter = config.get("name_filter", "")
    pattern = re.compile(config["pred_index_regex"])
    pred_glob = config.get("pred_glob", "*")
    candidates = sorted(pred_dir.glob(pred_glob))

    samples = []
    for path in candidates:
        if path.suffix.lower() not in {".npy", ".png", ".jpg", ".jpeg"}:
            continue
        if name_filter and name_filter not in path.name:
            continue
        sample_index = extract_index(path.name, pattern)
        if sample_index is None:
            continue
        source_kind = path.suffix.lower().lstrip(".")
        samples.append(
            PredictionSample(
                sample_index=sample_index,
                source_path=path,
                source_kind=source_kind,
                stem=path.stem,
            )
        )

    grouped: Dict[int, List[PredictionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.sample_index, []).append(sample)

    resolved_samples: List[PredictionSample] = []
    duplicate_errors: List[str] = []
    for sample_index, group in sorted(grouped.items()):
        if len(group) == 1:
            resolved_samples.append(group[0])
            continue

        npy_group = [sample for sample in group if sample.source_kind == "npy"]
        png_group = [sample for sample in group if sample.source_kind == "png"]
        if len(npy_group) == 1 and len(png_group) >= 1:
            resolved_samples.append(npy_group[0])
            continue

        duplicate_errors.append(
            f"sample_index={sample_index} matched multiple prediction files: "
            + ", ".join(sample.source_path.name for sample in group)
        )

    if duplicate_errors:
        raise ValueError(
            "Prediction files are ambiguous after matching. "
            "Use --name-filter and/or --pred-glob to select one prediction per sample.\n"
            + "\n".join(duplicate_errors)
        )

    samples = resolved_samples
    samples.sort(key=lambda s: (s.sample_index, s.source_path.name))
    max_samples = int(config.get("max_samples") or 0)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No prediction files matched in {pred_dir}")
    return samples


def load_prediction_map(sample: PredictionSample, id2c: Dict[str, str], c2rgb: Dict[str, List[int]]) -> Tuple[np.ndarray, List[str]]:
    if sample.source_kind == "npy":
        arr = np.load(sample.source_path)
        if arr.ndim != 2:
            raise ValueError(f"Prediction npy must be 2D label map, got {arr.shape} at {sample.source_path}")
        return arr.astype(np.uint8), []
    if sample.source_kind == "png":
        return decode_image_to_label_map(sample.source_path, id2c, c2rgb, use_nearest_color=False)
    if sample.source_kind in {"jpg", "jpeg"}:
        label_map, warnings = decode_image_to_label_map(sample.source_path, id2c, c2rgb, use_nearest_color=True)
        warnings.append(
            f"{sample.source_path.name}: JPEG input detected. Results may be biased due to lossy compression. Prefer PNG/NPY for final reporting."
        )
        return label_map, warnings
    raise ValueError(f"Unsupported prediction kind: {sample.source_kind}")


def count_instances_by_class(
    label_map: np.ndarray,
    target_label_ids: Sequence[int],
    min_instance_pixels: int,
) -> Dict[int, int]:
    counts = {label_id: 0 for label_id in target_label_ids}
    for label_id in target_label_ids:
        mask = (label_map == label_id).astype(np.uint8)
        if mask.max() == 0:
            continue
        num_components, component_map = cv2.connectedComponents(mask, connectivity=8)
        for component_id in range(1, num_components):
            area = int((component_map == component_id).sum())
            if area >= min_instance_pixels:
                counts[label_id] += 1
    return counts


def precision_recall_f1(overlap: int, pred_total: int, gt_total: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if pred_total == 0 and gt_total == 0:
        return 1.0, 1.0, 1.0
    precision = float(overlap / pred_total) if pred_total > 0 else 0.0
    recall = float(overlap / gt_total) if gt_total > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, float(2 * precision * recall / (precision + recall))


def extract_instances(
    label_map: np.ndarray,
    target_label_ids: Sequence[int],
    min_instance_pixels: int,
    connectivity: int = 8,
) -> List[Dict]:
    instances: List[Dict] = []
    for label_id in target_label_ids:
        mask = (label_map == label_id).astype(np.uint8)
        if mask.max() == 0:
            continue
        num_components, component_map, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
        for component_id in range(1, num_components):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_instance_pixels:
                continue
            left = int(stats[component_id, cv2.CC_STAT_LEFT])
            top = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            instances.append(
                {
                    "label_id": int(label_id),
                    "component_id": int(component_id),
                    "area": area,
                    "centroid_xy": (float(centroids[component_id][0]), float(centroids[component_id][1])),
                    "bbox_xyxy": (left, top, left + width - 1, top + height - 1),
                    "mask": component_map == component_id,
                }
            )
    return instances


def make_disk_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def largest_component_ratio(mask: np.ndarray, connectivity: int = 8) -> Optional[float]:
    mask_uint8 = mask.astype(np.uint8)
    total = int(mask_uint8.sum())
    if total == 0:
        return None
    num_components, component_map = cv2.connectedComponents(mask_uint8, connectivity=connectivity)
    if num_components <= 1:
        return 1.0
    counts = np.bincount(component_map.reshape(-1))
    largest = int(counts[1:].max()) if counts.shape[0] > 1 else 0
    return float(largest / total)


def build_room_wall_proxy(room_mask: np.ndarray, wall_band_pixels: int) -> np.ndarray:
    room_uint8 = room_mask.astype(np.uint8)
    if wall_band_pixels <= 0:
        return room_uint8.astype(bool)
    kernel = np.ones((2 * wall_band_pixels + 1, 2 * wall_band_pixels + 1), dtype=np.uint8)
    eroded = cv2.erode(room_uint8, kernel, iterations=1)
    wall_proxy = np.logical_and(room_mask, np.logical_not(eroded.astype(bool)))
    return wall_proxy


def get_room_centroid_and_diag(room_mask: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
    ys, xs = np.where(room_mask)
    if len(xs) == 0:
        return None, 0.0
    centroid = (float(xs.mean()), float(ys.mean()))
    diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    return centroid, diag


def build_relation_counts_for_sample(
    label_map: np.ndarray,
    furniture_label_ids: Sequence[int],
    id2c: Dict[str, str],
    min_instance_pixels: int,
    room_mask: np.ndarray,
    door_mask: Optional[np.ndarray],
    window_mask: Optional[np.ndarray],
    config: Dict,
) -> Tuple[Dict[str, int], List[Dict]]:
    connectivity = int(config.get("connectivity", 8))
    wall_band_pixels = int(config.get("wall_band_pixels", 3))
    near_opening_pixels = int(config.get("near_opening_pixels", 6))
    middle_threshold_ratio = float(config.get("middle_threshold_ratio", 0.18))
    supported_relations = list(config.get("supported_relations", ["against_wall", "near_door", "near_window", "middle_of_room"]))

    instances = extract_instances(label_map, furniture_label_ids, min_instance_pixels, connectivity=connectivity)
    wall_proxy = build_room_wall_proxy(room_mask, wall_band_pixels)
    room_centroid, room_diag = get_room_centroid_and_diag(room_mask)
    dilation_kernel = make_disk_kernel(near_opening_pixels)

    counts: Dict[str, int] = {}
    instance_summaries: List[Dict] = []
    for instance in instances:
        class_name = id2c[str(instance["label_id"])]
        inst_mask = instance["mask"].astype(np.uint8)
        dilated = cv2.dilate(inst_mask, dilation_kernel, iterations=1).astype(bool)
        centroid_xy = instance["centroid_xy"]

        relation_flags = {
            "against_wall": bool(np.logical_and(instance["mask"], wall_proxy).any()),
            "near_door": bool(door_mask is not None and np.logical_and(dilated, door_mask).any()),
            "near_window": bool(window_mask is not None and np.logical_and(dilated, window_mask).any()),
            "middle_of_room": False,
        }
        if room_centroid is not None and room_diag > 0:
            dist = float(np.hypot(centroid_xy[0] - room_centroid[0], centroid_xy[1] - room_centroid[1]))
            relation_flags["middle_of_room"] = bool(dist / room_diag <= middle_threshold_ratio)

        active_relations = []
        for relation_name in supported_relations:
            if relation_flags.get(relation_name, False):
                token = f"{class_name}::{relation_name}"
                counts[token] = counts.get(token, 0) + 1
                active_relations.append(relation_name)

        instance_summaries.append(
            {
                "label_id": int(instance["label_id"]),
                "class_name": class_name,
                "area": int(instance["area"]),
                "centroid_xy": [float(centroid_xy[0]), float(centroid_xy[1])],
                "active_relations": active_relations,
            }
        )

    return counts, instance_summaries


def compute_count_2d_metrics(
    gt_maps: Dict[int, np.ndarray],
    pred_maps: Dict[int, np.ndarray],
    furniture_label_ids: Sequence[int],
    id2c: Dict[str, str],
    min_instance_pixels: int,
    connectivity: int = 8,
) -> Dict:
    aggregate_gt_counts = {label_id: 0 for label_id in furniture_label_ids}
    aggregate_pred_counts = {label_id: 0 for label_id in furniture_label_ids}
    exact_match_hits = 0
    exact_match_total = 0
    l1_values: List[float] = []
    total_overlap = 0
    total_gt_instances = 0
    total_pred_instances = 0
    per_sample = []

    for sample_index, gt_map in gt_maps.items():
        pred_map = pred_maps[sample_index]
        gt_counts = count_instances_by_class(gt_map, furniture_label_ids, min_instance_pixels)
        pred_counts = count_instances_by_class(pred_map, furniture_label_ids, min_instance_pixels)

        gt_total = int(sum(gt_counts.values()))
        pred_total = int(sum(pred_counts.values()))
        overlap = int(sum(min(gt_counts[label_id], pred_counts[label_id]) for label_id in furniture_label_ids))
        precision, recall, f1 = precision_recall_f1(overlap, pred_total, gt_total)

        present_labels = [label_id for label_id in furniture_label_ids if gt_counts[label_id] > 0 or pred_counts[label_id] > 0]
        sample_exact = 0
        sample_l1_values: List[int] = []
        for label_id in present_labels:
            aggregate_gt_counts[label_id] += gt_counts[label_id]
            aggregate_pred_counts[label_id] += pred_counts[label_id]
            sample_l1 = abs(gt_counts[label_id] - pred_counts[label_id])
            sample_l1_values.append(sample_l1)
            if gt_counts[label_id] == pred_counts[label_id]:
                sample_exact += 1
        exact_match_hits += sample_exact
        exact_match_total += len(present_labels)
        l1_values.extend(float(v) for v in sample_l1_values)
        total_overlap += overlap
        total_gt_instances += gt_total
        total_pred_instances += pred_total

        per_sample.append(
            {
                "sample_index": sample_index,
                "gt_total_instances": gt_total,
                "pred_total_instances": pred_total,
                "instance_overlap": overlap,
                "instance_precision": precision,
                "instance_recall": recall,
                "instance_f1": f1,
                "class_exact_match_ratio": float(sample_exact / len(present_labels)) if present_labels else 1.0,
                "mean_count_l1_error_per_present_class": float(np.mean(sample_l1_values)) if sample_l1_values else 0.0,
                "gt_counts": {id2c[str(label_id)]: int(count) for label_id, count in gt_counts.items() if count > 0},
                "pred_counts": {id2c[str(label_id)]: int(count) for label_id, count in pred_counts.items() if count > 0},
            }
        )

    precision, recall, f1 = precision_recall_f1(total_overlap, total_pred_instances, total_gt_instances)
    per_class_aggregate = {}
    for label_id in furniture_label_ids:
        gt_count = int(aggregate_gt_counts[label_id])
        pred_count = int(aggregate_pred_counts[label_id])
        if gt_count == 0 and pred_count == 0:
            continue
        overlap = int(min(gt_count, pred_count))
        per_class_aggregate[str(label_id)] = {
            "class_name": id2c[str(label_id)],
            "gt_count": gt_count,
            "pred_count": pred_count,
            "overlap_count": overlap,
            "l1_error": int(abs(gt_count - pred_count)),
        }

    return {
        "mode": "gt_proxy",
        "num_matched_samples": len(gt_maps),
        "min_instance_pixels": int(min_instance_pixels),
        "connectivity": int(connectivity),
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": f1,
        "class_exact_match_ratio": float(exact_match_hits / exact_match_total) if exact_match_total > 0 else 1.0,
        "mean_count_l1_error_per_present_class": float(np.mean(l1_values)) if l1_values else 0.0,
        "gt_total_instances": int(total_gt_instances),
        "pred_total_instances": int(total_pred_instances),
        "instance_overlap": int(total_overlap),
        "per_class_aggregate": per_class_aggregate,
        "per_sample": per_sample,
    }


def compute_oar_2d_metrics(
    gt_maps: Dict[int, np.ndarray],
    pred_maps: Dict[int, np.ndarray],
    furniture_label_ids: Sequence[int],
    id2c: Dict[str, str],
    min_instance_pixels: int,
    void_label_id: int,
    door_label_id: Optional[int],
    window_label_id: Optional[int],
    config: Dict,
) -> Dict:
    aggregate_gt_relations: Dict[str, int] = {}
    aggregate_pred_relations: Dict[str, int] = {}
    exact_match_hits = 0
    exact_match_total = 0
    total_overlap = 0
    total_gt = 0
    total_pred = 0
    per_sample = []

    for sample_index, gt_map in gt_maps.items():
        pred_map = pred_maps[sample_index]
        room_mask = gt_map != void_label_id
        gt_door = gt_map == door_label_id if door_label_id is not None else None
        gt_window = gt_map == window_label_id if window_label_id is not None else None

        gt_counts, gt_instances = build_relation_counts_for_sample(
            label_map=gt_map,
            furniture_label_ids=furniture_label_ids,
            id2c=id2c,
            min_instance_pixels=min_instance_pixels,
            room_mask=room_mask,
            door_mask=gt_door,
            window_mask=gt_window,
            config=config,
        )
        pred_counts, pred_instances = build_relation_counts_for_sample(
            label_map=pred_map,
            furniture_label_ids=furniture_label_ids,
            id2c=id2c,
            min_instance_pixels=min_instance_pixels,
            room_mask=room_mask,
            door_mask=gt_door,
            window_mask=gt_window,
            config=config,
        )

        for key, value in gt_counts.items():
            aggregate_gt_relations[key] = aggregate_gt_relations.get(key, 0) + int(value)
        for key, value in pred_counts.items():
            aggregate_pred_relations[key] = aggregate_pred_relations.get(key, 0) + int(value)

        relation_keys = sorted(set(gt_counts) | set(pred_counts))
        overlap = int(sum(min(gt_counts.get(key, 0), pred_counts.get(key, 0)) for key in relation_keys))
        gt_total = int(sum(gt_counts.values()))
        pred_total = int(sum(pred_counts.values()))
        precision, recall, f1 = precision_recall_f1(overlap, pred_total, gt_total)

        sample_exact = 0
        for key in relation_keys:
            if gt_counts.get(key, 0) == pred_counts.get(key, 0):
                sample_exact += 1
        exact_match_hits += sample_exact
        exact_match_total += len(relation_keys)
        total_overlap += overlap
        total_gt += gt_total
        total_pred += pred_total

        per_sample.append(
            {
                "sample_index": sample_index,
                "relation_precision": precision,
                "relation_recall": recall,
                "relation_f1": f1,
                "relation_exact_match_ratio": float(sample_exact / len(relation_keys)) if relation_keys else 1.0,
                "gt_relation_counts": gt_counts,
                "pred_relation_counts": pred_counts,
                "gt_instances": gt_instances,
                "pred_instances": pred_instances,
            }
        )

    precision, recall, f1 = precision_recall_f1(total_overlap, total_pred, total_gt)
    per_relation_aggregate = {}
    for key in sorted(set(aggregate_gt_relations) | set(aggregate_pred_relations)):
        gt_count = int(aggregate_gt_relations.get(key, 0))
        pred_count = int(aggregate_pred_relations.get(key, 0))
        overlap = int(min(gt_count, pred_count))
        per_relation_aggregate[key] = {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "overlap_count": overlap,
        }

    return {
        "mode": "gt_proxy",
        "num_matched_samples": len(gt_maps),
        "min_instance_pixels": int(min_instance_pixels),
        "supported_relations": list(config.get("supported_relations", [])),
        "relation_precision": precision,
        "relation_recall": recall,
        "relation_f1": f1,
        "relation_exact_match_ratio": float(exact_match_hits / exact_match_total) if exact_match_total > 0 else 1.0,
        "gt_total_relations": int(total_gt),
        "pred_total_relations": int(total_pred),
        "relation_overlap": int(total_overlap),
        "per_relation_aggregate": per_relation_aggregate,
        "per_sample": per_sample,
    }


def compute_nav_2d_metrics(
    gt_maps: Dict[int, np.ndarray],
    pred_maps: Dict[int, np.ndarray],
    furniture_label_ids: Sequence[int],
    void_label_id: int,
    door_label_id: Optional[int],
    window_label_id: Optional[int],
    connectivity: int = 8,
) -> Dict:
    furniture_set = set(furniture_label_ids)
    pred_navs: List[float] = []
    gt_navs: List[float] = []
    abs_errors: List[float] = []
    per_sample = []

    for sample_index, gt_map in gt_maps.items():
        pred_map = pred_maps[sample_index]
        room_walkable = gt_map != void_label_id
        if door_label_id is not None:
            room_walkable = np.logical_and(room_walkable, gt_map != door_label_id)
        if window_label_id is not None:
            room_walkable = np.logical_and(room_walkable, gt_map != window_label_id)

        gt_furniture = np.isin(gt_map, list(furniture_set))
        pred_furniture = np.isin(pred_map, list(furniture_set))
        gt_free = np.logical_and(room_walkable, np.logical_not(gt_furniture))
        pred_free = np.logical_and(room_walkable, np.logical_not(pred_furniture))

        gt_nav = largest_component_ratio(gt_free, connectivity=connectivity)
        pred_nav = largest_component_ratio(pred_free, connectivity=connectivity)
        gt_free_ratio = float(gt_free.sum() / room_walkable.sum()) if room_walkable.sum() > 0 else None
        pred_free_ratio = float(pred_free.sum() / room_walkable.sum()) if room_walkable.sum() > 0 else None
        abs_error = abs(pred_nav - gt_nav) if gt_nav is not None and pred_nav is not None else None

        if gt_nav is not None:
            gt_navs.append(gt_nav)
        if pred_nav is not None:
            pred_navs.append(pred_nav)
        if abs_error is not None:
            abs_errors.append(float(abs_error))

        per_sample.append(
            {
                "sample_index": sample_index,
                "gt_navigability_ratio": gt_nav,
                "pred_navigability_ratio": pred_nav,
                "navigability_abs_error": abs_error,
                "gt_free_space_ratio": gt_free_ratio,
                "pred_free_space_ratio": pred_free_ratio,
            }
        )

    return {
        "mode": "gt_proxy",
        "num_matched_samples": len(gt_maps),
        "connectivity": int(connectivity),
        "mean_gt_navigability_ratio": float(np.mean(gt_navs)) if gt_navs else None,
        "mean_pred_navigability_ratio": float(np.mean(pred_navs)) if pred_navs else None,
        "mean_navigability_abs_error": float(np.mean(abs_errors)) if abs_errors else None,
        "per_sample": per_sample,
    }


def categorical_kl(p: np.ndarray, q: np.ndarray) -> float:
    return float((p * (np.log((p + 1e-6) / (q + 1e-6)))).sum())


def save_frequency_plot(
    class_names: Sequence[str],
    gt_frequencies: np.ndarray,
    pred_frequencies: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.25)))
    bar_width = 0.35
    y1 = np.arange(len(class_names))
    y2 = y1 + bar_width
    ax.barh(y1, pred_frequencies, height=bar_width, label="Pred", color="lightblue")
    ax.barh(y2, gt_frequencies, height=bar_width, label="GT", color="orange")
    ax.set_title("Semantic-layout category frequency comparison")
    ax.set_xlabel("Frequency")
    ax.set_yticks(y1 + bar_width / 2)
    ax.set_yticklabels(class_names)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_semantic_ckl(
    gt_maps: Dict[int, np.ndarray],
    pred_maps: Dict[int, np.ndarray],
    target_label_ids: Sequence[int],
    class_names: Sequence[str],
    min_instance_pixels: int,
    output_dir: Path,
) -> Dict:
    gt_instance_totals = {label_id: 0 for label_id in target_label_ids}
    pred_instance_totals = {label_id: 0 for label_id in target_label_ids}

    per_sample = []
    for sample_index, gt_map in gt_maps.items():
        pred_map = pred_maps[sample_index]
        gt_counts = count_instances_by_class(gt_map, target_label_ids, min_instance_pixels)
        pred_counts = count_instances_by_class(pred_map, target_label_ids, min_instance_pixels)
        for label_id in target_label_ids:
            gt_instance_totals[label_id] += gt_counts[label_id]
            pred_instance_totals[label_id] += pred_counts[label_id]
        per_sample.append(
            {
                "sample_index": sample_index,
                "gt_counts": {str(k): int(v) for k, v in gt_counts.items() if v > 0},
                "pred_counts": {str(k): int(v) for k, v in pred_counts.items() if v > 0},
            }
        )

    gt_total = sum(gt_instance_totals.values())
    pred_total = sum(pred_instance_totals.values())
    gt_freq = np.array(
        [gt_instance_totals[label_id] / max(gt_total, 1) for label_id in target_label_ids],
        dtype=np.float64,
    )
    pred_freq = np.array(
        [pred_instance_totals[label_id] / max(pred_total, 1) for label_id in target_label_ids],
        dtype=np.float64,
    )
    kl_div = categorical_kl(gt_freq, pred_freq)

    plot_path = output_dir / "semantic_ckl_frequency_comparison.png"
    save_frequency_plot(class_names, gt_freq, pred_freq, plot_path)

    return {
        "semantic_ckl": kl_div,
        "min_instance_pixels": min_instance_pixels,
        "class_names": list(class_names),
        "target_label_ids": list(target_label_ids),
        "gt_instance_totals": [int(gt_instance_totals[label_id]) for label_id in target_label_ids],
        "pred_instance_totals": [int(pred_instance_totals[label_id]) for label_id in target_label_ids],
        "gt_frequencies": gt_freq.tolist(),
        "pred_frequencies": pred_freq.tolist(),
        "num_matched_samples": len(gt_maps),
        "per_sample_counts": per_sample,
        "frequency_plot": str(plot_path),
    }


def compute_iou_metrics(
    gt_maps: Dict[int, np.ndarray],
    pred_maps: Dict[int, np.ndarray],
    all_label_ids: Sequence[int],
    furniture_label_ids: Sequence[int],
    id2c: Dict[str, str],
) -> Dict:
    all_intersections = {label_id: 0 for label_id in all_label_ids}
    all_unions = {label_id: 0 for label_id in all_label_ids}
    furniture_set = set(furniture_label_ids)
    per_sample = []

    for sample_index, gt_map in gt_maps.items():
        pred_map = pred_maps[sample_index]
        sample_intersections = {}
        sample_unions = {}
        sample_ious = {}

        for label_id in all_label_ids:
            gt_mask = gt_map == label_id
            pred_mask = pred_map == label_id
            intersection = int(np.logical_and(gt_mask, pred_mask).sum())
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            all_intersections[label_id] += intersection
            all_unions[label_id] += union
            sample_intersections[label_id] = intersection
            sample_unions[label_id] = union
            if union > 0:
                sample_ious[str(label_id)] = float(intersection / union)

        gt_occ = np.isin(gt_map, list(furniture_set))
        pred_occ = np.isin(pred_map, list(furniture_set))
        occ_intersection = int(np.logical_and(gt_occ, pred_occ).sum())
        occ_union = int(np.logical_or(gt_occ, pred_occ).sum())
        occupancy_iou = float(occ_intersection / occ_union) if occ_union > 0 else None

        furniture_sample_ious = [
            float(sample_intersections[label_id] / sample_unions[label_id])
            for label_id in furniture_label_ids
            if sample_unions[label_id] > 0
        ]
        all_sample_ious = [
            float(sample_intersections[label_id] / sample_unions[label_id])
            for label_id in all_label_ids
            if sample_unions[label_id] > 0
        ]

        per_sample.append(
            {
                "sample_index": sample_index,
                "occupancy_iou": occupancy_iou,
                "sample_mean_iou_all_present_classes": float(np.mean(all_sample_ious)) if all_sample_ious else None,
                "sample_mean_iou_furniture_present_classes": float(np.mean(furniture_sample_ious)) if furniture_sample_ious else None,
                "per_class_iou": sample_ious,
            }
        )

    per_class_iou = {}
    present_class_ious = []
    furniture_present_ious = []
    for label_id in all_label_ids:
        union = all_unions[label_id]
        iou = float(all_intersections[label_id] / union) if union > 0 else None
        class_name = id2c[str(label_id)]
        per_class_iou[str(label_id)] = {
            "class_name": class_name,
            "iou": iou,
            "intersection": int(all_intersections[label_id]),
            "union": int(union),
        }
        if iou is not None:
            present_class_ious.append(iou)
            if label_id in furniture_set:
                furniture_present_ious.append(iou)

    occupancy_intersection_total = int(
        sum(
            np.logical_and(np.isin(gt_maps[idx], list(furniture_set)), np.isin(pred_maps[idx], list(furniture_set))).sum()
            for idx in gt_maps
        )
    )
    occupancy_union_total = int(
        sum(
            np.logical_or(np.isin(gt_maps[idx], list(furniture_set)), np.isin(pred_maps[idx], list(furniture_set))).sum()
            for idx in gt_maps
        )
    )
    occupancy_iou = float(occupancy_intersection_total / occupancy_union_total) if occupancy_union_total > 0 else None

    pixel_correct = int(sum((gt_maps[idx] == pred_maps[idx]).sum() for idx in gt_maps))
    pixel_total = int(sum(gt_maps[idx].size for idx in gt_maps))

    return {
        "num_matched_samples": len(gt_maps),
        "label_ids_all": list(all_label_ids),
        "label_ids_furniture": list(furniture_label_ids),
        "mean_iou_all_present_classes": float(np.mean(present_class_ious)) if present_class_ious else None,
        "mean_iou_furniture_present_classes": float(np.mean(furniture_present_ious)) if furniture_present_ious else None,
        "occupancy_iou": occupancy_iou,
        "pixel_accuracy": float(pixel_correct / pixel_total) if pixel_total > 0 else None,
        "occupancy_intersection": occupancy_intersection_total,
        "occupancy_union": occupancy_union_total,
        "per_class_iou": per_class_iou,
        "per_sample": per_sample,
    }


def get_label_id_by_name(id2c: Dict[str, str], target_name: str) -> Optional[int]:
    for label_id, class_name in id2c.items():
        if class_name == target_name:
            return int(label_id)
    return None


def compute_2d_oob_metrics(
    gt_maps: Dict[int, np.ndarray],
    pred_maps: Dict[int, np.ndarray],
    furniture_label_ids: Sequence[int],
    void_label_id: int,
    door_label_id: Optional[int],
    window_label_id: Optional[int],
) -> Dict:
    furniture_set = set(furniture_label_ids)
    total_pred_furniture_pixels = 0
    total_oob_pixels = 0
    total_inside_pixels = 0
    total_blocked_door_pixels = 0
    total_blocked_window_pixels = 0
    total_door_pixels = 0
    total_window_pixels = 0
    per_sample = []

    for sample_index, gt_map in gt_maps.items():
        pred_map = pred_maps[sample_index]
        pred_furniture = np.isin(pred_map, list(furniture_set))
        gt_room_mask = gt_map != void_label_id
        pred_furniture_pixels = int(pred_furniture.sum())
        oob_pixels = int(np.logical_and(pred_furniture, ~gt_room_mask).sum())
        inside_pixels = pred_furniture_pixels - oob_pixels

        door_pixels = 0
        blocked_door_pixels = 0
        if door_label_id is not None:
            gt_door = gt_map == door_label_id
            door_pixels = int(gt_door.sum())
            blocked_door_pixels = int(np.logical_and(pred_furniture, gt_door).sum())

        window_pixels = 0
        blocked_window_pixels = 0
        if window_label_id is not None:
            gt_window = gt_map == window_label_id
            window_pixels = int(gt_window.sum())
            blocked_window_pixels = int(np.logical_and(pred_furniture, gt_window).sum())

        total_pred_furniture_pixels += pred_furniture_pixels
        total_oob_pixels += oob_pixels
        total_inside_pixels += inside_pixels
        total_blocked_door_pixels += blocked_door_pixels
        total_blocked_window_pixels += blocked_window_pixels
        total_door_pixels += door_pixels
        total_window_pixels += window_pixels

        per_sample.append(
            {
                "sample_index": sample_index,
                "predicted_furniture_pixels": pred_furniture_pixels,
                "oob_pixels": oob_pixels,
                "inside_room_pixels": inside_pixels,
                "furniture_oob_pixel_ratio": float(oob_pixels / pred_furniture_pixels) if pred_furniture_pixels > 0 else None,
                "mask_respect_ratio": float(inside_pixels / pred_furniture_pixels) if pred_furniture_pixels > 0 else None,
                "blocked_door_pixels": blocked_door_pixels,
                "blocked_window_pixels": blocked_window_pixels,
                "door_blocking_ratio": float(blocked_door_pixels / door_pixels) if door_pixels > 0 else None,
                "window_blocking_ratio": float(blocked_window_pixels / window_pixels) if window_pixels > 0 else None,
            }
        )

    return {
        "num_matched_samples": len(gt_maps),
        "predicted_furniture_pixels": total_pred_furniture_pixels,
        "oob_pixels": total_oob_pixels,
        "inside_room_pixels": total_inside_pixels,
        "furniture_oob_pixel_ratio": float(total_oob_pixels / total_pred_furniture_pixels) if total_pred_furniture_pixels > 0 else None,
        "mask_respect_ratio": float(total_inside_pixels / total_pred_furniture_pixels) if total_pred_furniture_pixels > 0 else None,
        "door_pixels": total_door_pixels,
        "window_pixels": total_window_pixels,
        "blocked_door_pixels": total_blocked_door_pixels,
        "blocked_window_pixels": total_blocked_window_pixels,
        "door_blocking_ratio": float(total_blocked_door_pixels / total_door_pixels) if total_door_pixels > 0 else None,
        "window_blocking_ratio": float(total_blocked_window_pixels / total_window_pixels) if total_window_pixels > 0 else None,
        "per_sample": per_sample,
    }


def build_summary_metrics(results: Dict) -> Dict:
    summary = {}
    if "iou" in results:
        iou = results["iou"]
        summary["pixel_accuracy"] = iou.get("pixel_accuracy")
        summary["mean_iou_all_present_classes"] = iou.get("mean_iou_all_present_classes")
        summary["mean_iou_furniture_present_classes"] = iou.get("mean_iou_furniture_present_classes")
        summary["occupancy_iou"] = iou.get("occupancy_iou")
    if "count_2d" in results:
        count = results["count_2d"]
        summary["count_instance_f1"] = count.get("instance_f1")
        summary["count_class_exact_match_ratio"] = count.get("class_exact_match_ratio")
        summary["count_mean_l1_error_per_present_class"] = count.get("mean_count_l1_error_per_present_class")
    if "oar_2d" in results:
        oar = results["oar_2d"]
        summary["oar_relation_f1"] = oar.get("relation_f1")
        summary["oar_relation_exact_match_ratio"] = oar.get("relation_exact_match_ratio")
    if "nav_2d" in results:
        nav = results["nav_2d"]
        summary["mean_pred_navigability_ratio"] = nav.get("mean_pred_navigability_ratio")
        summary["mean_gt_navigability_ratio"] = nav.get("mean_gt_navigability_ratio")
        summary["mean_navigability_abs_error"] = nav.get("mean_navigability_abs_error")
    if "oob_2d" in results:
        oob = results["oob_2d"]
        summary["mask_respect_ratio"] = oob.get("mask_respect_ratio")
        summary["furniture_oob_pixel_ratio"] = oob.get("furniture_oob_pixel_ratio")
        summary["door_blocking_ratio"] = oob.get("door_blocking_ratio")
        summary["window_blocking_ratio"] = oob.get("window_blocking_ratio")
    if "ckl" in results:
        summary["semantic_ckl"] = results["ckl"].get("semantic_ckl")
    if "fid_kid" in results and results["fid_kid"].get("status") == "ok":
        summary["fid_mean"] = results["fid_kid"].get("fid_mean")
        summary["kid_mean"] = results["fid_kid"].get("kid_mean")
    if "sca" in results and results["sca"].get("status") == "ok":
        summary["sca_mean_accuracy"] = results["sca"].get("mean_accuracy")
    return summary


def maybe_run_fid_kid(config: Dict, gt_dir: Path, pred_dir: Path, output_dir: Path) -> Dict:
    try:
        fid_module = load_module_from_project("sldn_eval_fid_metric", "eval/fid_metric.py")
        FIDEvaluator = fid_module.FIDEvaluator
    except Exception as exc:
        return {"status": "failed", "error": f"FID/KID dependency unavailable: {exc}"}

    evaluator = FIDEvaluator(
        device=config.get("fid", {}).get("device", None),
        num_iterations=int(config.get("fid", {}).get("num_iterations", 10)),
        num_workers=int(config.get("fid", {}).get("num_workers", 0)),
        use_dataparallel=bool(config.get("fid", {}).get("use_dataparallel", False)),
    )
    try:
        result = evaluator.evaluate(
            path_to_real_renderings=str(gt_dir),
            path_to_synthesized_renderings=str(pred_dir),
            output_directory=str(output_dir),
            temp_dir_base=config.get("fid", {}).get("temp_dir_base", None),
            verbose=True,
        )
        result["status"] = "ok"
        return result
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def build_sca_synth_dir(pred_render_dir: Path, output_dir: Path) -> Path:
    sca_dir = output_dir / "renderings" / "pred_sca"
    if sca_dir.exists():
        shutil.rmtree(sca_dir)
    sca_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_render_dir.glob("*.png"))
    for idx, src in enumerate(pred_files):
        shutil.copy2(src, sca_dir / f"{idx:05d}.png")
    offset = len(pred_files)
    for idx, src in enumerate(pred_files):
        shutil.copy2(src, sca_dir / f"{offset + idx:05d}.png")
    return sca_dir


def maybe_run_sca(config: Dict, gt_train_dir: Path, gt_eval_dir: Path, pred_render_dir: Path, output_dir: Path) -> Dict:
    try:
        sca_module = load_module_from_project("sldn_eval_sca_metric", "eval/sca.py")
        SceneClassificationAccuracyEvaluator = sca_module.SceneClassificationAccuracyEvaluator
    except Exception as exc:
        return {"status": "failed", "error": f"SCA dependency unavailable: {exc}"}

    synth_dir = build_sca_synth_dir(pred_render_dir, output_dir)
    evaluator = SceneClassificationAccuracyEvaluator(
        batch_size=int(config.get("sca", {}).get("batch_size", 256)),
        num_workers=int(config.get("sca", {}).get("num_workers", 0)),
        epochs=int(config.get("sca", {}).get("epochs", 10)),
        device=config.get("sca", {}).get("device", None),
    )
    try:
        result = evaluator.evaluate(
            path_to_train_renderings=str(gt_train_dir),
            path_to_test_renderings=str(gt_eval_dir),
            path_to_synthesized_renderings=str(synth_dir),
            output_directory=str(output_dir),
            verbose=True,
        )
        result["status"] = "ok"
        result["synthetic_renderings_dir"] = str(synth_dir)
        return result
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "synthetic_renderings_dir": str(synth_dir)}


def main() -> None:
    import time
    start_time = time.time()
    
    print("[INFO] ========== SLDN Evaluation Start ==========")
    args = parse_args()
    config = merge_config(args)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    print("[INFO] Loading configuration files...")
    id2c = load_json(Path(config["id2c_path"]))
    c2rgb = load_json(Path(config["c2rgb_path"]))
    palette = build_palette(id2c, c2rgb)
    print(f"[INFO] Loaded {len(id2c)} classes and {len(palette)} palette colors")

    gt_data_dir = Path(config["gt_data_dir"])
    split = config["split"]
    gt_path = gt_data_dir / config["gt_template"].format(split=split)
    gt_train_path = gt_data_dir / config["gt_template"].format(split="train")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT split file not found: {gt_path}")
    if not gt_train_path.exists():
        raise FileNotFoundError(f"GT train file not found: {gt_train_path}")

    print(f"[INFO] Loading GT {split} data: {gt_path.name} ({gt_path.stat().st_size / 1e6:.1f}MB)...")
    gt_split_data = np.load(gt_path, allow_pickle=True)
    print(f"[INFO] Loaded GT {split}: shape {gt_split_data.shape}")
    
    print(f"[INFO] Loading GT train data: {gt_train_path.name}...")
    gt_train_data = np.load(gt_train_path, allow_pickle=True)
    print(f"[INFO] Loaded GT train: shape {gt_train_data.shape}")

    prediction_samples = discover_prediction_samples(config)
    print(f"[INFO] Discovered {len(prediction_samples)} prediction samples")
    
    matched_samples = []
    gt_eval_maps: Dict[int, np.ndarray] = {}
    pred_eval_maps: Dict[int, np.ndarray] = {}
    decode_warnings = []

    gt_eval_render_dir = output_dir / "renderings" / "gt_eval"
    gt_train_render_dir = output_dir / "renderings" / "gt_train"
    pred_render_dir = output_dir / "renderings" / "pred_eval"
    for render_dir in [gt_eval_render_dir, gt_train_render_dir, pred_render_dir]:
        if render_dir.exists():
            shutil.rmtree(render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)

    # Render full GT train set once for SCA.
    train_channel = int(config.get("channel_index", 1))
    print(f"[INFO] Rendering GT train set ({len(gt_train_data)} samples)...")
    for idx in range(len(gt_train_data)):
        gt_map = normalize_map_array(gt_train_data[idx], train_channel)
        save_rendered_png(gt_map, palette, gt_train_render_dir / f"sample_{idx:05d}.png")
    print(f"[INFO] GT train rendering complete")

    print(f"[INFO] Processing {len(prediction_samples)} prediction samples...")
    for i, sample in enumerate(prediction_samples):
        if i % 10 == 0 or i == len(prediction_samples) - 1:
            print(f"[INFO]   {i+1}/{len(prediction_samples)} samples processed")
        if sample.sample_index >= len(gt_split_data):
            continue
        gt_map = normalize_map_array(gt_split_data[sample.sample_index], train_channel)
        pred_map, warnings = load_prediction_map(sample, id2c, c2rgb)
        decode_warnings.extend(
            [f"{sample.source_path.name}: {warning}" for warning in warnings]
        )
        gt_eval_maps[sample.sample_index] = gt_map
        pred_eval_maps[sample.sample_index] = pred_map

        render_name = f"sample_{sample.sample_index:05d}.png"
        save_rendered_png(gt_map, palette, gt_eval_render_dir / render_name)
        save_rendered_png(pred_map, palette, pred_render_dir / render_name)
        matched_samples.append(
            {
                "sample_index": sample.sample_index,
                "prediction_file": str(sample.source_path),
                "prediction_kind": sample.source_kind,
                "gt_file": str(gt_path),
                "render_name": render_name,
            }
        )
    print(f"[INFO] Matched {len(matched_samples)} samples")

    if not matched_samples:
        raise ValueError("No matched GT/prediction samples were found.")

    save_json(
        {
            "config": config,
            "num_matched_samples": len(matched_samples),
            "matched_samples": matched_samples,
            "decode_warnings": decode_warnings,
        },
        output_dir / "matched_samples.json",
    )

    ignore_label_names = set(config.get("ignore_label_names", []))
    target_pairs = [
        (int(label_id), class_name)
        for label_id, class_name in sorted(((int(k), v) for k, v in id2c.items()), key=lambda x: x[0])
        if class_name not in ignore_label_names
    ]
    target_label_ids = [label_id for label_id, _ in target_pairs]
    class_names = [class_name for _, class_name in target_pairs]
    all_non_void_pairs = [
        (int(label_id), class_name)
        for label_id, class_name in sorted(((int(k), v) for k, v in id2c.items()), key=lambda x: x[0])
        if class_name != "void"
    ]
    all_non_void_label_ids = [label_id for label_id, _ in all_non_void_pairs]
    void_label_id = get_label_id_by_name(id2c, "void")
    if void_label_id is None:
        raise ValueError("Could not find 'void' in id2c.json, cannot compute 2D OOB.")
    door_label_id = get_label_id_by_name(id2c, "door")
    window_label_id = get_label_id_by_name(id2c, "window")

    results = {
        "config": config,
        "matched_sample_count": len(matched_samples),
        "render_dirs": {
            "gt_train": str(gt_train_render_dir),
            "gt_eval": str(gt_eval_render_dir),
            "pred_eval": str(pred_render_dir),
        },
        "warnings": decode_warnings,
    }

    metrics = set(config.get("metrics", []))
    print(f"[INFO] Computing metrics: {', '.join(sorted(metrics))}")
    print(f"[INFO] Number of matched sample pairs: {len(matched_samples)}")
    
    if "iou" in metrics:
        print(f"[INFO] Computing IoU metrics...")
        iou_result = compute_iou_metrics(
            gt_maps=gt_eval_maps,
            pred_maps=pred_eval_maps,
            all_label_ids=all_non_void_label_ids,
            furniture_label_ids=target_label_ids,
            id2c=id2c,
        )
        iou_result["status"] = "ok"
        results["iou"] = iou_result
        save_json(iou_result, output_dir / "iou_results.json")
        print(f"[OK] IoU - mean_iou_furniture: {iou_result.get('mean_iou_furniture_present_classes', 'N/A'):.4f}")

    if "count_2d" in metrics:
        print(f"[INFO] Computing Count-2D metrics...")
        count_cfg = config.get("count_2d", {})
        count_result = compute_count_2d_metrics(
            gt_maps=gt_eval_maps,
            pred_maps=pred_eval_maps,
            furniture_label_ids=target_label_ids,
            id2c=id2c,
            min_instance_pixels=int(config.get("min_instance_pixels", 4)),
            connectivity=int(count_cfg.get("connectivity", 8)),
        )
        count_result["status"] = "ok"
        results["count_2d"] = count_result
        save_json(count_result, output_dir / "count_2d_results.json")
        print(f"[OK] Count-2D - F1: {count_result.get('instance_f1', 'N/A'):.4f}")

    if "oar_2d" in metrics:
        print(f"[INFO] Computing OAR-2D metrics...")
        oar_cfg = config.get("oar_2d", {})
        oar_result = compute_oar_2d_metrics(
            gt_maps=gt_eval_maps,
            pred_maps=pred_eval_maps,
            furniture_label_ids=target_label_ids,
            id2c=id2c,
            min_instance_pixels=int(config.get("min_instance_pixels", 4)),
            void_label_id=void_label_id,
            door_label_id=door_label_id,
            window_label_id=window_label_id,
            config=oar_cfg,
        )
        oar_result["status"] = "ok"
        results["oar_2d"] = oar_result
        save_json(oar_result, output_dir / "oar_2d_results.json")
        print(f"[OK] OAR-2D - F1: {oar_result.get('relation_f1', 'N/A'):.4f}")

    if "nav_2d" in metrics:
        print(f"[INFO] Computing NAV-2D metrics...")
        nav_cfg = config.get("nav_2d", {})
        nav_result = compute_nav_2d_metrics(
            gt_maps=gt_eval_maps,
            pred_maps=pred_eval_maps,
            furniture_label_ids=target_label_ids,
            void_label_id=void_label_id,
            door_label_id=door_label_id,
            window_label_id=window_label_id,
            connectivity=int(nav_cfg.get("connectivity", 8)),
        )
        nav_result["status"] = "ok"
        results["nav_2d"] = nav_result
        save_json(nav_result, output_dir / "nav_2d_results.json")
        print(f"[OK] NAV-2D - abs_error: {nav_result.get('mean_navigability_abs_error', 'N/A'):.4f}")

    if "oob_2d" in metrics:
        print(f"[INFO] Computing OOB-2D metrics...")
        oob_result = compute_2d_oob_metrics(
            gt_maps=gt_eval_maps,
            pred_maps=pred_eval_maps,
            furniture_label_ids=target_label_ids,
            void_label_id=void_label_id,
            door_label_id=door_label_id,
            window_label_id=window_label_id,
        )
        oob_result["status"] = "ok"
        results["oob_2d"] = oob_result
        save_json(oob_result, output_dir / "oob_2d_results.json")
        print(f"[OK] OOB-2D - furniture_oob_ratio: {oob_result.get('furniture_oob_pixel_ratio', 'N/A'):.4f}")

    if "ckl" in metrics:
        print(f"[INFO] Computing semantic CKL...")
        ckl_result = run_semantic_ckl(
            gt_maps=gt_eval_maps,
            pred_maps=pred_eval_maps,
            target_label_ids=target_label_ids,
            class_names=class_names,
            min_instance_pixels=int(config.get("min_instance_pixels", 4)),
            output_dir=output_dir,
        )
        ckl_result["status"] = "ok"
        results["ckl"] = ckl_result
        save_json(ckl_result, output_dir / "semantic_ckl_results.json")
        print(f"[OK] CKL - semantic_ckl: {ckl_result.get('semantic_ckl', 'N/A'):.6f}")

    if "fid_kid" in metrics:
        print(f"[INFO] Computing FID/KID...")
        fid_result = maybe_run_fid_kid(config, gt_eval_render_dir, pred_render_dir, output_dir)
        results["fid_kid"] = fid_result
        save_json(fid_result, output_dir / "fid_kid_results.json")

    if "sca" in metrics:
        print(f"[INFO] Computing SCA...")
        sca_result = maybe_run_sca(config, gt_train_render_dir, gt_eval_render_dir, pred_render_dir, output_dir)
        results["sca"] = sca_result
        save_json(sca_result, output_dir / "sca_results.json")

    results["summary_metrics"] = build_summary_metrics(results)
    save_json(results, output_dir / "evaluation_summary.json")
    
    elapsed = time.time() - start_time
    print(f"[INFO] ========== Evaluation Complete ==========")
    print(f"[INFO] Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"[INFO] Results saved to: {output_dir}")
    print(f"[INFO] Summary: {json.dumps(results.get('summary_metrics', {}), indent=2, ensure_ascii=False)}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
