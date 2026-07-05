#!/usr/bin/env python3
"""
Evaluate DINOv3 baseline volume without text prompts.
Fixes camera tilt via planar ground fitting (least-squares plane over ground pixels).
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Presets
# 2ft = 0.6096m
PRESET_REAL_WORLD_HEIGHT_M = 0.6096
PRESET_REAL_WORLD_WIDTH_M = 0.6096
# Camera is ~45cm above the scene; objects are 0–30cm tall.
# depth_range tells DINOv3 what metric depths to predict.
# (0.1, 0.7) comfortably covers the scene: ground at 0.45m,
# top of a 30cm object at 0.15m, ± some margin.
PRESET_TARGET_GROUND_DEPTH_M = 0.45
PRESET_MIN_HEIGHT_THRESHOLD_M = 0.005
PRESET_GROUND_PERCENTILE = 99.9

PRESET_DINOV3_REPO_DIR = os.getenv("DINOV3_REPO_DIR", "")
PRESET_DINOV3_GITHUB_REPO = os.getenv("DINOV3_GITHUB_REPO", "facebookresearch/dinov3")
PRESET_DINOV3_DEPTHER_WEIGHTS = os.getenv("DINOV3_DEPTHER_WEIGHTS", "SYNTHMIX")
PRESET_DINOV3_BACKBONE_WEIGHTS = os.getenv("DINOV3_BACKBONE_WEIGHTS", "LVD1689M")
PRESET_DINOV3_RESIZE = int(os.getenv("DINOV3_RESIZE", "896"))
# Correct depth range for 45-cm overhead shots: 0.1m (very close object top) to 0.7m (floor + margin)
PRESET_DINOV3_MIN_DEPTH = float(os.getenv("DINOV3_MIN_DEPTH", "0.1"))
PRESET_DINOV3_MAX_DEPTH = float(os.getenv("DINOV3_MAX_DEPTH", "0.7"))


def get_dinov3_depther(
    repo_dir: str,
    depther_weights: str,
    backbone_weights: str,
    min_depth: float,
    max_depth: float,
):
    if repo_dir:
        depther = torch.hub.load(
            repo_dir,
            "dinov3_vit7b16_dd",
            source="local",
            pretrained=False,
            weights=depther_weights,
            backbone_weights=backbone_weights,
            depth_range=(min_depth, max_depth),
        )
    else:
        depther = torch.hub.load(
            PRESET_DINOV3_GITHUB_REPO,
            "dinov3_vit7b16_dd",
            source="github",
            pretrained=False,
            weights=depther_weights,
            backbone_weights=backbone_weights,
            depth_range=(min_depth, max_depth),
        )
    depther.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depther = depther.to(device)
    return depther, device


def make_transform(resize_size: int = 896) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

def estimate_depth_with_dinov3(image_pil: Image.Image, depther: torch.nn.Module, device: torch.device) -> np.ndarray:
    transform = make_transform(PRESET_DINOV3_RESIZE)
    x = transform(image_pil)[None].to(device)
    h0, w0 = image_pil.size[1], image_pil.size[0]

    with torch.inference_mode():
        depth_pred = depther(x)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred,
            size=(h0, w0),
            mode="bilinear",
            align_corners=False,
        )

    return depth_pred[0, 0].detach().cpu().numpy()


def fit_ground_plane(depth_map: np.ndarray, ground_percentile: float = 99.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a least-squares plane to ground (deepest) pixels.
    Returns (plane, ground_mask).
    """
    h, w = depth_map.shape
    valid = depth_map > 0
    valid_depths = depth_map[valid]
    if valid_depths.size < 20:
        return np.full_like(depth_map, float(np.median(valid_depths))), valid

    threshold = np.percentile(valid_depths, ground_percentile)
    ground_mask = (depth_map >= threshold) & valid
    if ground_mask.sum() < 10:
        ground_mask = valid

    ys, xs = np.where(ground_mask)
    zs = depth_map[ground_mask]

    A = np.column_stack([xs.astype(np.float64),
                         ys.astype(np.float64),
                         np.ones(len(xs))])
    coeffs, _, _, _ = np.linalg.lstsq(A, zs.astype(np.float64), rcond=None)

    yy, xx = np.mgrid[0:h, 0:w]
    plane = (coeffs[0] * xx + coeffs[1] * yy + coeffs[2]).astype(np.float32)
    return plane, ground_mask


def estimate_height_map_plane(
    depth_map: np.ndarray,
    target_ground_depth: float,
    ground_percentile: float,
    min_height_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Height above the fitted (tilted) ground plane.
    Adaptive floor tolerance: residual std of ground pixels defines noise floor.
    Anything below (floor_mean_residual + 3*sigma) is zeroed out, eliminating
    tilt-induced false volume on flat areas.
    """
    plane, ground_mask = fit_ground_plane(depth_map, ground_percentile)

    valid = depth_map > 0
    plane_median = float(np.median(plane[valid])) if valid.any() else 1.0
    depth_scale = target_ground_depth / plane_median if plane_median > 0 else 1.0

    scaled_plane = plane * depth_scale
    scaled_depth = depth_map * depth_scale

    # Raw height above plane
    height_map = scaled_plane - scaled_depth
    height_map[~valid] = 0.0

    # Residuals on actual ground pixels (after scaling)
    floor_residuals = height_map[ground_mask]
    floor_mean = float(np.mean(floor_residuals)) if floor_residuals.size > 0 else 0.0
    floor_std = float(np.std(floor_residuals)) if floor_residuals.size > 0 else 0.0
    # Adaptive threshold: must clear floor noise AND user minimum
    adaptive_threshold = max(min_height_threshold, floor_mean + 3.0 * floor_std)

    height_map = np.where(height_map > adaptive_threshold, height_map - adaptive_threshold, 0.0)
    return height_map.astype(np.float32), scaled_plane, depth_scale


def pixel_area_m2(image_hw: Tuple[int, int], real_h: float, real_w: float) -> float:
    h, w = image_hw
    return (real_h * real_w) / float(h * w)

def height_to_colormap(height_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = np.where(mask, height_map, 0.0)
    vmax = float(np.max(vis)) if np.max(vis) > 0 else 1.0
    cmap = plt.get_cmap("hot")
    colored = cmap(np.clip(vis / (vmax + 1e-12), 0, 1))[..., :3]
    return (colored * 255).astype(np.uint8)


def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    if not y_true:
        return {}

    yt = np.array(y_true, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.float64)
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))

    nz = np.abs(yt) > 1e-12
    mape = float(np.mean(np.abs(err[nz] / yt[nz])) * 100.0) if np.any(nz) else float("nan")

    if yt.size >= 2:
        ss_res = float(np.sum(np.square(err)))
        ss_tot = float(np.sum(np.square(yt - np.mean(yt))))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        r2 = float("nan")

    return {"mae": mae, "rmse": rmse, "mape_percent": mape, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="dataset_1/eval_gt_volume_liters.csv")
    parser.add_argument("--img_dir", type=str, default="dataset_1/jpg_by_object")
    parser.add_argument("--output_dir", type=str, default="dataset_1/eval_baseline_output")
    args = parser.parse_args()

    csv_file = Path(args.csv_file)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Depther
    logger.info("loading DINOv3 model...")
    depther, device = get_dinov3_depther(
        repo_dir=PRESET_DINOV3_REPO_DIR,
        depther_weights=PRESET_DINOV3_DEPTHER_WEIGHTS,
        backbone_weights=PRESET_DINOV3_BACKBONE_WEIGHTS,
        min_depth=PRESET_DINOV3_MIN_DEPTH,
        max_depth=PRESET_DINOV3_MAX_DEPTH,
    )

    y_true = []
    y_pred = []
    results = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image_name"]
            gt_volume = float(row["gt_volume_liters"])
            folder = row["object_folder"]

            img_path = img_dir / folder / image_name
            if not img_path.exists():
                logger.warning(f"Image {img_path} not found. Skipping.")
                continue

            # Process
            image_pil = Image.open(img_path).convert("RGB")
            
            # 1. Depth
            depth_map = estimate_depth_with_dinov3(image_pil, depther, device)
            
            # Save 16-bit raw depth map
            depth_scale_factor = 10000.0
            depth_map_scaled = (depth_map * depth_scale_factor).astype(np.uint16)
            depth_map_path = out_dir / f"{Path(image_name).stem}_depth_raw.png"
            Image.fromarray(depth_map_scaled).save(depth_map_path)

            # Save 8-bit normalized depth visualization (higher = closer)
            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max > d_min:
                depth_vis = ((d_max - depth_map) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_map, dtype=np.uint8)
            cmap = plt.get_cmap("inferno")
            depth_color = (cmap(depth_vis / 255.0)[..., :3] * 255).astype(np.uint8)
            Image.fromarray(depth_color).save(out_dir / f"{Path(image_name).stem}_depth_vis.png")
            
            # 2. Height with planar tilt correction + adaptive floor noise removal
            height_map, ground_plane, depth_scale = estimate_height_map_plane(
                depth_map,
                target_ground_depth=PRESET_TARGET_GROUND_DEPTH_M,
                ground_percentile=PRESET_GROUND_PERCENTILE,
                min_height_threshold=PRESET_MIN_HEIGHT_THRESHOLD_M,
            )
            ground_depth = float(np.median(ground_plane))

            # 3. Volume — height_map already has floor zeroed; any remaining positive pixel is object
            area_m2 = pixel_area_m2(
                height_map.shape, PRESET_REAL_WORLD_HEIGHT_M, PRESET_REAL_WORLD_WIDTH_M
            )

            valid = height_map > 0
            object_heights = height_map[valid]
            
            if object_heights.size == 0:
                volume_m3 = 0.0
            else:
                volume_m3 = float(np.sum(object_heights) * area_m2)
                
            # Create a heatmap visualization
            image_rgb = np.array(image_pil)
            heat = height_to_colormap(height_map, mask=valid)
            # Blend the heatmap over the RGB
            # Only blend where there is valid height > threshold to avoid tinting ground
            alpha = 0.6
            blend = image_rgb.copy()
            blend[valid] = (image_rgb[valid] * (1 - alpha) + heat[valid] * alpha).astype(np.uint8)
            overlay_path = out_dir / f"{Path(image_name).stem}_heatmap_overlay.png"
            Image.fromarray(blend).save(overlay_path)

            volume_l = volume_m3 * 1000.0
            
            y_true.append(gt_volume)
            y_pred.append(volume_l)
            
            res = {
                "image_name": image_name,
                "gt_volume_liters": gt_volume,
                "pred_volume_liters": volume_l,
                "ground_depth_median_m": ground_depth,
                "depth_scale": depth_scale,
            }
            results.append(res)
            
            logger.info(f"{image_name}: GT = {gt_volume:.3f} L, Pred = {volume_l:.3f} L")
            
    metrics = regression_metrics(y_true, y_pred)
    logger.info(f"Metrics: {metrics}")
    
    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump({
            "metrics": metrics,
            "results": results
        }, f, indent=2)
        
    logger.info(f"Saved results and metrics to {out_dir}")

if __name__ == "__main__":
    main()
