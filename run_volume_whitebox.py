#!/usr/bin/env python3
"""
Evaluate DINOv3 volume for Solid Waste Dataset II (White Box)
- Detects the white box using OpenCV color/contour masking.
- Computes volume exclusively within the box.
- Adapts ground floor noise thresholding inside the box.
- Calculates intra-pile variance (CV) and density metrics.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
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
# Adjust these to the actual dimensions of the white box
PRESET_BOX_LENGTH_M = 0.4  # Example: 40 cm
PRESET_BOX_WIDTH_M = 0.3   # Example: 30 cm
PRESET_TARGET_GROUND_DEPTH_M = 0.45
PRESET_GROUND_PERCENTILE = 99.5

# Provide the GT mass (kg) and volume (L) for each pile if known.
# Fill in the actual values measured on site.
GT_DATA = {
    "pile1": {"mass_kg": 0.670, "volume_L": 5.0},
    "pile2": {"mass_kg": 0.480, "volume_L": 10.0},
    "pile3": {"mass_kg": 0.660, "volume_L": 15.0},
}

PRESET_DINOV3_REPO_DIR = os.getenv("DINOV3_REPO_DIR", "")
PRESET_DINOV3_GITHUB_REPO = os.getenv("DINOV3_GITHUB_REPO", "facebookresearch/dinov3")
PRESET_DINOV3_DEPTHER_WEIGHTS = os.getenv("DINOV3_DEPTHER_WEIGHTS", "SYNTHMIX")
PRESET_DINOV3_BACKBONE_WEIGHTS = os.getenv("DINOV3_BACKBONE_WEIGHTS", "LVD1689M")
PRESET_DINOV3_RESIZE = int(os.getenv("DINOV3_RESIZE", "896"))
PRESET_DINOV3_MIN_DEPTH = float(os.getenv("DINOV3_MIN_DEPTH", "0.1"))
PRESET_DINOV3_MAX_DEPTH = float(os.getenv("DINOV3_MAX_DEPTH", "0.7"))


def get_dinov3_depther(repo_dir: str, depther_weights: str, backbone_weights: str, min_depth: float, max_depth: float):
    if repo_dir:
        depther = torch.hub.load(
            repo_dir, "dinov3_vit7b16_dd", source="local", pretrained=False,
            weights=depther_weights, backbone_weights=backbone_weights, depth_range=(min_depth, max_depth)
        )
    else:
        depther = torch.hub.load(
            PRESET_DINOV3_GITHUB_REPO, "dinov3_vit7b16_dd", source="github", pretrained=False,
            weights=depther_weights, backbone_weights=backbone_weights, depth_range=(min_depth, max_depth)
        )
    depther.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depther = depther.to(device)
    return depther, device


def make_transform(resize_size: int = 896) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def estimate_depth_with_dinov3(image_pil: Image.Image, depther: torch.nn.Module, device: torch.device) -> np.ndarray:
    transform = make_transform(PRESET_DINOV3_RESIZE)
    x = transform(image_pil)[None].to(device)
    h0, w0 = image_pil.size[1], image_pil.size[0]
    with torch.inference_mode():
        depth_pred = depther(x)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred, size=(h0, w0), mode="bilinear", align_corners=False
        )
    return depth_pred[0, 0].detach().cpu().numpy()


def load_manual_box(image_np: np.ndarray, pile_folder: str, calibration_file: str = "box_calibration.json") -> Tuple[np.ndarray, float]:
    """
    Load the manual 4-point polygon for the white box from the calibration UI.
    """
    h, w, _ = image_np.shape
    box_mask = np.zeros((h, w), dtype=np.uint8)
    
    if not os.path.exists(calibration_file):
        return (box_mask > 0), 0.0
        
    with open(calibration_file, 'r') as f:
        data = json.load(f)
        
    if pile_folder not in data or len(data[pile_folder]) < 3:
        return (box_mask > 0), 0.0
        
    # Draw the filled polygon mask
    pts = np.array(data[pile_folder], dtype=np.int32)
    cv2.fillPoly(box_mask, [pts], 255)
    
    area_px = np.sum(box_mask > 0)
    return (box_mask > 0), float(area_px)


def fit_ground_plane_masked(depth_map: np.ndarray, mask: np.ndarray, ground_percentile: float = 99.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane using only the deepest pixels *within* the provided mask.
    Returns (plane, ground_mask_used_for_fit)
    """
    h, w = depth_map.shape
    valid = (depth_map > 0) & mask
    valid_depths = depth_map[valid]
    
    if valid_depths.size < 20:
        return np.full_like(depth_map, float(np.median(valid_depths)) if valid_depths.size > 0 else 0.45), valid

    threshold = np.percentile(valid_depths, ground_percentile)
    ground_mask = (depth_map >= threshold) & valid
    if ground_mask.sum() < 10:
        ground_mask = valid

    ys, xs = np.where(ground_mask)
    zs = depth_map[ground_mask]

    A = np.column_stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(xs))])
    coeffs, _, _, _ = np.linalg.lstsq(A, zs.astype(np.float64), rcond=None)

    yy, xx = np.mgrid[0:h, 0:w]
    plane = (coeffs[0] * xx + coeffs[1] * yy + coeffs[2]).astype(np.float32)
    return plane, ground_mask


def estimate_height_map_box(
    depth_map: np.ndarray,
    box_mask: np.ndarray,
    target_ground_depth: float,
    ground_percentile: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Height above the fitted internal box floor.
    Adaptive thresholding based on the noise variance of the fitted floor.
    """
    plane, ground_mask = fit_ground_plane_masked(depth_map, box_mask, ground_percentile)

    valid = depth_map > 0
    plane_median = float(np.median(plane[ground_mask])) if ground_mask.any() else 1.0
    depth_scale = target_ground_depth / plane_median if plane_median > 0 else 1.0

    scaled_plane = plane * depth_scale
    scaled_depth = depth_map * depth_scale

    height_map = scaled_plane - scaled_depth
    height_map[~(valid & box_mask)] = 0.0

    floor_residuals = height_map[ground_mask]
    floor_mean = float(np.mean(floor_residuals)) if floor_residuals.size > 0 else 0.0
    floor_std = float(np.std(floor_residuals)) if floor_residuals.size > 0 else 0.0
    
    # Adaptive threshold: Only pixels above the floor noise floor contribute to volume.
    # Because we want threshold as close to zero as possible without integrating structural noise,
    # we rely purely on +2 sigma above the floor mean residual.
    adaptive_threshold = floor_mean + 2.0 * floor_std

    # Apply threshold strictly within the box
    height_map = np.where((height_map > adaptive_threshold) & box_mask, height_map - adaptive_threshold, 0.0)
    return height_map.astype(np.float32), scaled_plane, depth_scale


def height_to_colormap(height_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = np.where(mask, height_map, 0.0)
    vmax = float(np.max(vis)) if np.max(vis) > 0 else 1.0
    cmap = plt.get_cmap("hot")
    colored = cmap(np.clip(vis / (vmax + 1e-12), 0, 1))[..., :3]
    return (colored * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="solid_waste_dataset2_iitb_site_pho")
    parser.add_argument("--output_dir", type=str, default="solid_waste_dataset2_iitb_site_pho/eval_whitebox_output")
    parser.add_argument("--pre_cropped", action="store_true", help="Assume the images are already cropped to the box boundary.")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading DINOv3 model...")
    depther, device = get_dinov3_depther(
        PRESET_DINOV3_REPO_DIR, PRESET_DINOV3_DEPTHER_WEIGHTS, PRESET_DINOV3_BACKBONE_WEIGHTS,
        PRESET_DINOV3_MIN_DEPTH, PRESET_DINOV3_MAX_DEPTH
    )

    results_by_pile = {}
    
    physical_box_area_m2 = PRESET_BOX_LENGTH_M * PRESET_BOX_WIDTH_M

    for pile_folder in ["pile1", "pile2", "pile3"]:
        pile_dir = img_dir / pile_folder
        if not pile_dir.exists() or not pile_dir.is_dir():
            continue
            
        logger.info(f"Processing {pile_folder}...")
        results_by_pile[pile_folder] = []
        
        for img_path in sorted(pile_dir.iterdir()):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            image_name = img_path.name
            image_pil = Image.open(img_path).convert("RGB")
            image_np = np.array(image_pil)
            
            # 1. Identify Box Region Manually
            if args.pre_cropped:
                h_img, w_img, _ = image_np.shape
                box_mask = np.ones((h_img, w_img), dtype=bool)
                box_area_px = float(h_img * w_img)
            else:
                box_mask, box_area_px = load_manual_box(image_np, pile_folder)
                
            if box_area_px == 0:
                logger.warning(f"No manual box found for {pile_folder}. Skipping {image_name}. Please use annotate_tool.py.")
                continue
                
            # Compute real-world pixel area using the proportion of the box
            pixel_area_m2 = physical_box_area_m2 / box_area_px

            # 2. Estimate Depth
            depth_map = estimate_depth_with_dinov3(image_pil, depther, device)
            
            # 3. Compute Adaptive Height Map
            height_map, ground_plane, depth_scale = estimate_height_map_box(
                depth_map, box_mask, PRESET_TARGET_GROUND_DEPTH_M, PRESET_GROUND_PERCENTILE
            )
            
            # 4. Integrate Volume
            valid_obj = height_map > 0
            object_heights = height_map[valid_obj]
            volume_m3 = float(np.sum(object_heights) * pixel_area_m2) if object_heights.size > 0 else 0.0
            volume_l = volume_m3 * 1000.0
            
            # Density Check
            gt_mass = GT_DATA.get(pile_folder, {}).get("mass_kg", 0.0)
            density_kg_l = gt_mass / volume_l if volume_l > 0 else 0.0
            
            logger.info(f"[{pile_folder}] {image_name}: Pred Vol = {volume_l:.2f} L, Apparent Density = {density_kg_l:.3f} kg/L")
            
            # Visualizations
            heat = height_to_colormap(height_map, mask=valid_obj)
            alpha = 0.6
            blend = image_np.copy()
            blend[valid_obj] = (image_np[valid_obj] * (1 - alpha) + heat[valid_obj] * alpha).astype(np.uint8)
            
            # Draw box contour for verification
            contours, _ = cv2.findContours((box_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blend, contours, -1, (0, 255, 0), 3)
            
            Image.fromarray(blend).save(out_dir / f"{pile_folder}_{img_path.stem}_heatmap.png")
            
            results_by_pile[pile_folder].append({
                "image_name": image_name,
                "pred_volume_liters": volume_l,
                "density_kg_l": density_kg_l,
                "box_area_px": box_area_px
            })

    # 5. Compute Variance Metrics
    summary_metrics = {}
    for pile, runs in results_by_pile.items():
        if not runs: continue
        vols = [r["pred_volume_liters"] for r in runs]
        densities = [r["density_kg_l"] for r in runs]
        
        mean_vol = np.mean(vols)
        std_vol = np.std(vols)
        cv = std_vol / mean_vol if mean_vol > 0 else 0.0
        
        summary_metrics[pile] = {
            "num_images": len(runs),
            "mean_pred_volume_L": float(mean_vol),
            "std_pred_volume_L": float(std_vol),
            "coefficient_of_variation_percent": float(cv * 100),
            "mean_density_kg_L": float(np.mean(densities)),
            "std_density_kg_L": float(np.std(densities))
        }
        logger.info(f"Pile {pile} CV (Spread): {cv*100:.2f}% | Mean Density: {np.mean(densities):.3f} ± {np.std(densities):.3f} kg/L")

    with open(out_dir / "whitebox_metrics.json", "w") as f:
        json.dump({"summary": summary_metrics, "raw": results_by_pile}, f, indent=2)

if __name__ == "__main__":
    main()
