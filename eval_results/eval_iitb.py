#!/usr/bin/env python3
"""
IITB Solid Waste Dataset-2 evaluation.

Features:
  - Auto white-box detection using HSV color segmentation + morphology
  - Falls back to full-image if detection fails
  - GPU-accelerated DINOv3 depth
  - Ground plane calibration anchored to box floor (white pixels = floor reference)
  - Intra-pile consistency metrics (CV)
  - Density estimates if GT mass is provided via --gt_json
  - Saves: per-image depth vis, height overlays, box boundary overlays, metrics JSON

Usage:
    python eval_results/eval_iitb.py \
        --img_dir solid_waste_dataset2_iitb_site_pho \
        --out_dir eval_results/iitb_gpu \
        --box_area_m2 0.12 \
        --camera_height_m 1.0 \
        --gt_json eval_results/iitb_gt.json   # optional

GT JSON format (optional, fill in actual measured values):
{
    "pile1": {"mass_kg": 0.670, "volume_L": null},
    "pile2": {"mass_kg": 0.480, "volume_L": null},
    "pile3": {"mass_kg": 0.660, "volume_L": null}
}
"""

import argparse
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
# Default values — override via CLI
DEFAULT_CAMERA_HEIGHT_M = 1.0     # will be calibrated from box floor
DEFAULT_BOX_AREA_M2     = 0.12    # ~35cm x 35cm white box (update to actual)
GROUND_PERCENTILE       = 99.0
MIN_HEIGHT_M            = 0.005   # 5mm minimum threshold
DINOV3_RESIZE           = 518
DEPTH_RANGE_IITB        = (0.3, 2.0)  # camera 0.5-2m above waste pile

# White box HSV detection thresholds
WHITE_S_MAX  = 50
WHITE_V_MIN  = 160
MIN_BOX_FRAC = 0.05   # minimum fraction of image that must be in box
# ───────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def pick_device(prefer: str = "cuda:1") -> torch.device:
    if torch.cuda.is_available():
        for cand in [prefer, "cuda:0", "cuda:1"]:
            idx = int(cand.split(":")[-1]) if ":" in cand else 0
            if idx < torch.cuda.device_count():
                free, _ = torch.cuda.mem_get_info(idx)
                if free > 2 * 1024**3:
                    return torch.device(cand)
    return torch.device("cpu")


def load_dinov3(device: torch.device, depth_range: tuple) -> torch.nn.Module:
    repo = os.getenv("DINOV3_REPO_DIR", "")
    kw = dict(
        pretrained=False,
        weights=os.getenv("DINOV3_DEPTHER_WEIGHTS", "SYNTHMIX"),
        backbone_weights=os.getenv("DINOV3_BACKBONE_WEIGHTS", "LVD1689M"),
        depth_range=depth_range,
    )
    _orig_module_cuda = torch.nn.Module.cuda
    _orig_tensor_cuda = torch.Tensor.cuda
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
    return m.to(device)


def predict_depth(model, image_pil: Image.Image, device: torch.device) -> np.ndarray:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((DINOV3_RESIZE, DINOV3_RESIZE), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    x = tf(image_pil)[None].to(device=device, dtype=torch.float32)
    h, w = image_pil.size[1], image_pil.size[0]
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                d = model(x)
        else:
            d = model(x)
        d = torch.nn.functional.interpolate(d.float(), size=(h, w), mode="bilinear", align_corners=False)
    return d[0, 0].cpu().numpy()


def detect_white_box(img_rgb: np.ndarray, min_box_frac: float = MIN_BOX_FRAC) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auto-detect the white box region.
    Returns (box_mask bool HxW, box_pts 4x2 in pixel coords).
    Falls back to full image if detection fails.
    """
    h, w = img_rgb.shape[:2]
    full_mask = np.ones((h, w), dtype=bool)
    full_pts  = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Work at 10% scale for speed
    scale = 0.1
    sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
    small  = cv2.resize(img_rgb, (sw, sh))

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    white_m = ((hsv[:, :, 1] < WHITE_S_MAX) & (hsv[:, :, 2] > WHITE_V_MIN)).astype(np.uint8) * 255

    kern = np.ones((5, 5), np.uint8)
    wm = cv2.morphologyEx(white_m, cv2.MORPH_CLOSE, kern, iterations=3)
    wm = cv2.morphologyEx(wm, cv2.MORPH_OPEN,  kern, iterations=2)

    cnts, _ = cv2.findContours(wm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return full_mask, full_pts

    # Largest contour
    cnt = max(cnts, key=cv2.contourArea)
    area_frac = cv2.contourArea(cnt) / (sh * sw)
    if area_frac < min_box_frac:
        return full_mask, full_pts

    # Approximate as rectangle
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect)   # 4 points at small scale
    box_full = (box / scale).astype(np.int32)

    # Draw filled polygon on full-res mask
    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(box_mask, [box_full], 255)

    # Sanity: must cover min_box_frac of image
    if box_mask.mean() < min_box_frac * 255 * 0.5:
        return full_mask, full_pts

    return box_mask.astype(bool), box_full.astype(np.float32)


def floor_calibrated_height_map(
    depth_map: np.ndarray,
    box_mask: np.ndarray,
    camera_height_m: float,
    img_rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute height-above-floor map calibrated within the box.
    Strategy:
      1. If white floor pixels identifiable inside box, use those as floor reference
      2. Else fall back to deepest percentile within box
    Returns (height_map, depth_scale, floor_depth_raw).
    """
    h, w = depth_map.shape
    valid = (depth_map > 0) & box_mask

    if not valid.any():
        return np.zeros((h, w), dtype=np.float32), 1.0, camera_height_m

    # ── Floor reference: white pixels inside the box (waste-free floor areas) ──
    floor_mask = np.zeros((h, w), dtype=bool)
    if img_rgb is not None:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        white = (hsv[:, :, 1] < WHITE_S_MAX) & (hsv[:, :, 2] > WHITE_V_MIN)
        floor_candidates = white & box_mask & valid
        # Only use if we have enough floor pixels (at least 2% of box)
        if floor_candidates.sum() > 0.02 * box_mask.sum():
            floor_mask = floor_candidates

    # Fallback: deepest percentile within box
    if not floor_mask.any():
        d_vals = depth_map[valid]
        thr = np.percentile(d_vals, GROUND_PERCENTILE)
        floor_mask = valid & (depth_map >= thr)

    # Fit plane to floor pixels
    ys, xs = np.where(floor_mask)
    zs = depth_map[floor_mask].astype(np.float64)
    if len(zs) >= 10:
        A = np.column_stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(xs))])
        coeff, *_ = np.linalg.lstsq(A, zs, rcond=None)
        yy, xx = np.mgrid[0:h, 0:w]
        plane = (coeff[0] * xx + coeff[1] * yy + coeff[2]).astype(np.float32)
    else:
        plane = np.full((h, w), float(np.median(zs)), dtype=np.float32)

    med_floor = float(np.median(plane[floor_mask]))
    scale = camera_height_m / med_floor if med_floor > 1e-9 else 1.0

    scaled_plane = plane * scale
    scaled_depth = depth_map * scale

    height_raw = scaled_plane - scaled_depth
    height_raw[~valid] = 0.0
    height_raw[~box_mask] = 0.0

    # Adaptive noise floor from floor residuals
    floor_res = height_raw[floor_mask]
    floor_mean = float(np.mean(floor_res)) if floor_res.size else 0.0
    floor_std  = float(np.std(floor_res))  if floor_res.size else 0.0
    thr = max(MIN_HEIGHT_M, floor_mean + 2.5 * floor_std)

    height_map = np.where((height_raw > thr) & box_mask, height_raw - thr, 0.0)
    return height_map.astype(np.float32), scale, med_floor


def compute_volume(height_map: np.ndarray, box_area_m2: float, box_mask: np.ndarray) -> float:
    box_px = float(box_mask.sum())
    if box_px < 1:
        return 0.0
    px_area = box_area_m2 / box_px
    return float(np.sum(height_map) * px_area)


def draw_box_overlay(img_rgb: np.ndarray, box_pts: np.ndarray, height_map: np.ndarray,
                     box_mask: np.ndarray) -> np.ndarray:
    vis = img_rgb.copy()
    obj_mask = height_map > 0
    # Height heatmap
    vmax = float(np.max(height_map[obj_mask])) if obj_mask.any() else 1.0
    heat = plt.get_cmap("hot")(np.clip(height_map / (vmax + 1e-9), 0, 1))[..., :3]
    heat = (heat * 255).astype(np.uint8)
    # Blend
    alpha = 0.6
    vis[obj_mask] = ((1 - alpha) * vis[obj_mask] + alpha * heat[obj_mask]).astype(np.uint8)
    # Draw box boundary
    pts = box_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], True, (0, 255, 0), 4)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",          default="solid_waste_dataset2_iitb_site_pho")
    ap.add_argument("--out_dir",          default="eval_results/iitb_gpu")
    ap.add_argument("--piles",            default="pile1,pile2,pile3")
    ap.add_argument("--box_area_m2",      type=float, default=DEFAULT_BOX_AREA_M2,
                    help="Physical area of white box interior in m² (measure this!)")
    ap.add_argument("--camera_height_m",  type=float, default=DEFAULT_CAMERA_HEIGHT_M,
                    help="Camera height above box floor in metres")
    ap.add_argument("--gt_json",          default=None,
                    help="Optional JSON with GT volumes/masses per pile")
    ap.add_argument("--device",           default="cuda:1")
    ap.add_argument("--pre_cropped",      action="store_true",
                    help="Images already cropped to box → skip auto detection")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load GT if provided
    gt_data: Dict[str, Dict] = {}
    if args.gt_json and Path(args.gt_json).exists():
        with open(args.gt_json) as f:
            gt_data = json.load(f)
        print(f"[gt] loaded from {args.gt_json}")
    else:
        print("[gt] no GT JSON provided — density/error metrics will be skipped")

    device = pick_device(args.device)
    print(f"[device] {device}  ({torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'})")

    depth_range = tuple(map(float, os.getenv("IITB_DEPTH_RANGE", f"{DEPTH_RANGE_IITB[0]},{DEPTH_RANGE_IITB[1]}").split(",")))
    print(f"[config] box_area={args.box_area_m2:.4f}m²  camera_h={args.camera_height_m:.2f}m  depth_range={depth_range}")

    print("[model] loading DINOv3...")
    t0 = time.perf_counter()
    model = load_dinov3(device, depth_range)
    print(f"[model] loaded in {time.perf_counter()-t0:.1f}s")

    piles = [p.strip() for p in args.piles.split(",")]
    img_dir = Path(args.img_dir)
    all_results = {}

    for pile in piles:
        pile_dir = img_dir / pile
        if not pile_dir.exists():
            print(f"[skip] {pile_dir} not found")
            continue

        imgs = sorted([p for p in pile_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
        if not imgs:
            print(f"[skip] no images in {pile_dir}")
            continue

        print(f"\n{'='*50}\n[pile] {pile} — {len(imgs)} images\n{'='*50}")
        pile_results = []

        for j, img_path in enumerate(imgs, 1):
            t_img = time.perf_counter()
            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)

            # ── Box detection ──
            if args.pre_cropped:
                h_i, w_i = img_np.shape[:2]
                box_mask = np.ones((h_i, w_i), dtype=bool)
                box_pts  = np.array([[0, 0], [w_i, 0], [w_i, h_i], [0, h_i]], dtype=np.float32)
            else:
                box_mask, box_pts = detect_white_box(img_np)

            box_coverage = float(box_mask.mean() * 100)

            # ── Depth ──
            depth = predict_depth(model, img_pil, device)

            # ── Height map ──
            hmap, scale, floor_d = floor_calibrated_height_map(
                depth, box_mask, args.camera_height_m, img_np
            )

            # ── Volume ──
            vol_m3 = compute_volume(hmap, args.box_area_m2, box_mask)
            vol_L  = vol_m3 * 1000.0

            # ── Stats ──
            obj_mask = hmap > 0
            fill_pct = float(np.mean(obj_mask[box_mask]) * 100) if box_mask.any() else 0.0
            mean_h_cm = float(np.mean(hmap[obj_mask]) * 100) if obj_mask.any() else 0.0
            max_h_cm  = float(np.max(hmap) * 100)

            # ── GT comparison ──
            gt = gt_data.get(pile, {})
            gt_vol   = gt.get("volume_L")
            gt_mass  = gt.get("mass_kg")
            density  = gt_mass / vol_L if gt_mass and vol_L > 0 else None
            err_pct  = (vol_L - gt_vol) / gt_vol * 100 if gt_vol else None

            elapsed = time.perf_counter() - t_img

            entry = {
                "image": img_path.name,
                "vol_L": round(vol_L, 4),
                "fill_pct": round(fill_pct, 1),
                "mean_h_cm": round(mean_h_cm, 2),
                "max_h_cm": round(max_h_cm, 2),
                "box_coverage_pct": round(box_coverage, 1),
                "depth_scale": round(scale, 4),
                "floor_depth_raw": round(floor_d, 4),
                "runtime_s": round(elapsed, 2),
            }
            if gt_vol is not None:
                entry["gt_vol_L"] = gt_vol
                entry["pct_error"] = round(err_pct, 1)
            if density is not None:
                entry["density_kg_L"] = round(density, 4)

            pile_results.append(entry)

            msg = (f"[{j}/{len(imgs)}] {img_path.name}  vol={vol_L:.2f}L  "
                   f"fill={fill_pct:.0f}%  h_max={max_h_cm:.1f}cm  t={elapsed:.1f}s")
            if err_pct is not None:
                msg += f"  err={err_pct:+.0f}%"
            print(msg)

            # ── Visualise ──
            vis = draw_box_overlay(img_np, box_pts, hmap, box_mask)
            # Downscale for saving (original is 24MP)
            scale_vis = min(1.0, 1024 / max(vis.shape[:2]))
            if scale_vis < 1.0:
                vis = cv2.resize(vis, (int(vis.shape[1]*scale_vis), int(vis.shape[0]*scale_vis)))
            Image.fromarray(vis).save(vis_dir / f"{pile}_{img_path.stem}_overlay.jpg",
                                      quality=85, optimize=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Pile-level summary ──
        vols = [r["vol_L"] for r in pile_results]
        mean_v = float(np.mean(vols))
        std_v  = float(np.std(vols))
        cv_pct = float(std_v / mean_v * 100) if mean_v > 0 else 0.0

        gt = gt_data.get(pile, {})
        gt_vol  = gt.get("volume_L")
        gt_mass = gt.get("mass_kg")

        pile_summary = {
            "pile": pile,
            "n_images": len(vols),
            "mean_vol_L": round(mean_v, 3),
            "std_vol_L":  round(std_v, 3),
            "cv_pct":     round(cv_pct, 1),
            "median_vol_L": round(float(np.median(vols)), 3),
        }
        if gt_vol:
            pile_summary["gt_vol_L"] = gt_vol
            pile_summary["mean_pct_error"] = round((mean_v - gt_vol) / gt_vol * 100, 1)
        if gt_mass:
            pile_summary["gt_mass_kg"] = gt_mass
            pile_summary["implied_density_kg_L"] = round(gt_mass / mean_v, 3) if mean_v > 0 else None

        all_results[pile] = {
            "summary": pile_summary,
            "per_image": pile_results,
        }

        print(f"\n[{pile}] mean={mean_v:.2f}L  std={std_v:.2f}L  CV={cv_pct:.1f}%")

    # ── Global save ──
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary plot ──
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 5), squeeze=False)
    for k, (pile, data) in enumerate(all_results.items()):
        ax = axes[0][k]
        vols = [r["vol_L"] for r in data["per_image"]]
        imgs = [r["image"][:10] for r in data["per_image"]]
        ax.bar(range(len(vols)), vols, color="steelblue", alpha=0.8)
        if data["summary"].get("gt_vol_L"):
            ax.axhline(data["summary"]["gt_vol_L"], color="red", linestyle="--", lw=2, label="GT")
        ax.axhline(data["summary"]["mean_vol_L"], color="orange", linestyle=":", lw=1.5, label="Mean")
        ax.set_xticks(range(len(vols)))
        ax.set_xticklabels(imgs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Volume (L)")
        cv_str = f"CV={data['summary']['cv_pct']:.1f}%"
        ax.set_title(f"{pile} | mean={data['summary']['mean_vol_L']:.2f}L | {cv_str}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "iitb_volume_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Print table ──
    print("\n" + "="*60)
    print("IITB EVALUATION SUMMARY")
    print("="*60)
    for pile, data in all_results.items():
        s = data["summary"]
        print(f"\n  {pile}:")
        for k, v in s.items():
            print(f"    {k:30s}: {v}")

    print(f"\nResults saved to: {out_dir}/")
    print("  results.json, iitb_volume_chart.png, vis/")

    # ── Config reminder ──
    print(f"\n{'─'*60}")
    print("IMPORTANT CALIBRATION NOTES:")
    print(f"  box_area_m2    = {args.box_area_m2:.4f} m²  ← verify against physical box!")
    print(f"  camera_height  = {args.camera_height_m:.2f} m     ← measure actual setup height!")
    print(f"  depth_range    = {depth_range}  ← must bracket (camera_h - max_pile_h) to camera_h")
    print("  Depth range formula: min = camera_h - max_pile_height_m, max = camera_h + 0.3")
    print("─"*60)


if __name__ == "__main__":
    main()
