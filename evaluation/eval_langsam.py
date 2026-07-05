#!/usr/bin/env python3
"""
Approach 2: LangSAM segmentation + DINOv3 depth for volume estimation.

Pipeline:
  1. Input image → LangSAM (GroundingDINO + SAM2) with text prompt
  2. Get binary segmentation mask for the waste pile
  3. Run DINOv3 depth estimation
  4. Compute height-above-floor ONLY within segmented mask
  5. Integrate volume from masked height map

Advantages over depth-only:
  - Eliminates background/floor from volume integration
  - Handles multiple waste types via text prompt
  - More robust to camera tilt (floor excluded by mask)
  - Can classify waste type from the text prompt

Usage:
    python evaluation/eval_langsam.py \
        --img_dir solid_waste_dataset2_iitb_site_pho \
        --out_dir evaluation/langsam_gpu \
        --prompt "solid waste pile" \
        --box_area_m2 0.12 \
        --camera_height_m 1.0

    # For dataset_1 (comparing to GT):
    python evaluation/eval_langsam.py \
        --img_dir dataset_1/jpg_by_object \
        --csv     dataset_1/eval_gt_volume_liters.csv \
        --out_dir evaluation/langsam_dataset1 \
        --prompt  "object waste item" \
        --scene_area_m2 0.3716 \
        --camera_height_m 0.45
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
DINOV3_RESIZE   = 518
MIN_HEIGHT_M    = 0.005
DEPTH_RANGE_IITB = (0.3, 2.0)
DEPTH_RANGE_DS1  = (0.05, 0.80)
WHITE_S_MAX     = 50
WHITE_V_MIN     = 160
# ───────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def pick_device(prefer: str = "cuda:1") -> torch.device:
    if torch.cuda.is_available():
        for cand in [prefer, "cuda:0", "cuda:1"]:
            idx = int(cand.split(":")[-1]) if ":" in cand else 0
            if idx < torch.cuda.device_count():
                free, _ = torch.cuda.mem_get_info(idx)
                if free > 4 * 1024**3:  # LangSAM + DINOv3 needs ~4GB+
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


def predict_depth(model, img_pil: Image.Image, device: torch.device) -> np.ndarray:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((DINOV3_RESIZE, DINOV3_RESIZE), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    x = tf(img_pil)[None].to(device=device, dtype=torch.float32)
    h, w = img_pil.size[1], img_pil.size[0]
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                d = model(x)
        else:
            d = model(x)
        d = torch.nn.functional.interpolate(d.float(), size=(h, w), mode="bilinear", align_corners=False)
    return d[0, 0].cpu().numpy()


def get_langsam_mask(
    lang_sam_model,
    img_pil: Image.Image,
    prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Run LangSAM to get a segmentation mask for the text prompt.
    Returns (binary mask HxW bool, confidence_score).
    """
    try:
        masks, boxes, phrases, logits = lang_sam_model.predict(
            img_pil, prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if masks is None or len(masks) == 0:
            return None, 0.0

        # Convert to numpy
        if hasattr(masks, "cpu"):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)

        if masks_np.ndim == 3:
            masks_np = masks_np  # shape: (N, H, W)
        elif masks_np.ndim == 4:
            masks_np = masks_np[:, 0, :, :]

        # Union of all masks (all detected waste regions)
        combined = np.any(masks_np.astype(bool), axis=0)

        conf = float(np.mean([l.item() if hasattr(l, "item") else float(l) for l in logits])) if len(logits) > 0 else 0.0
        return combined, conf

    except Exception as e:
        print(f"    [langsam] WARNING: {e}")
        return None, 0.0


def height_map_with_mask(
    depth_map: np.ndarray,
    seg_mask: np.ndarray,
    camera_height_m: float,
    img_rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute height map using floor pixels outside the segmentation mask as reference.
    This is the key advantage: the floor reference is determined from non-pile pixels.
    """
    h, w = depth_map.shape
    valid = depth_map > 0
    background = valid & ~seg_mask  # pixels outside the pile mask

    if background.sum() < 20:
        # Fallback: use deepest percentile of all valid pixels
        thr = np.percentile(depth_map[valid], 99.0)
        background = valid & (depth_map >= thr)

    # Optional: refine background to white floor pixels (most reliable reference)
    if img_rgb is not None:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        white = (hsv[:, :, 1] < WHITE_S_MAX) & (hsv[:, :, 2] > WHITE_V_MIN)
        white_bg = white & background
        if white_bg.sum() > 50:
            background = white_bg

    # Fit plane to background
    ys, xs = np.where(background)
    zs = depth_map[background].astype(np.float64)
    if len(zs) >= 10:
        A = np.column_stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(xs))])
        coeff, *_ = np.linalg.lstsq(A, zs, rcond=None)
        yy, xx = np.mgrid[0:h, 0:w]
        plane = (coeff[0] * xx + coeff[1] * yy + coeff[2]).astype(np.float32)
    else:
        plane = np.full((h, w), float(np.median(zs)) if zs.size > 0 else camera_height_m, dtype=np.float32)

    med_floor = float(np.median(plane[background]))
    scale = camera_height_m / med_floor if med_floor > 1e-9 else 1.0

    scaled_plane = plane * scale
    scaled_depth = depth_map * scale

    height_raw = scaled_plane - scaled_depth
    height_raw[~valid] = 0.0

    # Apply mask: only count height within the segmentation mask
    height_masked = np.where(seg_mask & valid, height_raw, 0.0)

    # Noise threshold from background residuals
    bg_res  = height_raw[background]
    bg_mean = float(np.mean(bg_res)) if bg_res.size else 0.0
    bg_std  = float(np.std(bg_res))  if bg_res.size else 0.0
    thr = max(MIN_HEIGHT_M, bg_mean + 2.0 * bg_std)

    height_map = np.where(height_masked > thr, height_masked - thr, 0.0)
    return height_map.astype(np.float32), scale, med_floor


def compute_volume_from_mask(
    height_map: np.ndarray,
    seg_mask: np.ndarray,
    scene_area_m2: float,
) -> float:
    """Volume from height map, using segmentation mask footprint for pixel area."""
    h, w = height_map.shape
    px_area = scene_area_m2 / float(h * w)
    return float(np.sum(height_map) * px_area)


def regression_metrics(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    err = yp - yt
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz   = np.abs(yt) > 1e-9
    mape = float(np.mean(np.abs(err[nz] / yt[nz])) * 100) if nz.any() else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"mae_L": mae, "rmse_L": rmse, "mape_pct": mape, "r2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",         default="solid_waste_dataset2_iitb_site_pho")
    ap.add_argument("--out_dir",         default="evaluation/langsam_gpu")
    ap.add_argument("--prompt",          default="solid waste pile",
                    help="Text prompt for LangSAM segmentation")
    ap.add_argument("--box_threshold",   type=float, default=0.3)
    ap.add_argument("--text_threshold",  type=float, default=0.25)
    ap.add_argument("--scene_area_m2",   type=float, default=0.12,
                    help="Physical area of scene in m² (box interior or whole frame)")
    ap.add_argument("--camera_height_m", type=float, default=1.0)
    ap.add_argument("--csv",             default=None,
                    help="GT CSV (for dataset_1 eval mode)")
    ap.add_argument("--mode",            choices=["iitb", "dataset1"], default="iitb")
    ap.add_argument("--device",          default="cuda:1")
    ap.add_argument("--gt_json",         default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──
    device = pick_device(args.device)
    print(f"[device] {device}")

    print("[model] loading LangSAM...")
    from lang_sam import LangSAM
    lang_sam = LangSAM()
    print("[model] LangSAM ready")

    depth_range = DEPTH_RANGE_DS1 if args.mode == "dataset1" else DEPTH_RANGE_IITB
    print("[model] loading DINOv3...")
    t0 = time.perf_counter()
    dinov3 = load_dinov3(device, depth_range)
    print(f"[model] DINOv3 ready in {time.perf_counter()-t0:.1f}s")

    # ── Load GT ──
    gt_data = {}
    if args.gt_json and Path(args.gt_json).exists():
        with open(args.gt_json) as f:
            gt_data = json.load(f)

    rows = []
    y_true_all, y_pred_all = [], []

    # ── IITB mode ──
    if args.mode == "iitb":
        piles = ["pile1", "pile2", "pile3"]
        img_dir = Path(args.img_dir)

        all_results = {}
        for pile in piles:
            pile_dir = img_dir / pile
            if not pile_dir.exists():
                continue
            imgs = sorted([p for p in pile_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
            print(f"\n[pile] {pile} — {len(imgs)} images")
            pile_results = []

            for j, img_path in enumerate(imgs, 1):
                t_img = time.perf_counter()
                img_pil = Image.open(img_path).convert("RGB")
                img_np  = np.array(img_pil)

                # LangSAM segmentation
                print(f"  [{j}] {img_path.name} — segmenting '{args.prompt}'...")
                seg_mask, conf = get_langsam_mask(
                    lang_sam, img_pil, args.prompt,
                    args.box_threshold, args.text_threshold
                )
                h_i, w_i = img_np.shape[:2]
                if seg_mask is None:
                    print(f"    no mask found — using full frame")
                    seg_mask = np.ones((h_i, w_i), dtype=bool)
                    conf = 0.0

                mask_frac = float(seg_mask.mean() * 100)

                # Depth
                depth = predict_depth(dinov3, img_pil, device)

                # Height map
                hmap, scale, floor_d = height_map_with_mask(
                    depth, seg_mask, args.camera_height_m, img_np
                )

                # Volume
                vol_L = compute_volume_from_mask(hmap, seg_mask, args.scene_area_m2) * 1000.0

                obj_mask = hmap > 0
                mean_h_cm = float(np.mean(hmap[obj_mask]) * 100) if obj_mask.any() else 0.0
                max_h_cm  = float(np.max(hmap) * 100)
                elapsed   = time.perf_counter() - t_img

                gt = gt_data.get(pile, {})
                density = gt.get("mass_kg", None)
                density_val = density / vol_L if density and vol_L > 0 else None

                entry = {
                    "pile": pile, "image": img_path.name,
                    "vol_L": round(vol_L, 4),
                    "mask_frac_pct": round(mask_frac, 1),
                    "sam_conf": round(conf, 3),
                    "mean_h_cm": round(mean_h_cm, 2),
                    "max_h_cm": round(max_h_cm, 2),
                    "runtime_s": round(elapsed, 2),
                }
                if density_val is not None:
                    entry["density_kg_L"] = round(density_val, 4)
                pile_results.append(entry)
                rows.append(entry)

                print(f"    vol={vol_L:.2f}L  mask={mask_frac:.0f}%  conf={conf:.2f}  "
                      f"h_max={max_h_cm:.1f}cm  t={elapsed:.1f}s")

                # ── Visualise ──
                vis = img_np.copy()
                # Draw mask
                seg_overlay = np.zeros_like(vis)
                seg_overlay[seg_mask] = [0, 200, 0]
                vis = cv2.addWeighted(vis, 0.7, seg_overlay, 0.3, 0)
                # Height heatmap
                vmax = float(np.max(hmap)) if hmap.max() > 0 else 1.0
                heat = plt.get_cmap("hot")(np.clip(hmap / (vmax + 1e-9), 0, 1))[..., :3]
                heat = (heat * 255).astype(np.uint8)
                hobj = hmap > 0
                if hobj.any():
                    vis[hobj] = ((1-0.6)*vis[hobj] + 0.6*heat[hobj]).astype(np.uint8)
                # Downscale
                sc = min(1.0, 1024 / max(vis.shape[:2]))
                if sc < 1.0:
                    vis = cv2.resize(vis, (int(vis.shape[1]*sc), int(vis.shape[0]*sc)))
                Image.fromarray(vis).save(vis_dir / f"{pile}_{img_path.stem}_langsam.jpg", quality=80)

            vols = [r["vol_L"] for r in pile_results]
            mean_v = float(np.mean(vols))
            std_v  = float(np.std(vols))
            cv_pct = float(std_v/mean_v*100) if mean_v > 0 else 0
            all_results[pile] = {
                "mean_vol_L": round(mean_v, 3), "std_vol_L": round(std_v, 3),
                "cv_pct": round(cv_pct, 1), "per_image": pile_results,
            }
            print(f"[{pile}] mean={mean_v:.2f}L  CV={cv_pct:.1f}%  prompt='{args.prompt}'")

        with open(out_dir / "results.json", "w") as f:
            json.dump({"prompt": args.prompt, "results": all_results}, f, indent=2)

        # Plot
        fig, axes = plt.subplots(1, max(1, len(all_results)), figsize=(5*max(1,len(all_results)), 5), squeeze=False)
        for k, (pile, data) in enumerate(all_results.items()):
            ax = axes[0][k]
            vols = [r["vol_L"] for r in data["per_image"]]
            ax.bar(range(len(vols)), vols, color="forestgreen", alpha=0.8)
            ax.axhline(data["mean_vol_L"], color="orange", linestyle="--", lw=2, label="Mean")
            ax.set_title(f"{pile} LangSAM | mean={data['mean_vol_L']:.2f}L | CV={data['cv_pct']:.1f}%")
            ax.set_ylabel("Volume (L)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")
        plt.suptitle(f"LangSAM prompt: '{args.prompt}'", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / "langsam_chart.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Dataset-1 mode ──
    elif args.mode == "dataset1" and args.csv:
        with open(args.csv) as f:
            gt_rows = list(csv.DictReader(f))
        img_dir = Path(args.img_dir)
        y_true, y_pred = [], []

        for i, row in enumerate(gt_rows, 1):
            img_name = row["image_name"]
            gt_vol   = float(row["gt_volume_liters"])
            folder   = row["object_folder"]
            img_path = img_dir / folder / img_name
            if not img_path.exists():
                continue

            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)
            t_img   = time.perf_counter()

            seg_mask, conf = get_langsam_mask(lang_sam, img_pil, args.prompt,
                                               args.box_threshold, args.text_threshold)
            h_i, w_i = img_np.shape[:2]
            if seg_mask is None:
                seg_mask = np.zeros((h_i, w_i), dtype=bool)

            depth = predict_depth(dinov3, img_pil, device)
            hmap, scale, floor_d = height_map_with_mask(depth, seg_mask, args.camera_height_m, img_np)
            vol_L = compute_volume_from_mask(hmap, seg_mask, args.scene_area_m2) * 1000.0

            elapsed = time.perf_counter() - t_img
            err_pct = (vol_L - gt_vol) / gt_vol * 100 if gt_vol > 0 else float("nan")
            y_true.append(gt_vol)
            y_pred.append(vol_L)
            rows.append({
                "image_name": img_name, "folder": folder,
                "gt_vol_L": gt_vol, "pred_vol_L": round(vol_L, 4),
                "pct_error": round(err_pct, 1), "sam_conf": round(conf, 3),
                "mask_frac_pct": round(float(seg_mask.mean()*100), 1),
                "runtime_s": round(elapsed, 2),
            })
            print(f"[{i}/{len(gt_rows)}] {img_name}  GT={gt_vol:.3f}L  Pred={vol_L:.3f}L  "
                  f"err={err_pct:+.0f}%  conf={conf:.2f}  t={elapsed:.1f}s")

            # Save visual
            vis = img_np.copy()
            seg_ov = np.zeros_like(vis)
            seg_ov[seg_mask] = [0, 200, 0]
            vis = cv2.addWeighted(vis, 0.7, seg_ov, 0.3, 0)
            hobj = hmap > 0
            if hobj.any():
                vmax = float(np.max(hmap))
                heat = (plt.get_cmap("hot")(np.clip(hmap/(vmax+1e-9),0,1))[...,:3]*255).astype(np.uint8)
                vis[hobj] = ((1-0.6)*vis[hobj]+0.6*heat[hobj]).astype(np.uint8)
            Image.fromarray(vis).save(vis_dir / f"{Path(img_name).stem}_langsam.jpg", quality=80)

        metrics = regression_metrics(y_true, y_pred) if y_true else {}
        print(f"\nDataset-1 LangSAM metrics: {metrics}")
        with open(out_dir / "results.json", "w") as f:
            json.dump({"prompt": args.prompt, "metrics": metrics, "rows": rows}, f, indent=2)

    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()
