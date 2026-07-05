#!/usr/bin/env python3
"""
IITB calibration — full-frame approach.

Key insight: at 65cm camera height, the 60x60cm box fills ~90% of the iPhone frame.
So we use the ENTIRE image as the box (no detection needed), set box_area=0.36m²,
and calibrate the depth scale against the known camera height.

Also checks the depth maps visually:
  - saves raw depth colourised + height overlay
  - prints per-image sanity: pile shallower than edges (floor)?

Usage:
    python evaluation/run_iitb_fullframe.py --device cuda:1
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

BOX_AREA_M2      = 0.36    # 60 × 60 cm
CAMERA_HEIGHT_M  = 0.65    # camera above box floor
GT_VOLUME_L      = 18.0    # geometric 20×30×30 cm
GT_MASSES        = {"pile1": 0.670, "pile2": 0.480, "pile3": 0.660}

# Fixed height threshold: 1 cm above floor.
# Adaptive was too aggressive; floor noise is ~1-2 cm so 1 cm is reasonable.
HEIGHT_THRESH_M  = 0.01

SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png"}


# ── DAv2 ────────────────────────────────────────────────────────────────────
def load_dav2(device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    mid = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    proc  = AutoImageProcessor.from_pretrained(mid)
    model = AutoModelForDepthEstimation.from_pretrained(mid).eval().to(device)
    return proc, model


def infer_dav2(proc, model, img_pil, device):
    inputs = {k: v.to(device) for k, v in proc(images=img_pil, return_tensors="pt").items()}
    with torch.inference_mode():
        ctx = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad()
        with ctx:
            out = model(**inputs)
        d = torch.nn.functional.interpolate(
            out.predicted_depth.unsqueeze(1).float(),
            (img_pil.size[1], img_pil.size[0]), mode="bilinear", align_corners=False)
    return d[0, 0].cpu().numpy()


# ── Height map (full frame) ──────────────────────────────────────────────────
def compute_height_fullframe(depth_map: np.ndarray, camera_h: float) -> Tuple[np.ndarray, Dict]:
    """
    Full-frame height estimation.
    1. Use the deepest 1% of pixels as the floor reference.
    2. Fit a tilted plane to those pixels (handles camera tilt).
    3. Scale so median floor depth = camera_height_m.
    4. height = scaled_plane - scaled_depth; clip at HEIGHT_THRESH_M.
    Returns (height_map, diagnostics_dict).
    """
    h, w = depth_map.shape
    valid = depth_map > 0

    if not valid.any():
        return np.zeros((h, w), np.float32), {}

    # ── Floor: deepest 1% of valid pixels (box edges/corners = real floor) ──
    thr = np.percentile(depth_map[valid], 99.0)
    floor_mask = valid & (depth_map >= thr)

    ys, xs = np.where(floor_mask)
    zs = depth_map[floor_mask].astype(np.float64)
    if len(zs) >= 10:
        A = np.column_stack([xs.astype(float), ys.astype(float), np.ones(len(xs))])
        coeff, *_ = np.linalg.lstsq(A, zs, rcond=None)
        yy, xx = np.mgrid[0:h, 0:w]
        plane = (coeff[0]*xx + coeff[1]*yy + coeff[2]).astype(np.float32)
    else:
        plane = np.full((h, w), float(np.median(zs)), np.float32)

    med_floor = float(np.median(plane[floor_mask]))
    scale = camera_h / med_floor if med_floor > 1e-9 else 1.0

    scaled_plane = plane * scale
    scaled_depth = depth_map * scale

    height_raw = np.where(valid, scaled_plane - scaled_depth, 0.0)

    # Fixed threshold: above HEIGHT_THRESH_M
    height_map = np.where(height_raw > HEIGHT_THRESH_M, height_raw - HEIGHT_THRESH_M, 0.0).astype(np.float32)

    # ── Sanity check ──────────────────────────────────────────────────────────
    # Compare center region depth vs edge region depth (before scaling)
    he, we = int(0.15*h), int(0.15*w)  # 15% edge strip
    center = depth_map[he:h-he, we:w-we]
    edge_top    = depth_map[:he, :]
    edge_bot    = depth_map[h-he:, :]
    edge_l      = depth_map[:, :we]
    edge_r      = depth_map[:, w-we:]
    edges_all   = np.concatenate([edge_top.flat, edge_bot.flat, edge_l.flat, edge_r.flat])

    mean_center = float(np.mean(center[center > 0]))  if (center > 0).any() else 0.0
    mean_edge   = float(np.mean(edges_all[edges_all > 0])) if (edges_all > 0).any() else 0.0

    # After scaling:
    scaled_diff_cm = (mean_edge - mean_center) * scale * 100
    sanity_ok = mean_center < mean_edge  # pile shallower (lower depth) than floor

    obj_mask  = height_map > 0
    fill_pct  = float(np.mean(obj_mask) * 100)
    mean_h_cm = float(np.mean(height_map[obj_mask]) * 100) if obj_mask.any() else 0.0
    max_h_cm  = float(np.max(height_map) * 100)

    diagnostics = {
        "scale": round(scale, 4),
        "med_floor_raw": round(med_floor, 4),
        "med_floor_m": round(med_floor * scale, 4),
        "center_depth_raw": round(mean_center, 4),
        "edge_depth_raw":   round(mean_edge, 4),
        "pile_height_cm_vs_edges": round(scaled_diff_cm, 2),
        "sanity_ok": sanity_ok,
        "fill_pct": round(fill_pct, 1),
        "mean_h_cm": round(mean_h_cm, 2),
        "max_h_cm": round(max_h_cm, 2),
    }
    return height_map, diagnostics


def volume_fullframe(hmap, box_area_m2):
    """Volume in litres from full-frame height map."""
    h, w = hmap.shape
    px_area = box_area_m2 / float(h * w)
    return float(np.sum(hmap) * px_area) * 1000.0


# ── Visualization ────────────────────────────────────────────────────────────
def save_vis(img_np, depth_map, hmap, diag, out_path_prefix):
    h, w = img_np.shape[:2]

    # 1. Depth map (closer = brighter/lighter)
    valid = depth_map > 0
    d_lo = np.percentile(depth_map[valid], 2) if valid.any() else 0
    d_hi = np.percentile(depth_map[valid], 98) if valid.any() else 1
    inv  = np.clip((d_hi - depth_map) / max(d_hi-d_lo, 1e-9), 0, 1)
    depth_vis = (plt.get_cmap("magma")(inv)[...,:3]*255).astype(np.uint8)
    # Mark center region with rectangle
    he, we = int(0.15*h), int(0.15*w)
    cv2.rectangle(depth_vis, (we, he), (w-we, h-he), (0,255,0), 8)
    sc = min(1.0, 1200/max(h,w))
    if sc < 1.0:
        depth_vis = cv2.resize(depth_vis, (int(w*sc), int(h*sc)))
    Image.fromarray(depth_vis).save(str(out_path_prefix)+"_depth.jpg", quality=82)

    # 2. Height overlay
    overlay = img_np.copy()
    obj = hmap > 0
    if obj.any():
        vmax = float(np.max(hmap))
        heat = (plt.get_cmap("hot")(np.clip(hmap/(vmax+1e-9),0,1))[...,:3]*255).astype(np.uint8)
        overlay[obj] = (0.35*overlay[obj] + 0.65*heat[obj]).astype(np.uint8)
    sc = min(1.0, 1200/max(h,w))
    if sc < 1.0:
        overlay = cv2.resize(overlay, (int(w*sc), int(h*sc)))
    Image.fromarray(overlay).save(str(out_path_prefix)+"_overlay.jpg", quality=82)

    # 3. Side-by-side summary figure (small)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np[:,:,::-1] if img_np.dtype == np.uint8 else img_np)
    axes[0].set_title("Input RGB"); axes[0].axis("off")

    axes[1].imshow(depth_vis)
    axes[1].set_title(f"Depth (magma)\ncentre-edge: {diag['pile_height_cm_vs_edges']:.1f}cm  sanity={'✓' if diag['sanity_ok'] else '✗'}")
    axes[1].axis("off")

    h_show = cv2.resize(np.array(Image.fromarray(overlay)), (640, int(640*h/w)))
    axes[2].imshow(h_show)
    axes[2].set_title(f"Height overlay\nfill={diag['fill_pct']:.0f}%  max_h={diag['max_h_cm']:.1f}cm")
    axes[2].axis("off")

    plt.tight_layout()
    # Downscale input for the figure
    rgb_small = cv2.resize(img_np, (640, int(640*h/w))) if w > 640 else img_np
    axes[0].imshow(rgb_small)
    plt.savefig(str(out_path_prefix)+"_summary.jpg", dpi=100, bbox_inches="tight"); plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",  default="solid_waste_dataset2_iitb_site_pho")
    ap.add_argument("--out_dir",  default="evaluation/iitb_fullframe")
    ap.add_argument("--piles",    default="pile1,pile2,pile3")
    ap.add_argument("--device",   default="cuda:1")
    ap.add_argument("--max_imgs", type=int, default=0)
    ap.add_argument("--save_all_vis", action="store_true",
                    help="Save vis for every image (default: first image only per pile)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    device  = pick_device(args.device)
    piles   = [p.strip() for p in args.piles.split(",")]

    print(f"\n{'='*60}")
    print(f"IITB Full-Frame Volume Calibration — DepthAnythingV2")
    print(f"{'='*60}")
    print(f"  camera_height = {CAMERA_HEIGHT_M}m  |  box_area = {BOX_AREA_M2}m²")
    print(f"  GT_volume     = {GT_VOLUME_L}L  |  height_thresh = {HEIGHT_THRESH_M*100}cm")
    print(f"  device        = {device}")
    print(f"{'='*60}\n")

    print("[model] loading DepthAnythingV2...")
    t0 = time.perf_counter()
    proc, model = load_dav2(device)
    print(f"[model] ready in {time.perf_counter()-t0:.1f}s\n")

    all_rows = []
    pile_summaries = {}

    for pile in piles:
        pile_dir = Path(args.img_dir) / pile
        if not pile_dir.exists():
            print(f"[skip] {pile_dir}"); continue

        imgs = sorted([p for p in pile_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
        if args.max_imgs > 0:
            imgs = imgs[:args.max_imgs]

        print(f"\n{'─'*50}")
        print(f"[pile] {pile} — {len(imgs)} images")
        print(f"{'─'*50}")
        pile_vols, pile_cfs = [], []

        for j, img_path in enumerate(imgs, 1):
            t_img = time.perf_counter()
            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)

            depth = infer_dav2(proc, model, img_pil, device)
            hmap, diag = compute_height_fullframe(depth, CAMERA_HEIGHT_M)
            vol_L = volume_fullframe(hmap, BOX_AREA_M2)
            cf    = GT_VOLUME_L / vol_L if vol_L > 0.1 else float("nan")
            elapsed = time.perf_counter() - t_img

            if not np.isnan(cf):
                pile_vols.append(vol_L); pile_cfs.append(cf)

            row = {
                "pile": pile, "image": img_path.name,
                "pred_vol_L": round(vol_L, 3), "gt_vol_L": GT_VOLUME_L,
                "calib_factor": round(cf, 3) if not np.isnan(cf) else "nan",
                "gt_mass_kg": GT_MASSES.get(pile, 0),
                **diag, "runtime_s": round(elapsed, 2),
            }
            all_rows.append(row)

            sanity_str = "✓" if diag["sanity_ok"] else "✗"
            cf_str = f"CF={cf:.2f}" if not np.isnan(cf) else "CF=?"
            print(f"  [{j}/{len(imgs)}] {img_path.name}  "
                  f"vol={vol_L:.2f}L  {cf_str}  "
                  f"fill={diag['fill_pct']:.0f}%  max_h={diag['max_h_cm']:.1f}cm  "
                  f"pile_vs_floor={diag['pile_height_cm_vs_edges']:.1f}cm  "
                  f"sanity={sanity_str}  t={elapsed:.1f}s")

            # Save vis
            if j == 1 or args.save_all_vis:
                pfx = vis_dir / f"{pile}_{img_path.stem}"
                save_vis(img_np, depth, hmap, diag, pfx)

        torch.cuda.empty_cache() if device.type == "cuda" else None

        # Pile summary
        if pile_vols:
            mv  = float(np.mean(pile_vols))
            sv  = float(np.std(pile_vols))
            cv  = float(sv/mv*100) if mv>0 else 0
            mcf = float(np.mean(pile_cfs))
            scf = float(np.std(pile_cfs))

            gt_mass = GT_MASSES.get(pile, 0.0)
            density = gt_mass / mv if mv > 0 else float("nan")

            pile_summaries[pile] = {
                "n_imgs": len(pile_vols), "mean_vol_L": round(mv,3),
                "std_vol_L": round(sv,3), "cv_pct": round(cv,1),
                "mean_cf": round(mcf,3), "std_cf": round(scf,3),
                "gt_vol_L": GT_VOLUME_L, "gt_mass_kg": gt_mass,
                "pred_density_kg_L": round(density, 4) if not np.isnan(density) else None,
                "calibrated_vol_L": round(mv*mcf, 2),
            }
            print(f"\n  [summary] mean={mv:.2f}L  CF={mcf:.2f}±{scf:.2f}  "
                  f"CV={cv:.0f}%  density={density:.3f}kg/L  "
                  f"calibrated={mv*mcf:.1f}L")
        else:
            pile_summaries[pile] = {"mean_vol_L": 0, "n_imgs": len(imgs)}
            print(f"  [summary] no valid volume predictions")

    # ── Global calibration factor ─────────────────────────────────────────────
    all_cfs = [r["calib_factor"] for r in all_rows if r["calib_factor"] != "nan"]
    if all_cfs:
        gcf = float(np.median(all_cfs))  # median is more robust than mean
        gcf_std = float(np.std(all_cfs))
        print(f"\n{'='*60}")
        print(f"GLOBAL CALIBRATION FACTOR (median): {gcf:.3f} ± {gcf_std:.3f}")
        print(f"  Meaning: multiply raw prediction by {gcf:.2f} to get ~GT volume")
        print(f"  Or equivalently: GT ≈ {gcf:.2f} × predicted")
        print(f"{'='*60}\n")
    else:
        gcf = float("nan")
        print("\n[warn] No valid predictions to compute global CF")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir.mkdir(exist_ok=True)
    if all_rows:
        with open(out_dir/"per_image_results.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader(); w.writerows(all_rows)

    summary = {
        "setup": {"camera_height_m": CAMERA_HEIGHT_M, "box_area_m2": BOX_AREA_M2,
                  "gt_volume_L": GT_VOLUME_L, "height_thresh_m": HEIGHT_THRESH_M,
                  "model": "DepthAnythingV2-Metric-Indoor-Large"},
        "global_calib_factor_median": round(gcf, 3) if not np.isnan(gcf) else None,
        "global_calib_factor_std": round(gcf_std, 3) if all_cfs else None,
        "piles": pile_summaries,
    }
    with open(out_dir/"calibration_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    # ── Comparison chart ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"pile1": "steelblue", "pile2": "darkorange", "pile3": "forestgreen"}
    for pile in piles:
        pile_rows = [r for r in all_rows if r["pile"]==pile and r["pred_vol_L"]>0]
        if pile_rows:
            vols = [r["pred_vol_L"] for r in pile_rows]
            ax.plot(vols, marker="o", label=pile, color=colors.get(pile,"gray"))
    ax.axhline(GT_VOLUME_L, color="red", linestyle="--", lw=2, label=f"GT={GT_VOLUME_L}L")
    if not np.isnan(gcf) and pile_rows:
        # Show example of calibrated range
        sample_pred = np.mean([r["pred_vol_L"] for r in all_rows if r["pred_vol_L"]>0])
        ax.axhline(sample_pred * gcf, color="green", linestyle=":", lw=1.5, label=f"Avg calibrated (CF={gcf:.2f})")
    ax.set_xlabel("Image index"); ax.set_ylabel("Predicted Volume (L)")
    ax.set_title(f"IITB DepthAnythingV2 Full-Frame\ncamera={CAMERA_HEIGHT_M}m, box={BOX_AREA_M2}m², GT={GT_VOLUME_L}L")
    ax.legend(); ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(out_dir/"volume_chart.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"\nOutputs saved to: {out_dir}/")
    print(f"  per_image_results.csv  calibration_summary.json  volume_chart.png")
    print(f"  vis/ — open these to visually verify depth detection:")
    print(f"    *_depth.jpg   : depth map (brighter = closer to camera, green box = center region)")
    print(f"    *_overlay.jpg : height heatmap on RGB (hot = higher pile)")
    print(f"    *_summary.jpg : side-by-side 3-panel")


def pick_device(prefer="cuda:1"):
    if torch.cuda.is_available():
        for cand in [prefer, "cuda:0", "cuda:1"]:
            idx = int(cand.split(":")[-1]) if ":" in cand else 0
            if idx < torch.cuda.device_count():
                free, _ = torch.cuda.mem_get_info(idx)
                if free > 2 * 1024**3:
                    return torch.device(cand)
    return torch.device("cpu")


if __name__ == "__main__":
    main()
