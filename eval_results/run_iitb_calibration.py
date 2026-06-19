#!/usr/bin/env python3
"""
IITB calibration evaluation: run both DINOv3 and DepthAnythingV2 on all piles,
compute calibration factors (CF = GT / predicted), check depth map sanity,
and produce a comprehensive report.

Setup:
  - Camera height: 65 cm
  - Box interior:  60 x 60 cm = 0.36 m²
  - GT volume:     18 L per pile (20x30x30 cm geometric)
  - Waste fluffs, so actual bulk > 18L — CF will tell us the correction

Run:
    python eval_results/run_iitb_calibration.py --device cuda:1
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

# ── Physical setup ──────────────────────────────────────────────────────────
CAMERA_HEIGHT_M  = 0.65      # measured camera height above box floor
BOX_AREA_M2      = 0.60 * 0.60  # 60 x 60 cm interior
GT_VOLUME_L      = 18.0      # geometric measurement (L)
GT_MASSES        = {"pile1": 0.670, "pile2": 0.480, "pile3": 0.660}

# Depth range tuned for 65cm camera height
# min = camera_h - max_pile_height (0.65 - 0.35 = 0.30)
# max = camera_h + margin          (0.65 + 0.25 = 0.90)
DEPTH_RANGE_IITB = (0.30, 0.90)

# ── Constants ────────────────────────────────────────────────────────────────
WHITE_S_MAX      = 50
WHITE_V_MIN      = 160
GROUND_PCT       = 99.0
MIN_HEIGHT_M     = 0.003
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png"}
# ────────────────────────────────────────────────────────────────────────────


def pick_device(prefer="cuda:1"):
    if torch.cuda.is_available():
        for cand in [prefer, "cuda:0", "cuda:1"]:
            idx = int(cand.split(":")[-1]) if ":" in cand else 0
            if idx < torch.cuda.device_count():
                free, _ = torch.cuda.mem_get_info(idx)
                if free > 2 * 1024**3:
                    return torch.device(cand)
    return torch.device("cpu")


# ── DINOv3 loader ────────────────────────────────────────────────────────────
def load_dinov3(device, depth_range=DEPTH_RANGE_IITB):
    kw = dict(pretrained=False,
              weights=os.getenv("DINOV3_DEPTHER_WEIGHTS", "SYNTHMIX"),
              backbone_weights=os.getenv("DINOV3_BACKBONE_WEIGHTS", "LVD1689M"),
              depth_range=depth_range)
    repo = os.getenv("DINOV3_REPO_DIR", "")
    # Patch cuda to prevent auto-move to GPU 0 during init
    _orig_mc = torch.nn.Module.cuda
    _orig_tc = torch.Tensor.cuda
    torch.nn.Module.cuda = lambda self, d=None: self
    torch.Tensor.cuda    = lambda self, d=None, **kw2: self
    try:
        src = repo if repo else os.getenv("DINOV3_GITHUB_REPO", "facebookresearch/dinov3")
        mode = "local" if repo else "github"
        m = torch.hub.load(src, "dinov3_vit7b16_dd", source=mode, trust_repo=True, **kw)
    finally:
        torch.nn.Module.cuda = _orig_mc
        torch.Tensor.cuda    = _orig_tc
    return m.eval().to(device)


def infer_dinov3(model, img_pil, device):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((518, 518), antialias=True),
        transforms.Normalize(mean=(.485,.456,.406), std=(.229,.224,.225)),
    ])
    x = tf(img_pil)[None].to(device, dtype=torch.float32)
    h, w = img_pil.size[1], img_pil.size[0]
    with torch.inference_mode():
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type=="cuda" else torch.no_grad()
        with ctx:
            d = model(x)
        d = torch.nn.functional.interpolate(d.float(), (h, w), mode="bilinear", align_corners=False)
    return d[0, 0].cpu().numpy()


# ── DepthAnythingV2 loader ───────────────────────────────────────────────────
def load_dav2(device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    mid = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    print(f"  loading {mid} ...")
    proc  = AutoImageProcessor.from_pretrained(mid)
    model = AutoModelForDepthEstimation.from_pretrained(mid).eval().to(device)
    return proc, model


def infer_dav2(proc, model, img_pil, device):
    inputs = {k: v.to(device) for k, v in proc(images=img_pil, return_tensors="pt").items()}
    with torch.inference_mode():
        ctx = torch.autocast("cuda", dtype=torch.float16) if device.type=="cuda" else torch.no_grad()
        with ctx:
            out = model(**inputs)
        d = torch.nn.functional.interpolate(
            out.predicted_depth.unsqueeze(1).float(),
            (img_pil.size[1], img_pil.size[0]), mode="bilinear", align_corners=False)
    return d[0, 0].cpu().numpy()


# ── White-box detection ──────────────────────────────────────────────────────
def detect_white_box(img_rgb):
    h, w = img_rgb.shape[:2]
    full = np.ones((h, w), bool)
    full_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    scale = 0.08
    sh, sw = max(1, int(h*scale)), max(1, int(w*scale))
    small = cv2.resize(img_rgb, (sw, sh))
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    wm  = ((hsv[:,:,1] < WHITE_S_MAX) & (hsv[:,:,2] > WHITE_V_MIN)).astype(np.uint8)*255
    k   = np.ones((5,5), np.uint8)
    wm  = cv2.morphologyEx(wm, cv2.MORPH_CLOSE, k, iterations=3)
    wm  = cv2.morphologyEx(wm, cv2.MORPH_OPEN,  k, iterations=2)
    cnts, _ = cv2.findContours(wm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return full, full_pts
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt)/(sh*sw) < 0.04:
        return full, full_pts
    rect = cv2.minAreaRect(cnt)
    box  = (cv2.boxPoints(rect)/scale).astype(np.int32)
    bm   = np.zeros((h,w), np.uint8)
    cv2.fillPoly(bm, [box], 255)
    return bm.astype(bool), box.astype(np.float32)


# ── Height-map calibration ───────────────────────────────────────────────────
def compute_height_map(depth_map, camera_height_m, box_mask, img_rgb=None):
    """
    Returns (height_map, depth_scale, floor_depth_raw, depth_sanity_ok).
    depth_sanity_ok = True if pile pixels are measurably closer to camera than floor.
    """
    h, w = depth_map.shape
    valid = (depth_map > 0) & box_mask

    if not valid.any():
        return np.zeros((h,w), np.float32), 1.0, 0.0, False

    # Floor reference: white pixels inside the box (the box floor is white)
    floor_mask = np.zeros((h,w), bool)
    if img_rgb is not None:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        wht = (hsv[:,:,1]<WHITE_S_MAX) & (hsv[:,:,2]>WHITE_V_MIN) & box_mask & valid
        if wht.sum() > 0.01 * box_mask.sum():
            floor_mask = wht

    if not floor_mask.any():
        thr = np.percentile(depth_map[valid], GROUND_PCT)
        floor_mask = valid & (depth_map >= thr)

    # Fit plane to floor pixels
    ys, xs = np.where(floor_mask)
    zs = depth_map[floor_mask].astype(np.float64)
    if len(zs) >= 10:
        A = np.column_stack([xs.astype(float), ys.astype(float), np.ones(len(xs))])
        coeff,*_ = np.linalg.lstsq(A, zs, rcond=None)
        yy,xx = np.mgrid[0:h,0:w]
        plane = (coeff[0]*xx + coeff[1]*yy + coeff[2]).astype(np.float32)
    else:
        plane = np.full((h,w), float(np.median(zs)), np.float32)

    med_floor = float(np.median(plane[floor_mask]))
    scale = camera_height_m / med_floor if med_floor > 1e-9 else 1.0

    scaled_plane = plane * scale
    scaled_depth = depth_map * scale

    height_raw = np.where(valid, scaled_plane - scaled_depth, 0.0)
    height_raw = np.where(box_mask, height_raw, 0.0)

    # Depth sanity check: mean depth of tall pixels vs floor
    high_pixels = height_raw > 0.02   # pixels > 2cm
    sanity_ok = False
    if high_pixels.any() and floor_mask.any():
        mean_d_pile  = float(np.mean(depth_map[high_pixels]))
        mean_d_floor = float(np.mean(depth_map[floor_mask]))
        # Pile should be CLOSER (lower depth value) than floor
        sanity_ok = mean_d_pile < mean_d_floor
        sanity_diff_cm = (mean_d_floor - mean_d_pile) * 100 * scale
    else:
        sanity_diff_cm = 0.0

    # Adaptive threshold from floor residuals
    floor_res = height_raw[floor_mask]
    fm = float(np.mean(floor_res)) if floor_res.size else 0.0
    fs = float(np.std(floor_res))  if floor_res.size else 0.0
    thr = max(MIN_HEIGHT_M, fm + 2.5*fs)

    hmap = np.where((height_raw > thr) & box_mask, height_raw - thr, 0.0).astype(np.float32)
    return hmap, scale, med_floor, sanity_ok, float(sanity_diff_cm)


def volume_from_height(hmap, box_mask, box_area_m2):
    box_px = float(box_mask.sum())
    if box_px < 1: return 0.0
    px_area = box_area_m2 / box_px
    return float(np.sum(hmap) * px_area) * 1000.0  # → litres


def make_overlay(img_rgb, hmap, box_pts):
    """Create blended overlay: green box boundary + hot heatmap on height."""
    vis = img_rgb.copy()
    obj = hmap > 0
    if obj.any():
        vmax = float(np.max(hmap))
        heat = (plt.get_cmap("hot")(np.clip(hmap/(vmax+1e-9),0,1))[...,:3]*255).astype(np.uint8)
        vis[obj] = (0.4*vis[obj] + 0.6*heat[obj]).astype(np.uint8)
    pts = box_pts.astype(np.int32).reshape(-1,1,2)
    cv2.polylines(vis, [pts], True, (0,255,0), 4)
    # Downscale for saving
    sc = min(1.0, 1200/max(vis.shape[:2]))
    if sc < 1.0:
        vis = cv2.resize(vis, (int(vis.shape[1]*sc), int(vis.shape[0]*sc)))
    return vis


def make_depth_vis(depth_map, box_pts, img_rgb=None):
    """Depth map colourised (closer=brighter) with box outline."""
    valid = depth_map > 0
    vis_d = np.zeros_like(depth_map, np.float32)
    if valid.any():
        d_lo = np.percentile(depth_map[valid], 2)
        d_hi = np.percentile(depth_map[valid], 98)
        # Invert: closer = higher value (brighter)
        vis_d = np.clip((d_hi - depth_map) / max(d_hi-d_lo, 1e-9), 0, 1)
    colored = (plt.get_cmap("magma")(vis_d)[...,:3]*255).astype(np.uint8)
    pts = box_pts.astype(np.int32).reshape(-1,1,2)
    cv2.polylines(colored, [pts], True, (0,255,0), 4)
    sc = min(1.0, 1200/max(colored.shape[:2]))
    if sc < 1.0:
        colored = cv2.resize(colored, (int(colored.shape[1]*sc), int(colored.shape[0]*sc)))
    return colored


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",  default="solid_waste_dataset2_iitb_site_pho")
    ap.add_argument("--out_dir",  default="eval_results/iitb_calibration")
    ap.add_argument("--piles",    default="pile1,pile2,pile3")
    ap.add_argument("--device",   default="cuda:1")
    ap.add_argument("--models",   default="dinov3,dav2",
                    help="Comma-separated list of models to run: dinov3, dav2")
    ap.add_argument("--max_imgs", type=int, default=0,
                    help="Max images per pile (0=all, use 3 for quick test)")
    args = ap.parse_args()

    out_dir  = Path(args.out_dir)
    vis_dir  = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    device  = pick_device(args.device)
    piles   = [p.strip() for p in args.piles.split(",")]
    models  = [m.strip() for m in args.models.split(",")]
    img_dir = Path(args.img_dir)

    print(f"[config] camera_height={CAMERA_HEIGHT_M}m  box={BOX_AREA_M2:.2f}m²  "
          f"GT_vol={GT_VOLUME_L}L  device={device}")

    # ── Load models ──────────────────────────────────────────────────────────
    dinov3_model = dav2_proc = dav2_model = None

    if "dinov3" in models:
        print("\n[dinov3] loading...")
        t0 = time.perf_counter()
        dinov3_model = load_dinov3(device)
        print(f"[dinov3] loaded in {time.perf_counter()-t0:.1f}s")

    if "dav2" in models:
        print("\n[dav2] loading...")
        t0 = time.perf_counter()
        dav2_proc, dav2_model = load_dav2(device)
        print(f"[dav2] loaded in {time.perf_counter()-t0:.1f}s")

    # ── Results containers ────────────────────────────────────────────────────
    # { model_name: { pile_name: [vol_L, ...] } }
    results: Dict[str, Dict[str, List[float]]] = {m: {p: [] for p in piles} for m in models}
    sanity:  Dict[str, Dict[str, List[bool]]]  = {m: {p: [] for p in piles} for m in models}
    all_rows = []

    # ── Process ───────────────────────────────────────────────────────────────
    for pile in piles:
        pile_dir = img_dir / pile
        if not pile_dir.exists():
            print(f"[skip] {pile_dir} not found"); continue

        imgs = sorted([p for p in pile_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
        if args.max_imgs > 0:
            imgs = imgs[:args.max_imgs]
        print(f"\n{'='*55}\n[pile] {pile} — {len(imgs)} images\n{'='*55}")

        for j, img_path in enumerate(imgs, 1):
            img_pil  = Image.open(img_path).convert("RGB")
            img_np   = np.array(img_pil)
            box_mask, box_pts = detect_white_box(img_np)

            # Depth inference + height map for each model
            for model_name in models:
                t_img = time.perf_counter()

                if model_name == "dinov3" and dinov3_model is not None:
                    depth = infer_dinov3(dinov3_model, img_pil, device)
                elif model_name == "dav2" and dav2_model is not None:
                    depth = infer_dav2(dav2_proc, dav2_model, img_pil, device)
                else:
                    continue

                out = compute_height_map(depth, CAMERA_HEIGHT_M, box_mask, img_np)
                hmap, scale, floor_d, sanity_ok, sanity_diff = out

                vol_L     = volume_from_height(hmap, box_mask, BOX_AREA_M2)
                obj_mask  = hmap > 0
                fill_pct  = float(np.mean(obj_mask[box_mask])*100) if box_mask.any() else 0.0
                mean_h_cm = float(np.mean(hmap[obj_mask])*100) if obj_mask.any() else 0.0
                max_h_cm  = float(np.max(hmap)*100)
                elapsed   = time.perf_counter() - t_img

                gt_mass = GT_MASSES.get(pile, 0.0)
                cf       = GT_VOLUME_L / vol_L if vol_L > 1e-3 else float("nan")
                density  = gt_mass / vol_L if vol_L > 1e-3 else float("nan")

                results[model_name][pile].append(vol_L)
                sanity[model_name][pile].append(sanity_ok)

                row = {
                    "model": model_name, "pile": pile, "image": img_path.name,
                    "vol_L": round(vol_L, 4), "gt_vol_L": GT_VOLUME_L,
                    "calib_factor": round(cf, 3) if not np.isnan(cf) else "nan",
                    "fill_pct": round(fill_pct, 1),
                    "mean_h_cm": round(mean_h_cm, 2),
                    "max_h_cm": round(max_h_cm, 2),
                    "depth_scale": round(scale, 4),
                    "floor_depth_raw": round(floor_d, 4),
                    "depth_sanity_ok": sanity_ok,
                    "sanity_diff_cm": round(sanity_diff, 2),
                    "gt_mass_kg": gt_mass,
                    "pred_density_kg_L": round(density, 4) if not np.isnan(density) else "nan",
                    "runtime_s": round(elapsed, 2),
                }
                all_rows.append(row)
                print(f"  [{model_name}][{j}/{len(imgs)}] {img_path.name}  "
                      f"vol={vol_L:.2f}L  CF={cf:.2f}  fill={fill_pct:.0f}%  "
                      f"max_h={max_h_cm:.1f}cm  sanity={'✓' if sanity_ok else '✗'}({sanity_diff:.1f}cm)  "
                      f"t={elapsed:.1f}s")

                # Save overlays (only first image per pile per model to save disk)
                if j == 1:
                    overlay_img = make_overlay(img_np, hmap, box_pts)
                    depth_img   = make_depth_vis(depth, box_pts, img_np)
                    Image.fromarray(overlay_img).save(vis_dir/f"{model_name}_{pile}_{img_path.stem}_overlay.jpg", quality=85)
                    Image.fromarray(depth_img).save(vis_dir/f"{model_name}_{pile}_{img_path.stem}_depth.jpg", quality=85)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Per-pile summary & calibration factors ────────────────────────────────
    print("\n" + "="*55)
    print("CALIBRATION FACTOR ANALYSIS")
    print("="*55)

    calib_summary = {}
    for model_name in models:
        calib_summary[model_name] = {}
        all_cf = []
        print(f"\n  Model: {model_name.upper()}")
        for pile in piles:
            vols = results[model_name].get(pile, [])
            if not vols:
                continue
            mv     = float(np.mean(vols))
            sv     = float(np.std(vols))
            cv_pct = float(sv/mv*100) if mv>0 else 0
            cf     = GT_VOLUME_L / mv if mv > 1e-3 else float("nan")
            all_cf.append(cf if not np.isnan(cf) else 0)

            san_pct = float(np.mean(sanity[model_name][pile])*100)
            gt_mass = GT_MASSES.get(pile, 0.0)
            density = gt_mass / mv if mv > 1e-3 else float("nan")

            calib_summary[model_name][pile] = {
                "n_imgs": len(vols), "mean_vol_L": round(mv,3),
                "std_vol_L": round(sv,3), "cv_pct": round(cv_pct,1),
                "gt_vol_L": GT_VOLUME_L, "calib_factor": round(cf,3) if not np.isnan(cf) else "nan",
                "pred_density_kg_L": round(density,3) if not np.isnan(density) else "nan",
                "depth_sanity_pct": round(san_pct,0),
            }
            print(f"    {pile}: pred={mv:.2f}L  CF={cf:.2f}  CV={cv_pct:.1f}%  "
                  f"density={density:.3f}kg/L  sanity={san_pct:.0f}%")

        valid_cfs = [c for c in all_cf if c > 0]
        if valid_cfs:
            mean_cf = float(np.mean(valid_cfs))
            std_cf  = float(np.std(valid_cfs))
            calib_summary[model_name]["global_calib_factor"] = round(mean_cf, 3)
            calib_summary[model_name]["global_cf_std"]       = round(std_cf, 3)
            print(f"    GLOBAL CF: {mean_cf:.3f} ± {std_cf:.3f}  "
                  f"(multiply predictions by {mean_cf:.2f} to get ~GT)")

    # ── Depth sanity report ───────────────────────────────────────────────────
    print("\n" + "="*55)
    print("DEPTH SANITY CHECK (pile closer to camera than floor?)")
    print("="*55)
    for model_name in models:
        for pile in piles:
            s = sanity[model_name].get(pile, [])
            if s:
                pct = float(np.mean(s)*100)
                print(f"  [{model_name}][{pile}]: {pct:.0f}% of images pass (pile shallower than floor)")
    print("\n  → If >80%: depth shape is correct, only scale calibration needed")
    print("  → If <50%: depth model is not detecting the pile correctly")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if all_rows:
        csv_path = out_dir / "per_image_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader(); w.writerows(all_rows)

    summary_path = out_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "setup": {
                "camera_height_m": CAMERA_HEIGHT_M,
                "box_area_m2": BOX_AREA_M2,
                "gt_volume_L_per_pile": GT_VOLUME_L,
                "gt_masses": GT_MASSES,
            },
            "models": calib_summary,
        }, f, indent=2)

    # ── Comparison plot ───────────────────────────────────────────────────────
    if len(models) >= 1:
        fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5), squeeze=False)
        colors = {"dinov3": "steelblue", "dav2": "darkorange"}
        for k, model_name in enumerate(models):
            ax = axes[0][k]
            for pile in piles:
                vols = results[model_name].get(pile, [])
                if not vols: continue
                ax.plot(vols, marker="o", label=pile)
            ax.axhline(GT_VOLUME_L, color="red", linestyle="--", lw=2, label=f"GT={GT_VOLUME_L}L")
            gcf_val = calib_summary.get(model_name, {}).get("global_calib_factor", "?")
            ax.set_title(f"{model_name.upper()}  | global CF={gcf_val}")
            ax.set_xlabel("Image index"); ax.set_ylabel("Predicted Volume (L)")
            ax.legend(fontsize=8); ax.grid(True, alpha=.3)
        plt.suptitle(f"IITB Volume Estimates (camera={CAMERA_HEIGHT_M}m, box={BOX_AREA_M2:.2f}m²)", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir/"calibration_chart.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nAll results saved to: {out_dir}/")
    print(f"  per_image_results.csv")
    print(f"  calibration_summary.json")
    print(f"  calibration_chart.png")
    print(f"  vis/ (depth maps + overlays for first image of each pile)")
    print(f"\nVIS FILES TO INSPECT (in eval_results/iitb_calibration/vis/):")
    for model_name in models:
        for pile in piles:
            pile_dir = img_dir / pile
            if not pile_dir.exists(): continue
            imgs = sorted([p for p in pile_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
            if imgs:
                stem = imgs[0].stem
                print(f"  {model_name}_{pile}_{stem}_depth.jpg     ← depth map")
                print(f"  {model_name}_{pile}_{stem}_overlay.jpg   ← height heatmap on RGB")


if __name__ == "__main__":
    main()
