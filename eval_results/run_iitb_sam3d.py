#!/usr/bin/env python3
"""
SAM-3D volume estimation for IITB white-plastic waste piles.

Pipeline per image:
  1. LangSAM (SAM2.1 + GroundingDINO) -- segment the pile by text prompt
  2. DepthAnythingV2-Metric-Indoor-Large -- metric depth map
  3. Back-project segmented pixels to 3D using the known camera height:
       X = col * (box_width / img_width)
       Y = row * (box_height / img_height)
       Z = height_above_floor  (from floor-plane fitting on the depth map)
  4. Volume estimation from the point cloud via three methods:
       a. Height integration (SAM-masked Riemann sum) -- fast, exact
       b. Voxel-grid (discretise XYZ into 5mm voxels, count occupied)
       c. Convex hull  (scipy.spatial.ConvexHull)
  5. Save the 3D point cloud as an ASCII .ply file (open in MeshLab / CloudCompare)
  6. Per-pile calibration factor derived from the calibration partition,
     applied blindly to the held-out evaluation partition.

Usage:
    python eval_results/run_iitb_sam3d.py --device cuda:1
    python eval_results/run_iitb_sam3d.py --device cuda:1 --prompt "plastic waste"
"""

import argparse, csv, json, os, sys, time, warnings
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from scipy.spatial import ConvexHull, QhullError

warnings.filterwarnings("ignore")

# ── project root on path (for lang_sam) ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── constants ─────────────────────────────────────────────────────────────────
BOX_W_M         = 0.60          # 60 cm interior width
BOX_H_M         = 0.60          # 60 cm interior height
BOX_AREA_M2     = BOX_W_M * BOX_H_M
CAMERA_HEIGHT_M = 0.65
GT_VOLUME_L     = 18.0
HEIGHT_THRESH_M = 0.01          # 1 cm noise floor
FLOOR_PCT       = 99.0          # deepest N% = floor candidates
VOXEL_SIZE_M    = 0.005         # 5 mm voxels

# calibration split: first N images of each pile used for CF derivation
CALIB_N         = 3             # 3 calibration images per pile

DAV2_MODEL_ID   = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

IITB_ROOT = Path(__file__).resolve().parents[1] / "solid_waste_dataset2_iitb_site_pho"
PILES = {
    "pile1": {"mass_kg": 0.670, "gt_vol_L": GT_VOLUME_L},
    "pile2": {"mass_kg": 0.480, "gt_vol_L": GT_VOLUME_L},
    "pile3": {"mass_kg": 0.660, "gt_vol_L": GT_VOLUME_L},
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_dav2(device: str):
    t0 = time.time()
    print("[model] loading DepthAnythingV2 ...")
    proc  = AutoImageProcessor.from_pretrained(DAV2_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DAV2_MODEL_ID)
    model.eval().to(device)
    print(f"[model] ready in {time.time()-t0:.1f}s")
    return proc, model

def load_langsam(device: str, prompt_override: str | None = None):
    t0 = time.time()
    print("[model] loading LangSAM (SAM2.1 + GroundingDINO) ...")
    from lang_sam import LangSAM
    lsam = LangSAM(device=device)
    print(f"[model] LangSAM ready in {time.time()-t0:.1f}s")
    return lsam

@torch.inference_mode()
def predict_depth(proc, model, img_pil: Image.Image, device: str) -> np.ndarray:
    """Return metric depth map in raw model units (metres-ish), same HxW as input."""
    h, w = img_pil.height, img_pil.width
    inputs = proc(images=img_pil, return_tensors="pt").to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(**inputs)
    depth = out.predicted_depth            # (1, H', W')
    depth = F.interpolate(depth[None], size=(h, w), mode="bilinear",
                          align_corners=False)[0, 0]
    return depth.float().cpu().numpy()


def segment_pile(lsam, img_pil: Image.Image, prompt: str,
                 box_thresh=0.25, text_thresh=0.20) -> np.ndarray | None:
    """
    Run LangSAM with 'prompt'.  Returns a binary uint8 mask (HxW) that is the
    union of all returned SAM masks, or None if nothing detected.
    Automatically retries with a looser threshold if no mask found.
    """
    h, w = img_pil.height, img_pil.width
    for bt, tt in [(box_thresh, text_thresh), (0.15, 0.10)]:
        results = lsam.predict([img_pil], [prompt],
                               box_threshold=bt, text_threshold=tt)
        masks = results[0].get("masks", [])
        if len(masks):
            union = np.zeros((h, w), dtype=bool)
            for m in masks:
                # mask can be (1,H,W) or (H,W)
                m2 = m[0] if m.ndim == 3 else m
                union |= m2.astype(bool)
            return union.astype(np.uint8)
    return None


def fit_floor_and_height(depth_map: np.ndarray,
                          camera_h: float) -> tuple[np.ndarray, float]:
    """
    Fit a tilted-plane floor to the deepest FLOOR_PCT percentile pixels.
    Returns:
        height_map : physical height above floor per pixel (metres, >=0)
        scale      : depth-unit -> metres conversion factor
    """
    h, w = depth_map.shape
    valid = np.isfinite(depth_map) & (depth_map > 0)
    dv    = depth_map[valid]

    # floor = deepest pixels
    thr        = np.percentile(dv, FLOOR_PCT)
    floor_mask = valid & (depth_map >= thr)

    # least-squares plane d = a*col + b*row + c
    rows, cols = np.where(floor_mask)
    d_floor    = depth_map[rows, cols]
    A          = np.c_[cols, rows, np.ones(len(cols))]
    coeffs, _, _, _ = np.linalg.lstsq(A, d_floor, rcond=None)

    # predicted floor at every pixel
    cc, rr = np.meshgrid(np.arange(w), np.arange(h))
    floor_pred = coeffs[0]*cc + coeffs[1]*rr + coeffs[2]

    # scale: median floor raw depth -> camera_h metres
    med_floor_raw = float(np.median(d_floor))
    scale         = camera_h / med_floor_raw

    # raw height above local floor (positive = above, i.e. closer to camera)
    height_raw = (floor_pred - depth_map) * scale
    height_raw = np.clip(height_raw, 0, None)
    return height_raw, scale


def compute_volume_methods(height_map: np.ndarray,
                            mask: np.ndarray | None,
                            box_area_m2: float) -> dict:
    """
    Given height_map (HxW, metres) and optional binary mask,
    compute volume by three methods.  Returns dict with keys:
      height_int_L, voxel_L, convex_hull_L, n_points, fill_pct, mean_h_cm, max_h_cm
    """
    h, w = height_map.shape

    # apply mask if provided
    if mask is not None:
        hmap = np.where(mask.astype(bool), height_map, 0.0)
    else:
        hmap = height_map.copy()

    # threshold noise
    hmap = np.where(hmap > HEIGHT_THRESH_M, hmap - HEIGHT_THRESH_M, 0.0)

    active = hmap > 0
    n_px   = int(active.sum())
    fill   = n_px / float(h * w) * 100.0

    # method a: height integration (Riemann sum)
    px_area  = box_area_m2 / float(h * w)
    vol_hint = float(np.sum(hmap)) * px_area * 1000.0   # litres

    # 3D point cloud in physical coordinates
    rows, cols = np.where(active)
    z_vals     = hmap[rows, cols]
    x_vals     = cols  * (BOX_W_M / w)
    y_vals     = rows  * (BOX_H_M / h)

    pts = np.column_stack([x_vals, y_vals, z_vals])   # (N, 3) metres

    # method b: voxel grid
    if len(pts) >= 4:
        vox_x = np.floor(pts[:, 0] / VOXEL_SIZE_M).astype(int)
        vox_y = np.floor(pts[:, 1] / VOXEL_SIZE_M).astype(int)
        # for each (vx,vy) column, fill from z=0 to z=height
        vox_z_max = np.floor(pts[:, 2] / VOXEL_SIZE_M).astype(int)
        occupied  = set()
        for vx, vy, vz_top in zip(vox_x, vox_y, vox_z_max):
            for vz in range(vz_top + 1):
                occupied.add((vx, vy, vz))
        voxel_vol_m3 = len(occupied) * (VOXEL_SIZE_M ** 3)
        vol_voxel    = voxel_vol_m3 * 1000.0
    else:
        vol_voxel = 0.0

    # method c: convex hull of surface points
    vol_hull = 0.0
    if len(pts) >= 10:
        # add floor points (z=0) to close the hull
        n_floor = min(len(pts), 200)
        idx_f   = np.random.choice(len(pts), n_floor, replace=False)
        floor_pts = pts[idx_f].copy(); floor_pts[:, 2] = 0.0
        hull_pts = np.vstack([pts, floor_pts])
        try:
            hull = ConvexHull(hull_pts)
            vol_hull = hull.volume * 1000.0
        except QhullError:
            vol_hull = 0.0

    return {
        "pts":          pts,
        "height_int_L": round(vol_hint, 3),
        "voxel_L":      round(vol_voxel, 3),
        "convex_hull_L":round(vol_hull, 3),
        "n_points":     n_px,
        "fill_pct":     round(fill, 1),
        "mean_h_cm":    round(float(z_vals.mean())*100 if len(z_vals) else 0, 2),
        "max_h_cm":     round(float(z_vals.max())*100  if len(z_vals) else 0, 2),
        "hmap":         hmap,
    }


def save_ply(pts: np.ndarray, img_rgb: np.ndarray,
             rows_active, cols_active, out_path: str):
    """Save a coloured ASCII PLY file from the pile surface point cloud."""
    n = len(pts)
    if n == 0:
        return
    # sample colour from original image at each point
    h_img, w_img = img_rgb.shape[:2]
    h_pts        = img_rgb.shape[0]  # same as pts rows range
    # rows_active and cols_active come from where(active)
    r_clip = np.clip(rows_active, 0, h_img-1)
    c_clip = np.clip(cols_active, 0, w_img-1)
    colours = img_rgb[r_clip, c_clip]   # (N, 3) uint8

    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = pts[i]
            r, g, b = colours[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def save_vis(img_pil: Image.Image, depth_map: np.ndarray, hmap: np.ndarray,
             mask_raw: np.ndarray | None, mask_used: np.ndarray,
             out_path: str, title: str):
    """4-panel visualisation: RGB | SAM mask | depth | height heatmap."""
    img_np  = np.array(img_pil)
    h, w    = depth_map.shape

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=12)

    # panel 1: RGB
    axes[0].imshow(img_np)
    axes[0].set_title("RGB input")
    axes[0].axis("off")

    # panel 2: SAM mask (or fallback note)
    if mask_raw is not None:
        overlay = img_np.copy()
        overlay[mask_raw.astype(bool)] = [255, 80, 0]
        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title("SAM mask (orange)")
    else:
        axes[1].imshow(img_np)
        axes[1].set_title("SAM: no detection\n(full-frame fallback)")
    axes[1].axis("off")

    # panel 3: depth map
    d_show = depth_map.copy()
    d_show = (d_show - d_show.min()) / (d_show.max() - d_show.min() + 1e-6)
    axes[2].imshow(1.0 - d_show, cmap="inferno")
    axes[2].set_title("Depth map (bright=close)")
    axes[2].axis("off")

    # panel 4: height heatmap
    hmax = hmap.max() if hmap.max() > 0 else 1.0
    h_show = hmap / hmax
    heatmap = cm.hot(h_show)[:, :, :3]
    blended = 0.55 * img_np / 255.0 + 0.45 * heatmap
    blended = np.clip(blended, 0, 1)
    axes[3].imshow(blended)
    axes[3].set_title("Height heatmap (red=high)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def sanity_check(depth_map: np.ndarray):
    h, w = depth_map.shape
    he   = int(0.15 * h)
    we   = int(0.15 * w)
    centre = depth_map[he:h-he, we:w-we]
    edges  = np.concatenate([
        depth_map[:he, :].ravel(),
        depth_map[h-he:, :].ravel(),
        depth_map[:, :we].ravel(),
        depth_map[:, w-we:].ravel(),
    ])
    mc, me = float(np.mean(centre)), float(np.mean(edges))
    return mc < me, round((me - mc) * 100, 1)   # ok, diff_cm (raw units x100)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = args.device
    prompt = args.prompt

    out_root  = Path("eval_results/sam3d_iitb")
    vis_dir   = out_root / "vis"
    ply_dir   = out_root / "point_clouds"
    for d in [out_root, vis_dir, ply_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # load models
    dav2_proc, dav2_model = load_dav2(device)
    lsam = load_langsam(device)

    print(f"\nSegmentation prompt: '{prompt}'")
    print(f"Calibration split: first {CALIB_N} images per pile")

    rows_all    = []
    pile_vols   = {p: [] for p in PILES}
    pile_calib  = {p: [] for p in PILES}

    for pile_name, pile_info in PILES.items():
        pile_dir  = IITB_ROOT / pile_name
        img_paths = sorted(pile_dir.glob("*.jpg"))
        n_imgs    = len(img_paths)
        gt_vol    = pile_info["gt_vol_L"]
        mass_kg   = pile_info["mass_kg"]

        print(f"\n{'':2}{'─'*50}")
        print(f"  [pile] {pile_name}  ({n_imgs} images,  mass={mass_kg}kg,  GT={gt_vol}L)")
        print(f"{'':2}{'─'*50}")

        for i, img_path in enumerate(img_paths):
            t0      = time.time()
            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)
            fname   = img_path.name

            # 1. segmentation
            mask_raw = segment_pile(lsam, img_pil, prompt)
            if mask_raw is not None:
                n_mask_px = int(mask_raw.sum())
                n_total   = mask_raw.size
                # if mask is tiny (<5% of frame) or huge (>97%): use full frame
                frac = n_mask_px / n_total
                if frac < 0.05 or frac > 0.97:
                    mask_used = None
                    mask_note = "ff"   # full-frame fallback
                else:
                    mask_used = mask_raw
                    mask_note = f"{frac*100:.0f}%"
            else:
                mask_used = None
                mask_note = "ff"

            # 2. depth
            depth_map = predict_depth(dav2_proc, dav2_model, img_pil, device)

            # 3. floor fit -> height map
            height_raw, scale = fit_floor_and_height(depth_map, CAMERA_HEIGHT_M)

            # 4. volume from point cloud (3 methods)
            res = compute_volume_methods(height_raw, mask_used, BOX_AREA_M2)

            # 5. sanity check
            ok, diff_cm = sanity_check(depth_map)

            elapsed = time.time() - t0

            # active pixels for PLY colour sampling
            active_pix = res["hmap"] > 0
            rows_a, cols_a = np.where(active_pix)

            # 6. save PLY
            ply_path = ply_dir / f"{pile_name}_{img_path.stem}.ply"
            save_ply(res["pts"], img_np, rows_a, cols_a, str(ply_path))

            # 7. save vis (first 2 images per pile + any with unusual mask)
            if i < 2 or (mask_used is None and mask_raw is not None):
                vis_path = vis_dir / f"{pile_name}_{img_path.stem}_vis.jpg"
                save_vis(img_pil, depth_map, res["hmap"],
                         mask_raw, mask_used, str(vis_path),
                         f"{pile_name} / {fname}  |  mask={mask_note}  "
                         f"|  vol={res['height_int_L']:.1f}L")

            is_calib = (i < CALIB_N)
            print(
                f"  [{i+1}/{n_imgs}] {fname:20s}  "
                f"hint={res['height_int_L']:5.1f}L  "
                f"vox={res['voxel_L']:5.1f}L  "
                f"hull={res['convex_hull_L']:5.1f}L  "
                f"mask={mask_note:5s}  "
                f"pts={res['n_points']:5d}  "
                f"max_h={res['max_h_cm']:4.1f}cm  "
                f"sanity={'ok' if ok else 'FAIL'}  "
                f"{'[CALIB]' if is_calib else '[EVAL] '}  "
                f"t={elapsed:.1f}s"
            )

            row = {
                "pile": pile_name, "image": fname,
                "is_calib": is_calib,
                "mask_note": mask_note,
                "n_points": res["n_points"],
                "height_int_L": res["height_int_L"],
                "voxel_L": res["voxel_L"],
                "convex_hull_L": res["convex_hull_L"],
                "fill_pct": res["fill_pct"],
                "mean_h_cm": res["mean_h_cm"],
                "max_h_cm": res["max_h_cm"],
                "sanity_ok": ok,
                "sanity_diff_cm": diff_cm,
                "gt_vol_L": gt_vol,
                "gt_mass_kg": mass_kg,
                "scale": round(scale, 4),
                "runtime_s": round(elapsed, 1),
            }
            rows_all.append(row)

            if is_calib:
                pile_calib[pile_name].append(res["height_int_L"])
            else:
                pile_vols[pile_name].append((fname, res["height_int_L"]))

    # ── calibration + evaluation ───────────────────────────────────────────────
    print("\n" + "="*62)
    print("CALIBRATION + EVALUATION")
    print("="*62)

    pile_summary = {}
    for pile_name, pile_info in PILES.items():
        gt_vol   = pile_info["gt_vol_L"]
        calib_vols = pile_calib[pile_name]
        eval_data  = pile_vols[pile_name]

        if not calib_vols:
            print(f"  {pile_name}: no calibration images")
            continue

        mean_calib = float(np.mean(calib_vols))
        cf         = gt_vol / mean_calib if mean_calib > 0 else float("nan")

        eval_vols_raw  = [v for _, v in eval_data]
        eval_vols_cal  = [v * cf for v in eval_vols_raw]
        mean_eval_cal  = float(np.mean(eval_vols_cal)) if eval_vols_cal else float("nan")
        err_pct        = abs(mean_eval_cal - gt_vol) / gt_vol * 100 if eval_vols_cal else float("nan")

        print(f"\n  [{pile_name}]  GT={gt_vol}L")
        print(f"    Calibration ({CALIB_N} imgs): mean_raw={mean_calib:.2f}L  CF={cf:.3f}")
        if eval_vols_cal:
            print(f"    Evaluation  ({len(eval_vols_cal)} imgs): mean_cal={mean_eval_cal:.2f}L  "
                  f"error={err_pct:.1f}%")

        # back-fill CF into rows
        for row in rows_all:
            if row["pile"] == pile_name:
                row["cf"] = round(cf, 4)
                row["height_int_cal_L"] = round(row["height_int_L"] * cf, 3)
                row["voxel_cal_L"]      = round(row["voxel_L"]      * cf, 3)

        pile_summary[pile_name] = {
            "n_calib": len(calib_vols),
            "n_eval":  len(eval_vols_cal),
            "mean_calib_raw_L": round(mean_calib, 3),
            "cf": round(cf, 4),
            "mean_eval_cal_L": round(mean_eval_cal, 3),
            "eval_error_pct": round(err_pct, 1),
            "gt_vol_L": gt_vol,
            "gt_mass_kg": pile_info["mass_kg"],
        }

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_root / "per_image_results.csv"
    if rows_all:
        fieldnames = list(rows_all[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_all)

    # ── save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "setup": {
            "camera_height_m": CAMERA_HEIGHT_M,
            "box_area_m2": BOX_AREA_M2,
            "gt_volume_L": GT_VOLUME_L,
            "height_thresh_m": HEIGHT_THRESH_M,
            "voxel_size_m": VOXEL_SIZE_M,
            "calib_n_per_pile": CALIB_N,
            "prompt": prompt,
            "depth_model": "DepthAnythingV2-Metric-Indoor-Large",
            "seg_model": "LangSAM (SAM2.1 + GroundingDINO)",
        },
        "piles": pile_summary,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── comparison chart ──────────────────────────────────────────────────────
    _make_chart(rows_all, pile_summary, out_root / "sam3d_chart.png")

    print(f"\n{'='*62}")
    print(f"Outputs: {out_root}/")
    print(f"  point_clouds/  -- .ply files (open in MeshLab / CloudCompare)")
    print(f"  vis/           -- 4-panel visualisations")
    print(f"  per_image_results.csv")
    print(f"  summary.json")
    print(f"{'='*62}")


def _make_chart(rows, pile_summary, out_path):
    piles  = [r for r in rows if not r.get("is_calib", False)]
    names  = sorted(set(r["pile"] for r in piles))
    n      = len(names)

    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colors = {"height_int_L": "#2196F3", "voxel_L": "#4CAF50", "convex_hull_L": "#FF9800"}
    labels = {"height_int_L": "Height int.", "voxel_L": "Voxel grid", "convex_hull_L": "Convex hull"}

    for ax, pname in zip(axes, names):
        pile_rows = [r for r in piles if r["pile"] == pname]
        if not pile_rows:
            continue
        x     = np.arange(len(pile_rows))
        width = 0.25

        cf = pile_summary.get(pname, {}).get("cf", 1.0)
        gt = pile_rows[0]["gt_vol_L"]

        for k_i, (key, col) in enumerate(colors.items()):
            vals = [r[key] * cf for r in pile_rows]
            ax.bar(x + (k_i - 1) * width, vals, width, label=labels[key],
                   color=col, alpha=0.8)

        ax.axhline(gt, color="red", linestyle="--", linewidth=1.5, label=f"GT={gt}L")
        ax.set_xticks(x)
        ax.set_xticklabels([r["image"].replace("IMG_", "").replace(".jpg","")
                            for r in pile_rows], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{pname}  (CF={cf:.2f})")
        ax.set_ylabel("Calibrated Volume (L)")
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(gt * 3, 5))

    plt.suptitle("SAM-3D Calibrated Volume Estimates (eval partition)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[chart] saved to {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--prompt", default="plastic waste",
                    help="GroundingDINO text prompt for pile segmentation")
    args = ap.parse_args()
    run(args)
