#!/usr/bin/env python3
"""
Full-frame 3-D point cloud + volume estimation for IITB piles.

No segmentation model, no text prompts.  The floor is fitted from the deepest
1% of pixels; every pixel above the 1 cm height threshold becomes a cloud point.

Two depth backends compared side-by-side:
  - DepthAnythingV2-Metric-Indoor-Large  (DAv2)
  - DINOv3 vit7b16_dd                   (DINOv3)

Pipeline per image:
  1. Depth inference  (DAv2 or DINOv3)
  2. Floor-plane fitting on deepest-1% pixels  -> scale depth -> height map
  3. Threshold at 1 cm  -> active pixel mask
  4. Volume via three methods:
       a. Height integration  (Riemann sum,  exact for the overhead geometry)
       b. Voxel grid  (5 mm voxels, fills columns z=0..h for each active pixel)
       c. Convex hull  (scipy, upper bound)
  5. Save coloured ASCII .ply  (open in MeshLab / CloudCompare)
  6. Calibration: first CALIB_N images per pile -> CF; remaining -> evaluation

Usage:
    python evaluation/run_iitb_pointcloud.py --device cuda:1
    python evaluation/run_iitb_pointcloud.py --device cuda:1 --model dinov3
    python evaluation/run_iitb_pointcloud.py --device cuda:1 --model both
"""

import argparse, csv, json, os, sys, time, warnings
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
from torchvision import transforms

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Physical constants ─────────────────────────────────────────────────────────
BOX_W_M         = 0.60
BOX_H_M         = 0.60
BOX_AREA_M2     = BOX_W_M * BOX_H_M
CAMERA_HEIGHT_M = 0.65
GT_VOLUME_L     = 18.0
HEIGHT_THRESH_M = 0.01          # 1 cm noise floor
FLOOR_PCT       = 99.0          # deepest N% pixels = floor candidates
VOXEL_SIZE_M    = 0.005         # 5 mm voxels
CALIB_N         = 3             # first N images per pile used for calibration

DAV2_MODEL_ID   = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
DEPTH_RANGE_DINOV3 = (0.30, 0.90)   # tuned for 65 cm camera height

IITB_ROOT = Path(__file__).resolve().parents[1] / "solid_waste_dataset2_iitb_site_pho"
PILES = {
    "pile1": {"mass_kg": 0.670, "gt_vol_L": GT_VOLUME_L},
    "pile2": {"mass_kg": 0.480, "gt_vol_L": GT_VOLUME_L},
    "pile3": {"mass_kg": 0.660, "gt_vol_L": GT_VOLUME_L},
}

# ── Model loaders ──────────────────────────────────────────────────────────────

def load_dav2(device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    t0 = time.time()
    print(f"[dav2] loading {DAV2_MODEL_ID} ...")
    proc  = AutoImageProcessor.from_pretrained(DAV2_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DAV2_MODEL_ID).eval().to(device)
    print(f"[dav2] ready in {time.time()-t0:.1f}s")
    return proc, model


@torch.inference_mode()
def infer_dav2(proc, model, img_pil, device):
    h, w  = img_pil.height, img_pil.width
    inp   = proc(images=img_pil, return_tensors="pt")
    inp   = {k: v.to(device) for k, v in inp.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(**inp)
    depth = F.interpolate(out.predicted_depth[None], size=(h, w),
                          mode="bilinear", align_corners=False)[0, 0]
    return depth.float().cpu().numpy()


def load_dinov3(device):
    kw  = dict(pretrained=False,
               weights=os.getenv("DINOV3_DEPTHER_WEIGHTS", "SYNTHMIX"),
               backbone_weights=os.getenv("DINOV3_BACKBONE_WEIGHTS", "LVD1689M"),
               depth_range=DEPTH_RANGE_DINOV3)
    repo = os.getenv("DINOV3_REPO_DIR", "")
    src  = repo if repo else os.getenv("DINOV3_GITHUB_REPO", "facebookresearch/dinov3")
    mode = "local" if repo else "github"
    t0   = time.time()
    print(f"[dinov3] loading from {src} ...")
    # Patch .cuda() so hub code doesn't force GPU 0 during init
    _mc = torch.nn.Module.cuda;  _tc = torch.Tensor.cuda
    torch.nn.Module.cuda = lambda self, d=None: self
    torch.Tensor.cuda    = lambda self, d=None, **kw2: self
    try:
        m = torch.hub.load(src, "dinov3_vit7b16_dd", source=mode, trust_repo=True, **kw)
    finally:
        torch.nn.Module.cuda = _mc;  torch.Tensor.cuda = _tc
    m = m.eval().to(device)
    print(f"[dinov3] ready in {time.time()-t0:.1f}s")
    return m


_dinov3_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((518, 518), antialias=True),
    transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
])

@torch.inference_mode()
def infer_dinov3(model, img_pil, device):
    h, w = img_pil.height, img_pil.width
    x    = _dinov3_tf(img_pil)[None].to(device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        d = model(x)
    d = F.interpolate(d.float(), (h, w), mode="bilinear", align_corners=False)
    return d[0, 0].cpu().numpy()


# ── Geometry helpers ────────────────────────────────────────────────────────────

def fit_floor_and_height(depth_map: np.ndarray, camera_h: float):
    """
    Least-squares plane fit to deepest-FLOOR_PCT percentile.
    Returns (height_map_metres, scale_factor).
    """
    h, w   = depth_map.shape
    valid  = np.isfinite(depth_map) & (depth_map > 0)
    dv     = depth_map[valid]
    thr    = np.percentile(dv, FLOOR_PCT)
    fmask  = valid & (depth_map >= thr)

    rows, cols = np.where(fmask)
    d_f        = depth_map[rows, cols]
    A          = np.c_[cols, rows, np.ones(len(cols))]
    coeffs, _, _, _ = np.linalg.lstsq(A, d_f, rcond=None)

    cc, rr    = np.meshgrid(np.arange(w), np.arange(h))
    floor_pred = coeffs[0]*cc + coeffs[1]*rr + coeffs[2]

    scale    = camera_h / float(np.median(d_f))
    height   = np.clip((floor_pred - depth_map) * scale, 0, None)
    return height, scale


def height_to_cloud(height_map: np.ndarray):
    """
    Convert full-frame height map to (N,3) point cloud in physical metres.
    Active pixels: height > HEIGHT_THRESH_M.
    Returns pts (N,3), rows_active, cols_active, clean height map.
    """
    h, w    = height_map.shape
    hclean  = np.where(height_map > HEIGHT_THRESH_M,
                       height_map - HEIGHT_THRESH_M, 0.0)
    active  = hclean > 0
    rows_a, cols_a = np.where(active)
    x = cols_a * (BOX_W_M / w)
    y = rows_a * (BOX_H_M / h)
    z = hclean[rows_a, cols_a]
    pts = np.column_stack([x, y, z])
    return pts, rows_a, cols_a, hclean


def compute_volumes(pts: np.ndarray, hclean: np.ndarray):
    """Three volume methods; returns dict."""
    from scipy.spatial import ConvexHull, QhullError

    h, w    = hclean.shape
    px_area = BOX_AREA_M2 / float(h * w)

    # a: height integration (Riemann sum -- exact for the overhead geometry)
    vol_hint = float(np.sum(hclean)) * px_area * 1000.0

    # b: voxel grid -- fast numpy implementation
    #    For each active (x,y) pixel column, all voxels from z=0 up to z_top
    #    are occupied.  Instead of a Python set loop (O(N * z_top)), we:
    #      1. compute (vox_x, vox_y, vox_z_top) for every point
    #      2. pack (vox_x, vox_y) into a single key
    #      3. np.maximum.at to get max z_top per (x,y) column  -- O(N) numpy
    #      4. total occupied = sum(max_z_top + 1) * voxel_size^3
    vol_vox = 0.0
    if len(pts) >= 4:
        vx = (pts[:, 0] / VOXEL_SIZE_M).astype(np.int32)
        vy = (pts[:, 1] / VOXEL_SIZE_M).astype(np.int32)
        vz = (pts[:, 2] / VOXEL_SIZE_M).astype(np.int32)
        # pack into 1-D key (vx and vy are at most 120 for a 0.6m / 0.005m grid)
        key = vx.astype(np.int64) * 100_000 + vy
        unique_keys, inv = np.unique(key, return_inverse=True)
        max_vz = np.zeros(len(unique_keys), dtype=np.int32)
        np.maximum.at(max_vz, inv, vz)
        vol_vox = float(np.sum(max_vz + 1)) * (VOXEL_SIZE_M ** 3) * 1000.0

    # c: convex hull of a subsample (closed with floor points)
    vol_hull = 0.0
    if len(pts) >= 10:
        n_sub = min(len(pts), 5000)
        idx_s = np.random.choice(len(pts), n_sub, replace=False)
        sub   = pts[idx_s]
        fp    = sub.copy(); fp[:, 2] = 0.0
        try:
            vol_hull = ConvexHull(np.vstack([sub, fp])).volume * 1000.0
        except QhullError:
            pass

    z_vals = pts[:, 2] if len(pts) else np.array([0.0])
    return {
        "height_int_L":  round(vol_hint, 3),
        "voxel_L":       round(vol_vox,  3),
        "convex_hull_L": round(vol_hull, 3),
        "n_points":      len(pts),
        "fill_pct":      round(float((hclean > 0).sum()) / hclean.size * 100, 1),
        "mean_h_cm":     round(float(z_vals.mean()) * 100, 2),
        "max_h_cm":      round(float(z_vals.max())  * 100, 2),
    }


def sanity_check(depth_map: np.ndarray):
    h, w  = depth_map.shape
    he, we = int(0.15 * h), int(0.15 * w)
    c  = float(np.mean(depth_map[he:h-he, we:w-we]))
    edges = np.concatenate([
        depth_map[:he, :].ravel(), depth_map[h-he:, :].ravel(),
        depth_map[:, :we].ravel(), depth_map[:, w-we:].ravel(),
    ])
    e = float(np.mean(edges))
    return c < e, round((e - c) * 100, 1)


# ── PLY / vis helpers ─────────────────────────────────────────────────────────

PLY_MAX_PTS = 500_000    # subsample PLY to keep file sizes reasonable

def save_ply(pts: np.ndarray, img_np: np.ndarray,
             rows_a: np.ndarray, cols_a: np.ndarray, path: str):
    if len(pts) == 0:
        return
    # subsample if needed
    if len(pts) > PLY_MAX_PTS:
        idx = np.random.choice(len(pts), PLY_MAX_PTS, replace=False)
        pts    = pts[idx]
        rows_a = rows_a[idx]
        cols_a = cols_a[idx]

    h_img, w_img = img_np.shape[:2]
    r = np.clip(rows_a, 0, h_img - 1)
    c = np.clip(cols_a, 0, w_img - 1)
    col = img_np[r, c]
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for i in range(len(pts)):
            x, y, z = pts[i]
            ri, gi, bi = col[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(ri)} {int(gi)} {int(bi)}\n")


def save_vis(img_pil, depth_map, hclean, out_path, title):
    img_np = np.array(img_pil)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=11)

    axes[0].imshow(img_np); axes[0].set_title("RGB"); axes[0].axis("off")

    d_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    axes[1].imshow(1 - d_norm, cmap="inferno"); axes[1].set_title("Depth (bright=near)"); axes[1].axis("off")

    hmax   = hclean.max() if hclean.max() > 0 else 1.0
    heat   = cm.hot(hclean / hmax)[:, :, :3]
    blend  = np.clip(0.55 * img_np / 255.0 + 0.45 * heat, 0, 1)
    axes[2].imshow(blend); axes[2].set_title("Height heatmap"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


# ── Per-model evaluation ───────────────────────────────────────────────────────

def run_model(model_tag: str, infer_fn, device, out_root: Path):
    vis_dir = out_root / model_tag / "vis"
    ply_dir = out_root / model_tag / "point_clouds"
    vis_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    rows_all    = []
    pile_calib  = {p: [] for p in PILES}
    pile_eval   = {p: [] for p in PILES}

    for pile_name, pile_info in PILES.items():
        pile_dir  = IITB_ROOT / pile_name
        img_paths = sorted(pile_dir.glob("*.jpg"))
        gt_vol    = pile_info["gt_vol_L"]
        mass_kg   = pile_info["mass_kg"]

        print(f"\n  [{model_tag}] {pile_name}  mass={mass_kg}kg  GT={gt_vol}L")

        for i, img_path in enumerate(img_paths):
            t0      = time.time()
            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)
            fname   = img_path.name

            # Resize to 1024px long edge for point-cloud work.
            # Depth models resize internally anyway; this just keeps
            # the output cloud at a manageable density.
            ratio      = 1024 / max(img_pil.width, img_pil.height)
            cloud_pil  = img_pil.resize(
                (int(img_pil.width * ratio), int(img_pil.height * ratio)),
                Image.LANCZOS)

            depth_map          = infer_fn(cloud_pil)
            height_raw, scale  = fit_floor_and_height(depth_map, CAMERA_HEIGHT_M)
            pts, rows_a, cols_a, hclean = height_to_cloud(height_raw)
            vols               = compute_volumes(pts, hclean)
            ok, diff_cm        = sanity_check(depth_map)
            elapsed            = time.time() - t0
            is_calib           = (i < CALIB_N)

            # save PLY + vis (first 2 per pile)
            ply_path = ply_dir / f"{pile_name}_{img_path.stem}.ply"
            save_ply(pts, img_np, rows_a, cols_a, str(ply_path))
            if i < 2:
                vis_path = vis_dir / f"{pile_name}_{img_path.stem}.jpg"
                save_vis(cloud_pil, depth_map, hclean, str(vis_path),
                         f"{model_tag} | {pile_name}/{fname} | "
                         f"hint={vols['height_int_L']:.1f}L | "
                         f"{'CALIB' if is_calib else 'EVAL'}")

            tag = "[C]" if is_calib else "[E]"
            print(f"    {tag} {fname:20s} "
                  f"hint={vols['height_int_L']:6.2f}L  "
                  f"vox={vols['voxel_L']:6.2f}L  "
                  f"hull={vols['convex_hull_L']:6.2f}L  "
                  f"fill={vols['fill_pct']:4.1f}%  "
                  f"max_h={vols['max_h_cm']:4.1f}cm  "
                  f"sanity={'ok' if ok else 'FAIL'}({diff_cm}cm)  "
                  f"t={elapsed:.1f}s", flush=True)

            row = {
                "model": model_tag, "pile": pile_name, "image": fname,
                "is_calib": is_calib, "scale": round(scale, 4),
                **vols,
                "sanity_ok": ok, "sanity_diff_cm": diff_cm,
                "gt_vol_L": gt_vol, "gt_mass_kg": mass_kg,
                "runtime_s": round(elapsed, 1),
            }
            rows_all.append(row)

            if is_calib:
                pile_calib[pile_name].append(vols["height_int_L"])
            else:
                pile_eval[pile_name].append((fname, vols))

    # calibration + evaluation
    print(f"\n  {'='*55}")
    print(f"  {model_tag.upper()}  --  CALIBRATION + EVALUATION")
    print(f"  {'='*55}")
    pile_summary = {}
    for pile_name, pile_info in PILES.items():
        gt_vol   = pile_info["gt_vol_L"]
        c_vols   = pile_calib[pile_name]
        e_data   = pile_eval[pile_name]
        if not c_vols:
            continue
        mean_c = float(np.mean(c_vols))
        cf     = gt_vol / mean_c if mean_c > 0 else float("nan")

        e_raw  = [v["height_int_L"] for _, v in e_data]
        e_cal  = [v * cf for v in e_raw]
        mean_e = float(np.mean(e_cal)) if e_cal else float("nan")
        err    = abs(mean_e - gt_vol) / gt_vol * 100 if e_cal else float("nan")

        print(f"\n  [{pile_name}]  calib_mean_raw={mean_c:.2f}L  CF={cf:.3f}")
        print(f"    eval ({len(e_cal)} imgs): mean_cal={mean_e:.2f}L  error={err:.1f}%")

        for row in rows_all:
            if row["model"] == model_tag and row["pile"] == pile_name:
                row["cf"]               = round(cf, 4)
                row["height_int_cal_L"] = round(row["height_int_L"] * cf, 3)
                row["voxel_cal_L"]      = round(row["voxel_L"]      * cf, 3)

        pile_summary[pile_name] = {
            "n_calib": len(c_vols), "n_eval": len(e_cal),
            "mean_calib_raw_L": round(mean_c, 3),
            "cf": round(cf, 4),
            "mean_eval_cal_L": round(mean_e, 3),
            "eval_error_pct": round(err, 1),
            "gt_vol_L": gt_vol, "gt_mass_kg": pile_info["mass_kg"],
        }

    return rows_all, pile_summary


# ── Chart helper ──────────────────────────────────────────────────────────────

def make_comparison_chart(all_rows, summaries, out_path):
    models = sorted(set(r["model"] for r in all_rows))
    pile_names = list(PILES.keys())
    n_piles = len(pile_names)
    fig, axes = plt.subplots(1, n_piles, figsize=(6 * n_piles, 5), sharey=False)
    if n_piles == 1:
        axes = [axes]

    colors = {"dav2": "#2196F3", "dinov3": "#FF9800"}
    for ax, pname in zip(axes, pile_names):
        gt = PILES[pname]["gt_vol_L"]
        ax.axhline(gt, color="red", linestyle="--", linewidth=1.8, label=f"GT={gt}L")
        for mdl in models:
            mdl_rows = [r for r in all_rows
                        if r["pile"] == pname and r["model"] == mdl
                        and not r.get("is_calib", False)]
            cf = summaries.get(mdl, {}).get(pname, {}).get("cf", 1.0)
            x  = np.arange(len(mdl_rows))
            offset = (models.index(mdl) - len(models) / 2) * 0.3 + 0.15
            vals = [r["height_int_L"] * cf for r in mdl_rows]
            ax.bar(x + offset, vals, 0.28, label=f"{mdl} (cal.)",
                   color=colors.get(mdl, "#666"), alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [r["image"].replace("IMG_", "").replace(".jpg", "")
                 for r in mdl_rows],
                rotation=45, ha="right", fontsize=7)
        ax.set_title(pname); ax.set_ylabel("Volume (L)"); ax.legend(fontsize=7)
        ax.set_ylim(0, gt * 3)

    plt.suptitle("Full-Frame 3D Point Cloud -- Calibrated Volume (eval partition)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[chart] {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--model",  default="both",
                    choices=["dav2", "dinov3", "both"],
                    help="which depth model(s) to run")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_root = Path("evaluation/iitb_pointcloud")
    out_root.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Full-Frame 3-D Point Cloud -- IITB piles")
    print(f"  camera_h={CAMERA_HEIGHT_M}m  box={BOX_W_M}x{BOX_H_M}m  "
          f"thresh={HEIGHT_THRESH_M*100:.0f}cm  voxel={VOXEL_SIZE_M*100:.0f}mm")
    print(f"  calib split: first {CALIB_N} images per pile")
    print("="*60)

    run_models = (["dav2", "dinov3"] if args.model == "both"
                  else [args.model])

    all_rows_combined = []
    all_summaries     = {}   # model -> pile -> summary

    for mdl_tag in run_models:
        if mdl_tag == "dav2":
            proc, model = load_dav2(device)
            infer_fn = lambda img, _p=proc, _m=model, _d=device: infer_dav2(_p, _m, img, _d)
        else:
            model    = load_dinov3(device)
            infer_fn = lambda img, _m=model, _d=device: infer_dinov3(_m, img, _d)

        rows, summary = run_model(mdl_tag, infer_fn, device, out_root)
        all_rows_combined.extend(rows)
        all_summaries[mdl_tag] = summary

        # free model memory before loading next
        del model; torch.cuda.empty_cache()

    # ── save combined CSV
    csv_path = out_root / "per_image_results.csv"
    if all_rows_combined:
        fnames = list(all_rows_combined[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fnames)
            w.writeheader(); w.writerows(all_rows_combined)

    # ── save JSON summary
    with open(out_root / "summary.json", "w") as f:
        json.dump({"setup": {
            "camera_height_m": CAMERA_HEIGHT_M, "box_area_m2": BOX_AREA_M2,
            "gt_volume_L": GT_VOLUME_L, "height_thresh_m": HEIGHT_THRESH_M,
            "voxel_size_m": VOXEL_SIZE_M, "calib_n_per_pile": CALIB_N,
        }, "models": all_summaries}, f, indent=2)

    # ── comparison chart
    make_comparison_chart(all_rows_combined, all_summaries,
                          out_root / "comparison_chart.png")

    print(f"\nOutputs in {out_root}/")
    print(f"  dav2/point_clouds/*.ply    -- open in MeshLab / CloudCompare")
    print(f"  dinov3/point_clouds/*.ply")
    print(f"  comparison_chart.png")
    print(f"  per_image_results.csv")


if __name__ == "__main__":
    main()
