#!/usr/bin/env python3
"""
DepthAnythingV2-Metric-Indoor evaluation for solid waste volume estimation.

Key advantage over DINOv3:
  - 3x better depth range span at close range (45cm-1m)
  - Specifically calibrated for indoor metric depth
  - Open source on HuggingFace (no special hub loading needed)

Modes:
  --mode dataset1  : dataset_1 GT evaluation
  --mode iitb      : IITB solid waste piles

Usage:
    python eval_results/eval_dav2.py --mode dataset1 --device cuda:1
    python eval_results/eval_dav2.py --mode iitb --device cuda:1 \
        --box_area_m2 0.12 --camera_height_m 1.0
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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ── Config ─────────────────────────────────────────────────────────────────
CAMERA_HEIGHT_DS1 = 0.45   # dataset_1: 45 cm
SCENE_AREA_DS1    = 0.6096 * 0.6096

GROUND_PERCENTILE = 99.0
MIN_HEIGHT_M      = 0.003   # 3 mm

WHITE_S_MAX = 50
WHITE_V_MIN = 160
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


def load_dav2(device: torch.device) -> Tuple:
    """Load DepthAnythingV2 Metric Indoor Large."""
    model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    print(f"[model] loading {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    if device.type == "cuda":
        model = model.to(device)
    return processor, model


def predict_depth_dav2(
    processor, model, img_pil: Image.Image, device: torch.device
) -> np.ndarray:
    """Run DepthAnythingV2 and return metric depth map (H×W) in metres."""
    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        depth_t = outputs.predicted_depth  # (1, H_out, W_out)
        depth_up = torch.nn.functional.interpolate(
            depth_t.unsqueeze(1).float(),
            size=(img_pil.size[1], img_pil.size[0]),
            mode="bilinear", align_corners=False,
        )
    return depth_up[0, 0].cpu().numpy()


def fit_ground_plane(depth_map: np.ndarray, percentile: float = 99.0) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth_map.shape
    valid = depth_map > 0
    if not valid.any():
        return np.full_like(depth_map, 0.5), valid
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
    camera_height_m: float,
    img_rgb: Optional[np.ndarray] = None,
    box_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute height-above-floor map.
    Uses white floor pixels (if img_rgb provided) or deepest percentile for floor reference.
    Calibrates scale so floor depth = camera_height_m.
    """
    h, w = depth_map.shape
    valid = depth_map > 0
    if box_mask is not None:
        valid = valid & box_mask

    if not valid.any():
        return np.zeros((h, w), dtype=np.float32), 1.0, camera_height_m

    # Floor reference: white pixels inside box
    floor_mask = None
    if img_rgb is not None and box_mask is not None:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        white = (hsv[:, :, 1] < WHITE_S_MAX) & (hsv[:, :, 2] > WHITE_V_MIN)
        wf = white & box_mask & valid
        if wf.sum() > 0.02 * (box_mask.sum() if box_mask is not None else h*w):
            floor_mask = wf

    if floor_mask is None:
        # Fall back to deepest percentile within valid region
        thr = np.percentile(depth_map[valid], GROUND_PERCENTILE)
        floor_mask = valid & (depth_map >= thr)

    plane, _ = fit_ground_plane(depth_map, GROUND_PERCENTILE) if floor_mask is None else (None, None)
    if plane is None:
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
    if box_mask is not None:
        height_raw[~box_mask] = 0.0

    floor_res = height_raw[floor_mask]
    fm = float(np.mean(floor_res)) if floor_res.size else 0.0
    fs = float(np.std(floor_res)) if floor_res.size else 0.0
    thr_h = max(MIN_HEIGHT_M, fm + 2.5 * fs)

    height_map = np.where(height_raw > thr_h, height_raw - thr_h, 0.0)
    if box_mask is not None:
        height_map = np.where(box_mask, height_map, 0.0)
    return height_map.astype(np.float32), scale, med_floor


def detect_white_box(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Auto-detect white box region."""
    h, w = img_rgb.shape[:2]
    full_mask = np.ones((h, w), dtype=bool)
    full_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)

    scale = 0.1
    sh, sw = max(1, int(h*scale)), max(1, int(w*scale))
    small = cv2.resize(img_rgb, (sw, sh))
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    wm = ((hsv[:,:,1] < WHITE_S_MAX) & (hsv[:,:,2] > WHITE_V_MIN)).astype(np.uint8) * 255
    kern = np.ones((5,5), np.uint8)
    wm = cv2.morphologyEx(wm, cv2.MORPH_CLOSE, kern, iterations=3)
    wm = cv2.morphologyEx(wm, cv2.MORPH_OPEN, kern, iterations=2)

    cnts, _ = cv2.findContours(wm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return full_mask, full_pts

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) / (sh * sw) < 0.05:
        return full_mask, full_pts

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box_full = (box / scale).astype(np.int32)
    bm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(bm, [box_full], 255)
    if bm.mean() < 0.05 * 255 * 0.5:
        return full_mask, full_pts
    return bm.astype(bool), box_full.astype(np.float32)


def compute_volume(height_map: np.ndarray, area_m2: float, mask: Optional[np.ndarray] = None) -> float:
    """Volume from height map. area_m2 = total scene area (divided by total pixels for pixel_area)."""
    h, w = height_map.shape
    px_area = area_m2 / float(h * w)
    return float(np.sum(height_map) * px_area)


def regression_metrics(y_true, y_pred):
    yt, yp = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    err = yp - yt
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    nz   = np.abs(yt) > 1e-9
    mape = float(np.mean(np.abs(err[nz]/yt[nz]))*100) if nz.any() else float("nan")
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt-np.mean(yt))**2))
    r2 = float(1-ss_res/ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"mae_L": mae, "rmse_L": rmse, "mape_pct": mape, "r2": r2,
            "mean_gt": float(np.mean(yt)), "mean_pred": float(np.mean(yp))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",            choices=["dataset1", "iitb"], default="dataset1")
    ap.add_argument("--device",          default="cuda:1")
    ap.add_argument("--out_dir",         default=None)
    ap.add_argument("--csv",             default="dataset_1/eval_gt_volume_liters.csv")
    ap.add_argument("--img_dir",         default="dataset_1/jpg_by_object")
    ap.add_argument("--iitb_dir",        default="solid_waste_dataset2_iitb_site_pho")
    ap.add_argument("--piles",           default="pile1,pile2,pile3")
    ap.add_argument("--box_area_m2",     type=float, default=0.12)
    ap.add_argument("--camera_height_m", type=float, default=1.0)
    ap.add_argument("--gt_json",         default="eval_results/iitb_gt.json")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = f"eval_results/dav2_{args.mode}"
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"[device] {device}  ({torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'})")

    t0 = time.perf_counter()
    processor, model = load_dav2(device)
    print(f"[model] loaded in {time.perf_counter()-t0:.1f}s")

    gt_data = {}
    if Path(args.gt_json).exists():
        with open(args.gt_json) as f:
            gt_data = json.load(f)

    # ──────────────────── DATASET-1 MODE ────────────────────
    if args.mode == "dataset1":
        rows, y_true, y_pred = [], [], []
        with open(args.csv) as f:
            gt_rows = list(csv.DictReader(f))
        total = len(gt_rows)

        for i, row in enumerate(gt_rows, 1):
            img_name = row["image_name"]
            gt_vol = float(row["gt_volume_liters"])
            folder = row["object_folder"]
            img_path = Path(args.img_dir) / folder / img_name
            if not img_path.exists():
                continue

            t_img = time.perf_counter()
            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)

            depth = predict_depth_dav2(processor, model, img_pil, device)

            hmap, scale, floor_d = height_map_calibrated(
                depth, CAMERA_HEIGHT_DS1, img_rgb=img_np
            )

            vol_L = compute_volume(hmap, SCENE_AREA_DS1) * 1000.0
            obj_mask = hmap > 0
            cov = float(np.mean(obj_mask) * 100)
            mean_h = float(np.mean(hmap[obj_mask]) * 100) if obj_mask.any() else 0.0
            max_h  = float(np.max(hmap) * 100)
            err_pct = (vol_L - gt_vol) / gt_vol * 100 if gt_vol > 0 else float("nan")

            elapsed = time.perf_counter() - t_img
            y_true.append(gt_vol)
            y_pred.append(vol_L)
            rows.append({
                "image_name": img_name, "folder": folder,
                "gt_vol_L": gt_vol, "pred_vol_L": round(vol_L, 4),
                "pct_error": round(err_pct, 1),
                "coverage_pct": round(cov, 1),
                "mean_h_cm": round(mean_h, 2),
                "max_h_cm": round(max_h, 2),
                "depth_scale": round(scale, 4),
                "floor_depth_raw": round(floor_d, 4),
                "runtime_s": round(elapsed, 2),
            })

            print(f"[{i}/{total}] {img_name:20s} GT={gt_vol:.3f}L  Pred={vol_L:.3f}L  "
                  f"err={err_pct:+.0f}%  cov={cov:.0f}%  max_h={max_h:.1f}cm  t={elapsed:.1f}s")

            # Vis
            heat = plt.get_cmap("hot")(np.clip(hmap/(max(float(np.max(hmap)),1e-9)),0,1))[...,:3]
            vis = img_np.copy()
            vis[obj_mask] = ((1-.6)*vis[obj_mask]+.6*(heat[obj_mask]*255)).astype(np.uint8)
            Image.fromarray(vis).save(vis_dir/f"{Path(img_name).stem}_dav2.jpg", quality=80)

        metrics = regression_metrics(y_true, y_pred) if y_true else {}

        # Per-object aggregate
        from collections import defaultdict
        by_obj = defaultdict(list); by_obj_gt = {}
        for r in rows:
            by_obj[r["folder"]].append(r["pred_vol_L"])
            by_obj_gt[r["folder"]] = r["gt_vol_L"]
        obj_rows, yt_obj, yp_obj = [], [], []
        for obj, preds in sorted(by_obj.items()):
            gt_v = by_obj_gt[obj]
            mp = float(np.mean(preds))
            cv_p = float(np.std(preds)/np.mean(preds)*100) if np.mean(preds)>0 else 0
            obj_rows.append({"object": obj, "gt_L": gt_v, "mean_pred_L": round(mp,3),
                              "std_L": round(float(np.std(preds)),3), "CV_pct": round(cv_p,1),
                              "pct_err": round((mp-gt_v)/gt_v*100,1) if gt_v>0 else "n/a"})
            yt_obj.append(gt_v); yp_obj.append(mp)
        metrics_obj = regression_metrics(yt_obj, yp_obj) if yt_obj else {}

        # Save
        with open(out_dir/"results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        with open(out_dir/"per_object.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=obj_rows[0].keys()); w.writeheader(); w.writerows(obj_rows)
        with open(out_dir/"summary.json", "w") as f:
            json.dump({"model": "DepthAnythingV2-Metric-Indoor-Large",
                       "camera_height_m": CAMERA_HEIGHT_DS1,
                       "metrics_all": metrics, "metrics_per_obj": metrics_obj}, f, indent=2)

        # Scatter plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        gt_arr, pd_arr = np.array(y_true), np.array(y_pred)
        ax = axes[0]
        ax.scatter(gt_arr, pd_arr, c="steelblue", alpha=0.7, edgecolors="k", linewidths=.5)
        lo, hi = min(gt_arr.min(),pd_arr.min())*.9, max(gt_arr.max(),pd_arr.max())*1.1
        ax.plot([lo,hi],[lo,hi],"r--",lw=1.5,label="Perfect")
        ax.set_xlabel("GT Volume (L)"); ax.set_ylabel("Pred Volume (L)")
        ax.set_title(f"All | MAE={metrics['mae_L']:.2f}L  MAPE={metrics['mape_pct']:.0f}%  R²={metrics['r2']:.2f}")
        ax.legend(); ax.grid(True, alpha=.3)

        ax2 = axes[1]
        objs = [r["object"].split("_",1)[1] if "_" in r["object"] else r["object"] for r in obj_rows]
        x = np.arange(len(objs)); wb = .35
        ax2.bar(x-wb/2, [r["gt_L"] for r in obj_rows], wb, label="GT", color="steelblue")
        ax2.bar(x+wb/2, [r["mean_pred_L"] for r in obj_rows], wb, label="Pred (DAv2)", color="coral")
        ax2.set_xticks(x); ax2.set_xticklabels(objs, rotation=40, ha="right", fontsize=8)
        ax2.set_ylabel("Volume (L)")
        ax2.set_title(f"Per-Object | MAE={metrics_obj['mae_L']:.2f}L  R²={metrics_obj['r2']:.2f}")
        ax2.legend(); ax2.grid(True, alpha=.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir/"eval_chart.png", dpi=150, bbox_inches="tight"); plt.close()

        print("\n" + "="*60)
        print("DATASET-1 DAv2 RESULTS")
        print("="*60)
        for k, v in metrics.items():
            print(f"  {k:20s}: {v:.3f}")
        print("\nPer-object:")
        for k, v in metrics_obj.items():
            print(f"  {k:20s}: {v:.3f}")

    # ──────────────────── IITB MODE ────────────────────
    elif args.mode == "iitb":
        piles = [p.strip() for p in args.piles.split(",")]
        img_dir = Path(args.iitb_dir)
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

                box_mask, box_pts = detect_white_box(img_np)
                depth = predict_depth_dav2(processor, model, img_pil, device)

                hmap, scale, floor_d = height_map_calibrated(
                    depth, args.camera_height_m, img_rgb=img_np, box_mask=box_mask
                )

                vol_m3 = compute_volume(hmap, args.box_area_m2)
                vol_L  = vol_m3 * 1000.0
                obj_mask = hmap > 0
                fill = float(np.mean(obj_mask[box_mask])*100) if box_mask.any() else 0.0
                mean_h = float(np.mean(hmap[obj_mask])*100) if obj_mask.any() else 0.0
                max_h  = float(np.max(hmap)*100)
                gt = gt_data.get(pile, {})
                density = gt.get("mass_kg") / vol_L if gt.get("mass_kg") and vol_L > 0 else None
                elapsed = time.perf_counter() - t_img

                entry = {
                    "image": img_path.name, "vol_L": round(vol_L,4),
                    "fill_pct": round(fill,1), "mean_h_cm": round(mean_h,2),
                    "max_h_cm": round(max_h,2), "scale": round(scale,4),
                    "floor_raw_m": round(floor_d,4), "runtime_s": round(elapsed,2),
                }
                if density is not None:
                    entry["density_kg_L"] = round(density, 4)
                pile_results.append(entry)
                print(f"  [{j}/{len(imgs)}] {img_path.name}  vol={vol_L:.2f}L  fill={fill:.0f}%  "
                      f"max_h={max_h:.1f}cm  t={elapsed:.1f}s")

                # Vis
                vis = img_np.copy()
                pts_int = box_pts.astype(np.int32).reshape(-1,1,2)
                cv2.polylines(vis, [pts_int], True, (0,255,0), 4)
                vmax = float(np.max(hmap)) if hmap.max()>0 else 1.0
                heat = (plt.get_cmap("hot")(np.clip(hmap/(vmax+1e-9),0,1))[...,:3]*255).astype(np.uint8)
                vis[obj_mask] = ((1-.6)*vis[obj_mask]+.6*heat[obj_mask]).astype(np.uint8)
                sc = min(1.0, 1024/max(vis.shape[:2]))
                if sc < 1.0: vis = cv2.resize(vis, (int(vis.shape[1]*sc), int(vis.shape[0]*sc)))
                Image.fromarray(vis).save(vis_dir/f"{pile}_{img_path.stem}_dav2.jpg", quality=80)

            vols = [r["vol_L"] for r in pile_results]
            mv, sv = float(np.mean(vols)), float(np.std(vols))
            cv = float(sv/mv*100) if mv>0 else 0
            gt = gt_data.get(pile, {})
            ps = {"pile": pile, "n": len(vols), "mean_vol_L": round(mv,3),
                  "std_vol_L": round(sv,3), "cv_pct": round(cv,1), "median_vol_L": round(float(np.median(vols)),3)}
            if gt.get("mass_kg"): ps["gt_mass_kg"] = gt["mass_kg"]; ps["impl_density_kg_L"] = round(gt["mass_kg"]/mv,3) if mv>0 else None
            if gt.get("volume_L"): ps["gt_vol_L"] = gt["volume_L"]; ps["pct_err"] = round((mv-gt["volume_L"])/gt["volume_L"]*100,1)
            all_results[pile] = {"summary": ps, "per_image": pile_results}
            print(f"[{pile}] mean={mv:.2f}L  std={sv:.2f}L  CV={cv:.1f}%")

        with open(out_dir/"results.json","w") as f: json.dump(all_results, f, indent=2)

        # Plot
        fig, axes = plt.subplots(1, max(1,len(all_results)), figsize=(5*max(1,len(all_results)),5), squeeze=False)
        for k, (pile, data) in enumerate(all_results.items()):
            ax = axes[0][k]
            vols = [r["vol_L"] for r in data["per_image"]]
            ax.bar(range(len(vols)), vols, color="darkorange", alpha=.8)
            ax.axhline(data["summary"]["mean_vol_L"], color="navy", linestyle="--", lw=2, label="Mean")
            if data["summary"].get("gt_vol_L"):
                ax.axhline(data["summary"]["gt_vol_L"], color="red", linestyle=":", lw=2, label="GT")
            ax.set_title(f"{pile} (DAv2) | {data['summary']['mean_vol_L']:.2f}L | CV={data['summary']['cv_pct']:.1f}%")
            ax.set_ylabel("Volume (L)"); ax.legend(fontsize=7); ax.grid(True,alpha=.3,axis="y")
        plt.suptitle("DepthAnythingV2 Metric Indoor", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir/"dav2_iitb_chart.png", dpi=150, bbox_inches="tight"); plt.close()

        print("\n" + "="*60 + "\nIITB DAv2 SUMMARY\n" + "="*60)
        for pile, data in all_results.items():
            s = data["summary"]
            print(f"\n  {pile}:")
            for k,v in s.items(): print(f"    {k:30s}: {v}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()
