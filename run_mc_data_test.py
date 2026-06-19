#!/usr/bin/env python3
"""
Run the SAM-3D + DAv2 volume estimation pipeline on mc_data_test/ images.
No ground-truth volumes available — assumptions made for camera/scene geometry.
"""

import os, sys, time, json, warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lang_sam import LangSAM

# ── assumptions ──────────────────────────────────────────────────────────────
CAMERA_HEIGHT_M = 4.0          # camera held ~4 m above pile (person on platform)
SCENE_W_M       = 4.0          # bunker bay width  ~4 m
SCENE_H_M       = 5.0          # visible pile length ~5 m
DENSITY_KGM3    = 200.0        # typical mixed MRF waste bulk density
HEIGHT_THRESH_M = 0.05         # 5 cm noise floor (large scene)
FLOOR_PCT       = 99.0
PROMPT          = "waste pile"

DAV2_ID  = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
SAM_TYPE = "sam2.1_hiera_large"
GDINO_ID = "IDEA-Research/grounding-dino-base"

SRC_DIR  = Path(__file__).resolve().parent / "mc_data_test"
OUT_DIR  = Path(__file__).resolve().parent / "eval_results" / "mc_data_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ── models ───────────────────────────────────────────────────────────────────
print(f"[init] device = {DEVICE}")
print("[model] loading DepthAnythingV2 ...")
t0 = time.time()
proc  = AutoImageProcessor.from_pretrained(DAV2_ID)
dmodel = AutoModelForDepthEstimation.from_pretrained(DAV2_ID).eval().to(DEVICE)
print(f"[model] DAv2 ready in {time.time()-t0:.1f}s")

print("[model] loading LangSAM ...")
t0 = time.time()
lsam = LangSAM(sam_type=SAM_TYPE, gdino_model_id=GDINO_ID)
print(f"[model] LangSAM ready in {time.time()-t0:.1f}s")


# ── helpers ──────────────────────────────────────────────────────────────────
@torch.inference_mode()
def predict_depth(img_pil):
    h, w = img_pil.height, img_pil.width
    inputs = proc(images=img_pil, return_tensors="pt").to(DEVICE)
    if DEVICE.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = dmodel(**inputs)
    else:
        out = dmodel(**inputs)
    d = out.predicted_depth
    d = F.interpolate(d[None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return d.float().cpu().numpy()


def segment(img_pil, prompt):
    h, w = img_pil.height, img_pil.width
    for bt, tt in [(0.30, 0.25), (0.15, 0.10)]:
        res = lsam.predict([img_pil], [prompt], box_threshold=bt, text_threshold=tt)
        masks = res[0].get("masks", [])
        if len(masks):
            union = np.zeros((h, w), dtype=bool)
            for m in masks:
                m2 = m[0] if m.ndim == 3 else m
                union |= m2.astype(bool)
            return union
    print("  [warn] no SAM mask -- using full frame")
    return np.ones((h, w), dtype=bool)


def fit_floor(depth, pile_mask, cam_h):
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    bg = valid & ~pile_mask.astype(bool)
    cand = bg if bg.sum() > 100 else valid
    thr = np.percentile(depth[cand], FLOOR_PCT)
    floor = cand & (depth >= thr)
    rr, cc = np.where(floor)
    df = depth[rr, cc]
    A = np.c_[cc, rr, np.ones(len(cc))]
    coef, *_ = np.linalg.lstsq(A, df, rcond=None)
    cg, rg = np.meshgrid(np.arange(w), np.arange(h))
    floor_pred = coef[0]*cg + coef[1]*rg + coef[2]
    med = float(np.median(df))
    scale = cam_h / med if med > 0 else 1.0
    height = np.clip((floor_pred - depth) * scale, 0, None)
    return height, scale, floor_pred


def colorize_depth(depth):
    valid = depth[depth > 0]
    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    return (cm.plasma(1.0 - norm)[..., :3] * 255).astype(np.uint8)


def heatmap_blend(rgb, height, mask, alpha=0.65):
    vis = np.where(mask, height, 0.0)
    vmax = float(vis.max()) if vis.max() > 0 else 1.0
    heat = (cm.inferno(np.clip(vis / (vmax + 1e-9), 0, 1))[..., :3] * 255).astype(np.uint8)
    blended = np.where(mask[..., None], (alpha*heat + (1-alpha)*rgb).astype(np.uint8), rgb)
    return blended


def overlay_mask(rgb, mask):
    out = rgb.copy()
    sel = mask.astype(bool)
    out[sel] = (0.5*out[sel] + 0.5*np.array([255,120,30])).astype(np.uint8)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, (255,80,0), 2)
    return out


def make_panel(rgb, mask, depth_vis, heat_vis, title, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes[0].imshow(rgb);                  axes[0].set_title("(a) RGB",            fontsize=12); axes[0].axis("off")
    axes[1].imshow(overlay_mask(rgb, mask)); axes[1].set_title("(b) SAM mask (orange)", fontsize=12); axes[1].axis("off")
    axes[2].imshow(depth_vis);            axes[2].set_title("(c) DAv2 depth",     fontsize=12); axes[2].axis("off")
    axes[3].imshow(heat_vis);             axes[3].set_title("(d) Height heatmap", fontsize=12); axes[3].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


# ── pipeline per image ───────────────────────────────────────────────────────
def process(img_path: Path):
    print(f"\n[{img_path.name}] processing ...")
    img_pil = Image.open(img_path).convert("RGB")
    img_np  = np.array(img_pil)
    iH, iW  = img_np.shape[:2]

    t = time.time()
    mask = segment(img_pil, PROMPT)
    seg_t = time.time() - t
    mask_pct = float(mask.mean()*100)
    print(f"  seg: {seg_t:.1f}s, mask = {mask_pct:.1f}% of frame")

    t = time.time()
    depth = predict_depth(img_pil)
    dep_t = time.time() - t
    print(f"  depth: {dep_t:.1f}s, range = [{depth.min():.3f}, {depth.max():.3f}]")

    h_map, scale, _ = fit_floor(depth, mask, CAMERA_HEIGHT_M)
    print(f"  scale = {scale:.4f}, max height = {h_map.max()*100:.1f} cm")

    px_area = (SCENE_W_M * SCENE_H_M) / (iH * iW)
    active = mask & (h_map > HEIGHT_THRESH_M)
    n_act  = int(active.sum())
    if n_act == 0:
        vol_L = 0.0
        mean_h = max_h = 0.0
    else:
        vol_m3 = float(np.sum(h_map[active] - HEIGHT_THRESH_M) * px_area)
        vol_L  = vol_m3 * 1000.0
        mean_h = float(h_map[active].mean()*100)
        max_h  = float(h_map[active].max()*100)
    mass_kg = (vol_L/1000.0) * DENSITY_KGM3

    print(f"  volume = {vol_L:.1f} L  ({vol_L/1000:.2f} m^3)")
    print(f"  mass   = {mass_kg:.1f} kg @ {DENSITY_KGM3} kg/m^3")
    print(f"  mean / max h = {mean_h:.1f} / {max_h:.1f} cm")

    depth_vis = colorize_depth(depth)
    heat_vis  = heatmap_blend(img_np, h_map, active)

    stem = img_path.stem.replace(" ", "_")
    panel_path = OUT_DIR / f"{stem}_panel.jpg"
    make_panel(img_np, mask, depth_vis, heat_vis,
               title=f"{img_path.name}   |   V = {vol_L:.0f} L   "
                     f"({vol_L/1000:.2f} m³, ~{mass_kg:.0f} kg)",
               out_path=panel_path)
    print(f"  panel -> {panel_path.name}")

    return {
        "image": img_path.name,
        "image_hw": [iH, iW],
        "mask_pct": round(mask_pct, 2),
        "depth_min_m": round(float(depth.min()), 3),
        "depth_max_m": round(float(depth.max()), 3),
        "scale": round(scale, 4),
        "max_height_cm": round(max_h, 1),
        "mean_height_cm": round(mean_h, 1),
        "active_px_pct": round(100.0 * n_act / (iH*iW), 2),
        "volume_L": round(vol_L, 1),
        "volume_m3": round(vol_L/1000, 3),
        "estimated_mass_kg": round(mass_kg, 1),
        "panel_image": panel_path.name,
    }


# ── run ─────────────────────────────────────────────────────────────────────
images = sorted([p for p in SRC_DIR.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
print(f"\nFound {len(images)} images in {SRC_DIR}")

results = []
for p in images:
    try:
        results.append(process(p))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"image": p.name, "error": str(e)})

summary = {
    "assumptions": {
        "camera_height_m": CAMERA_HEIGHT_M,
        "scene_w_m": SCENE_W_M,
        "scene_h_m": SCENE_H_M,
        "scene_area_m2": SCENE_W_M*SCENE_H_M,
        "density_kg_per_m3": DENSITY_KGM3,
        "height_threshold_m": HEIGHT_THRESH_M,
        "floor_percentile": FLOOR_PCT,
        "calibration_factor": 1.0,
        "prompt": PROMPT,
        "depth_model": DAV2_ID,
        "sam_model": SAM_TYPE,
        "gdino_model": GDINO_ID,
    },
    "results": results,
}
out_json = OUT_DIR / "summary.json"
out_json.write_text(json.dumps(summary, indent=2))
print(f"\nSaved summary -> {out_json}")
print(json.dumps(summary, indent=2))
