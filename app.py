#!/usr/bin/env python3
"""
Gradio text-based web app: calibrated 2D volume estimation.

Two modes:
  1. Full-frame mode (default, matches the IITB report pipeline)
     - No segmentation. The whole image is integrated.
     - Plane fit on deepest 1% of pixels = floor.
     - Metric scale s = H_cam / median floor depth.
     - Height = s * (d_plane - d), threshold 1 cm, integrate * pixel area.
     - V_est = CF * V_raw   (CF = 0.86 = IITB global median by default).

  2. Prompted segmentation mode
     - LangSAM (SAM 2.1 + GroundingDINO) on a text prompt -> pile mask.
     - Floor fit on non-pile background.
     - Height integrated only inside the prompt mask.
     - Same calibration factor.

Run:
    python app.py
"""

import os
import sys
import time
import warnings
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import gradio as gr
import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ── project root (for lang_sam) ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_sam import LangSAM

# ── constants ─────────────────────────────────────────────────────────────────
DAV2_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
SAM_TYPE      = "sam2.1_hiera_large"
GDINO_ID      = "IDEA-Research/grounding-dino-base"

# IITB defaults: derived from evaluation/`/calibration_summary.json
IITB_DEFAULTS = dict(
    camera_height_m=0.65,
    scene_w_m=0.60,
    scene_h_m=0.60,
    calib_factor=0.86,   # global median CF across pile1, pile2, pile3
    density=180.0,
    height_thresh_cm=1.0,
    floor_pct=99.0,
)

# Per-pile factors for the quick-select dropdown
PRESET_CFS = {
    "Global median (CF = 0.86)": 0.86,
    "Pile 1 (670 g, CF = 1.38)": 1.38,
    "Pile 2 (480 g, CF = 0.48)": 0.48,
    "Pile 3 (660 g, CF = 1.02)": 1.02,
    "Uncalibrated (CF = 1.00)":  1.00,
}


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_dav2(device: str):
    print(f"[DAv2] loading {DAV2_MODEL_ID} ...")
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(DAV2_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DAV2_MODEL_ID)
    model.eval().to(device)
    print(f"[DAv2] ready in {time.time()-t0:.1f}s on {device}")
    return proc, model


@lru_cache(maxsize=1)
def _load_langsam():
    print("[LangSAM] loading ...")
    t0 = time.time()
    model = LangSAM(sam_type=SAM_TYPE, gdino_model_id=GDINO_ID)
    print(f"[LangSAM] ready in {time.time()-t0:.1f}s")
    return model


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline functions
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _predict_depth(img_pil: Image.Image) -> np.ndarray:
    device = _device()
    proc, model = _load_dav2(device)
    h, w = img_pil.height, img_pil.width
    inputs = proc(images=img_pil, return_tensors="pt").to(device)
    if device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**inputs)
    else:
        out = model(**inputs)
    depth = out.predicted_depth
    depth = F.interpolate(depth[None], size=(h, w), mode="bilinear",
                          align_corners=False)[0, 0]
    return depth.float().cpu().numpy()


def _segment(img_pil: Image.Image, prompt: str,
             box_thresh: float, text_thresh: float) -> np.ndarray:
    """Return binary union mask (HxW bool). Falls back to full frame if nothing found."""
    lsam = _load_langsam()
    h, w = img_pil.height, img_pil.width
    for bt, tt in [(box_thresh, text_thresh), (box_thresh * 0.5, text_thresh * 0.5)]:
        res = lsam.predict([img_pil], [prompt], box_threshold=bt, text_threshold=tt)
        masks = res[0].get("masks", [])
        if len(masks):
            union = np.zeros((h, w), dtype=bool)
            for m in masks:
                m2 = m[0] if m.ndim == 3 else m
                union |= m2.astype(bool)
            return union
    return np.ones((h, w), dtype=bool)


def _fit_floor_plane(depth: np.ndarray,
                     candidate_mask: np.ndarray,
                     floor_pct: float) -> np.ndarray:
    """Fit a tilted plane to the deepest (floor) pixels inside candidate_mask."""
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0) & candidate_mask
    if valid.sum() < 50:
        valid = np.isfinite(depth) & (depth > 0)
    thr = np.percentile(depth[valid], floor_pct)
    floor = valid & (depth >= thr)
    rows, cols = np.where(floor)
    d_floor = depth[rows, cols]
    if len(d_floor) < 10:
        return np.full((h, w), float(np.median(d_floor)) if len(d_floor) else 1.0,
                       dtype=np.float32), floor
    A = np.c_[cols, rows, np.ones(len(cols))]
    coeffs, *_ = np.linalg.lstsq(A, d_floor, rcond=None)
    cc, rr = np.meshgrid(np.arange(w), np.arange(h))
    plane = (coeffs[0] * cc + coeffs[1] * rr + coeffs[2]).astype(np.float32)
    return plane, floor


def _height_map(depth: np.ndarray,
                plane: np.ndarray,
                floor_mask: np.ndarray,
                camera_h: float,
                height_thresh_m: float) -> Tuple[np.ndarray, float, float]:
    """Build a calibrated height map. Returns (height, depth_scale, med_floor_raw)."""
    med_floor = float(np.median(plane[floor_mask])) if floor_mask.any() else float(np.median(plane))
    scale = camera_h / med_floor if med_floor > 1e-9 else 1.0
    scaled_plane = plane * scale
    scaled_depth = depth * scale
    height_raw = scaled_plane - scaled_depth
    height_raw = np.where(depth > 0, height_raw, 0.0)
    height = np.where(height_raw > height_thresh_m, height_raw - height_thresh_m, 0.0)
    return height.astype(np.float32), scale, med_floor


def _depth_sanity_check(depth: np.ndarray) -> Tuple[bool, float, float, float]:
    """Centre depth should be smaller (closer) than border depth.
    Returns (ok, center_mean, edge_mean, diff_cm_scaled).  diff_cm_scaled is
    raw difference; caller must multiply by scale*100 if they want cm.
    """
    h, w = depth.shape
    he, we = int(0.15 * h), int(0.15 * w)
    center = depth[he:h-he, we:w-we]
    border = np.concatenate([
        depth[:he, :].flat, depth[h-he:, :].flat,
        depth[:, :we].flat, depth[:, w-we:].flat,
    ])
    cv = center[center > 0]
    ev = border[border > 0]
    if cv.size == 0 or ev.size == 0:
        return False, 0.0, 0.0, 0.0
    mean_c = float(np.mean(cv))
    mean_e = float(np.mean(ev))
    return mean_c < mean_e, mean_c, mean_e, mean_e - mean_c


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth[depth > 0]
    lo = float(np.percentile(valid, 2)) if valid.size else 0.0
    hi = float(np.percentile(valid, 98)) if valid.size else 1.0
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    rgba = cm.plasma(1.0 - norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _blend_heatmap(img_rgb: np.ndarray, height: np.ndarray,
                   mask: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    vis = np.where(mask, height, 0.0)
    vmax = float(np.max(vis)) if vis.max() > 0 else 1.0
    rgba = cm.inferno(np.clip(vis / (vmax + 1e-9), 0, 1))
    heat = (rgba[..., :3] * 255).astype(np.uint8)
    blended = np.where(mask[..., None],
                       (alpha * heat + (1 - alpha) * img_rgb).astype(np.uint8),
                       img_rgb)
    return blended


def _sam_overlay(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = img_rgb.copy()
    sel = mask.astype(bool)
    overlay[sel] = (0.5 * overlay[sel] + 0.5 * np.array([255, 120, 30])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 80, 0), 2)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Headless inference entry point (also used by Gradio button)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_volume(
    img_np: np.ndarray,
    mode: str = "Full-frame",
    prompt: str = "waste pile",
    camera_height: float = IITB_DEFAULTS["camera_height_m"],
    scene_w: float = IITB_DEFAULTS["scene_w_m"],
    scene_h: float = IITB_DEFAULTS["scene_h_m"],
    calib_factor: float = IITB_DEFAULTS["calib_factor"],
    density: float = IITB_DEFAULTS["density"],
    height_thresh_cm: float = IITB_DEFAULTS["height_thresh_cm"],
    floor_pct: float = IITB_DEFAULTS["floor_pct"],
    box_thresh: float = 0.30,
    text_thresh: float = 0.25,
):
    """Run the full pipeline. Returns:
        seg_vis, depth_vis, heat_vis, metrics_md, metrics_dict
    """
    img_pil = Image.fromarray(img_np).convert("RGB")
    iH, iW = img_np.shape[:2]
    t0 = time.time()

    # ── 1. Mask ────────────────────────────────────────────────────────────
    if mode == "Full-frame":
        pile_mask = np.ones((iH, iW), dtype=bool)
        seg_vis = img_np.copy()
    else:
        pile_mask = _segment(img_pil, prompt.strip() or "waste pile",
                             box_thresh, text_thresh)
        pile_mask = np.array(Image.fromarray((pile_mask.astype(np.uint8) * 255))
                             .resize((iW, iH), Image.NEAREST)).astype(bool)
        seg_vis = _sam_overlay(img_np, pile_mask)

    # ── 2. Depth ──────────────────────────────────────────────────────────
    depth = _predict_depth(img_pil)
    depth_vis = _colorize_depth(depth)

    # ── 3. Floor plane and height map ─────────────────────────────────────
    if mode == "Full-frame":
        candidate = np.ones_like(depth, dtype=bool)
    else:
        candidate = ~pile_mask
    plane, floor_mask = _fit_floor_plane(depth, candidate, float(floor_pct))
    height_map, scale, med_floor = _height_map(
        depth, plane, floor_mask, float(camera_height),
        float(height_thresh_cm) / 100.0,
    )

    # ── 4. Sanity check ───────────────────────────────────────────────────
    sanity_ok, c_d, e_d, diff_raw = _depth_sanity_check(depth)
    diff_cm = diff_raw * scale * 100.0

    # ── 5. Volume integration ─────────────────────────────────────────────
    scene_area_m2 = float(scene_w) * float(scene_h)
    px_area_m2 = scene_area_m2 / (iH * iW)
    active = pile_mask & (height_map > 0)
    active_h = height_map[active]

    if active_h.size == 0:
        vol_raw_L = 0.0
        mean_h_cm = max_h_cm = 0.0
    else:
        vol_raw_L = float(np.sum(active_h) * px_area_m2 * 1000.0)
        mean_h_cm = float(np.mean(active_h) * 100.0)
        max_h_cm = float(np.max(active_h) * 100.0)

    vol_cal_L = vol_raw_L * float(calib_factor)
    mass_kg = (vol_cal_L / 1000.0) * float(density)
    fill_pct = float(np.mean(active) * 100.0)
    elapsed = time.time() - t0

    # ── 6. Visualisation ──────────────────────────────────────────────────
    heat_vis = _blend_heatmap(img_np, height_map, active)

    metrics = dict(
        mode=mode, prompt=prompt, camera_h=camera_height,
        scene_area_m2=scene_area_m2, calib_factor=calib_factor,
        vol_raw_L=vol_raw_L, vol_cal_L=vol_cal_L,
        mass_kg=mass_kg, density=density,
        mean_h_cm=mean_h_cm, max_h_cm=max_h_cm,
        depth_scale=scale, med_floor_raw=med_floor,
        sanity_ok=sanity_ok, sanity_diff_cm=diff_cm,
        fill_pct=fill_pct, runtime_s=elapsed,
    )

    sanity_label = "Verified" if sanity_ok else "Check failed"
    sanity_class = "ok" if sanity_ok else "bad"

    md = f"""
<div class="result-card">
  <div class="hero">
    <div class="hero-block">
      <div class="hero-label">Volume</div>
      <div class="hero-value">{vol_cal_L:.1f}<span class="hero-unit"> L</span></div>
      <div class="hero-sub">calibration factor {calib_factor:.2f}</div>
    </div>
    <div class="hero-divider"></div>
    <div class="hero-block">
      <div class="hero-label">Mass</div>
      <div class="hero-value">{mass_kg:.2f}<span class="hero-unit"> kg</span></div>
      <div class="hero-sub">at {density:.0f} kg / m&sup3;</div>
    </div>
  </div>

  <div class="stat-row">
    <div class="stat"><div class="stat-label">Raw volume</div><div class="stat-value">{vol_raw_L:.1f} L</div></div>
    <div class="stat"><div class="stat-label">Mean height</div><div class="stat-value">{mean_h_cm:.1f} cm</div></div>
    <div class="stat"><div class="stat-label">Max height</div><div class="stat-value">{max_h_cm:.1f} cm</div></div>
    <div class="stat"><div class="stat-label">Coverage</div><div class="stat-value">{fill_pct:.0f}%</div></div>
  </div>

  <div class="footnote-row">
    <span class="badge {sanity_class}">Depth check &middot; {sanity_label}</span>
    <span class="muted">centre vs edges {diff_cm:.1f} cm &nbsp;&middot;&nbsp; runtime {elapsed:.1f} s &nbsp;&middot;&nbsp; mode {mode.lower()}</span>
  </div>
</div>
"""
    return seg_vis, depth_vis, heat_vis, md, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Gradio wrapper
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    image, mode, prompt, camera_height, scene_w, scene_h,
    calib_preset, calib_factor_manual,
    density, height_thresh_cm, floor_pct, box_thresh, text_thresh,
):
    if image is None:
        raise gr.Error("Upload an image first.")
    if mode == "Prompted segmentation" and (not prompt or not prompt.strip()):
        raise gr.Error("In prompted-segmentation mode, enter a text prompt.")

    # If the user picked a preset, prefer it. Otherwise use the manual number.
    if calib_preset and calib_preset in PRESET_CFS:
        cf = PRESET_CFS[calib_preset]
    else:
        cf = float(calib_factor_manual)

    seg_vis, depth_vis, heat_vis, md, _ = estimate_volume(
        img_np=image, mode=mode, prompt=prompt or "",
        camera_height=camera_height, scene_w=scene_w, scene_h=scene_h,
        calib_factor=cf, density=density,
        height_thresh_cm=height_thresh_cm, floor_pct=floor_pct,
        box_thresh=box_thresh, text_thresh=text_thresh,
    )
    return seg_vis, depth_vis, heat_vis, md


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ===== base ===== */
body, .gradio-container {
    background: #f4f6fb !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Inter, "Helvetica Neue", Arial, sans-serif !important;
    color: #1e293b;
    font-size: 18px;
    line-height: 1.55;
}
.gradio-container { max-width: 1280px !important; margin: 24px auto !important; }

/* hide gradio's default footer */
footer, .footer { display: none !important; }

/* ===== header ===== */
#hero {
    text-align: center;
    padding: 38px 24px 30px 24px;
    margin-bottom: 22px;
}
#hero h1 {
    font-size: 2.7em;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #0f172a;
    margin: 0 0 12px 0;
}
#hero p {
    font-size: 1.18em;
    color: #5b6678;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ===== panels ===== */
.panel {
    background: #ffffff;
    border-radius: 18px;
    padding: 26px 28px;
    box-shadow: 0 2px 6px rgba(15,23,42,0.06),
                0 1px 2px rgba(15,23,42,0.04);
    border: 1px solid #e6e8ef;
    margin-bottom: 20px;
}
.panel h3 {
    font-size: 1.3em !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 0 0 16px 0 !important;
    letter-spacing: -0.01em;
}
.panel p { font-size: 1.02em; }

/* ===== form inputs - bigger and friendlier ===== */
input, select, textarea {
    font-size: 1.1em !important;
    border-radius: 12px !important;
    border: 1.5px solid #d4d8e2 !important;
    background: #fafbfd !important;
    padding: 12px 14px !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
label > span, .label-wrap > span {
    font-size: 1.05em !important;
    font-weight: 600 !important;
    color: #334155 !important;
}
.label-wrap .info, .info-wrap { color: #475569 !important; font-size: 0.97em !important; font-weight: 500 !important; }

/* radio - cleaner, larger segmented look */
.gr-form .gr-box { border-radius: 12px !important; }
.gradio-radio { gap: 10px !important; }
.gradio-radio label {
    background: #f1f4f9;
    border: 1.5px solid #e2e6ef;
    border-radius: 12px;
    padding: 12px 18px !important;
    margin: 3px;
    font-size: 1.05em !important;
    font-weight: 600 !important;
    transition: all .12s ease;
}
.gradio-radio label:hover { background: #e8ecf5; }
.gradio-radio label.selected,
.gradio-radio input:checked + span {
    background: #eef0ff !important;
    border-color: #6366f1 !important;
    color: #4338ca !important;
}

/* ===== text prompt - make it pop ===== */
#prompt-box textarea, #prompt-box input {
    font-size: 1.2em !important;
    padding: 16px !important;
    border: 2px solid #c7cdf5 !important;
    background: #ffffff !important;
}

/* ===== run button - big and bold ===== */
#run-btn {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 1.25em !important;
    font-weight: 700 !important;
    color: #fff !important;
    padding: 18px 0 !important;
    margin-top: 10px !important;
    box-shadow: 0 6px 18px rgba(79,70,229,0.30) !important;
    transition: transform .12s ease, box-shadow .12s ease !important;
}
#run-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(79,70,229,0.38) !important;
}

/* ===== tabs ===== */
.tab-nav { border-bottom: 2px solid #e6e8ef !important; }
.tab-nav button {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 1.08em !important;
    padding: 12px 20px !important;
    border-radius: 10px 10px 0 0 !important;
}
.tab-nav button.selected {
    color: #4f46e5 !important;
    border-bottom: 3px solid #4f46e5 !important;
    background: transparent !important;
}

/* ===== images ===== */
.output-image img, .input-image img { border-radius: 14px !important; }

/* ===== accordion ===== */
details {
    background: #fafbfd;
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 12px;
    border: 1px solid #eceef4;
}
details summary {
    cursor: pointer;
    font-weight: 600;
    font-size: 1.05em;
    color: #4f46e5 !important;
    padding: 8px 0;
}

/* ===== result hero card ===== */
.result-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px 28px 22px 28px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05),
                0 1px 2px rgba(15,23,42,0.04);
    border: 1px solid #e6e8ef;
}
.hero {
    display: flex;
    align-items: center;
    gap: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid #f1f3f8;
}
.hero-block { flex: 1; }
.hero-divider {
    width: 1px;
    align-self: stretch;
    background: #f1f3f8;
}
.hero-label {
    font-size: 0.85em;
    font-weight: 500;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
}
.hero-value {
    font-size: 3.4em;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #0f172a;
    line-height: 1.05;
}
.hero-unit {
    font-size: 0.45em;
    font-weight: 500;
    color: #64748b;
    margin-left: 4px;
}
.hero-sub {
    font-size: 0.9em;
    color: #64748b;
    margin-top: 6px;
}

.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    padding: 18px 0 12px 0;
}
.stat {
    background: #fafbfd;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: left;
    border: 1px solid #eceef4;
}
.stat-label {
    font-size: 0.92em;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 5px;
}
.stat-value {
    font-size: 1.5em;
    font-weight: 700;
    color: #0f172a;
}

.footnote-row {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    padding-top: 12px;
    border-top: 1px solid #f1f3f8;
}
.badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.85em;
    font-weight: 500;
}
.badge.ok  { background: #dcfce7; color: #166534; }
.badge.bad { background: #fee2e2; color: #991b1b; }
.muted { color: #475569; font-size: 0.98em; }

/* result placeholder */
.placeholder {
    text-align: center;
    padding: 48px 24px;
    color: #475569;
    background: #ffffff;
    border-radius: 16px;
    border: 1px dashed #d4d8e2;
}
.placeholder .ico {
    font-size: 2.2em;
    margin-bottom: 10px;
    color: #94a3b8;
}
.placeholder p { margin: 0; font-size: 1em; }
"""

HEADER_HTML = """
<div id="hero">
  <h1>Volume Estimator</h1>
  <p>
    Upload a top-down photo, type what you want to measure (e.g.
    &ldquo;waste pile&rdquo;), and get a calibrated volume in litres plus a
    mass estimate &mdash; with the depth map and height heatmap.
  </p>
</div>
"""

PLACEHOLDER_HTML = """
<div class="placeholder">
  <div class="ico">&#9633;</div>
  <p>Upload an image and tap <strong>Estimate</strong> to see results here.</p>
</div>
"""


def _build_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="Volume Estimator") as demo:
        gr.HTML(HEADER_HTML)

        # ── result hero (top, prominent) ──
        metrics_out = gr.Markdown(value=PLACEHOLDER_HTML)

        with gr.Row(equal_height=False):

            # ── LEFT: image + controls ──
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### 1. Upload an image")
                    image_in = gr.Image(
                        type="numpy",
                        label=" ",
                        show_label=False,
                        height=320,
                    )

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### 2. What do you want to measure?")
                    mode = gr.Radio(
                        choices=["Prompted segmentation", "Full-frame"],
                        value="Prompted segmentation",
                        label="Selection method",
                        info=("Type what to measure (recommended), or use the "
                              "whole frame."),
                    )
                    prompt_in = gr.Textbox(
                        label="Describe the object to measure",
                        value="waste pile",
                        placeholder="e.g. waste pile, cardboard boxes, sand heap",
                        lines=1,
                        visible=True,
                        elem_id="prompt-box",
                        info="Plain English. The app finds and outlines it, then "
                             "measures only that region.",
                    )

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### 3. Choose calibration")
                    calib_preset = gr.Dropdown(
                        choices=list(PRESET_CFS.keys()),
                        value="Global median (CF = 0.86)",
                        label="Calibration preset",
                        info="Per-pile factors derived from the IITB experiment.",
                    )

                run_btn = gr.Button("Estimate volume", elem_id="run-btn")

                with gr.Accordion("Advanced settings", open=False):
                    gr.Markdown(
                        "Defaults match the IITB three-pile setup. "
                        "Adjust only if your camera / box / waste differs."
                    )
                    with gr.Row():
                        cam_h = gr.Number(
                            label="Camera height (m)",
                            value=IITB_DEFAULTS["camera_height_m"], precision=3,
                        )
                        scene_w = gr.Number(
                            label="Box width (m)",
                            value=IITB_DEFAULTS["scene_w_m"], precision=3,
                        )
                        scene_h = gr.Number(
                            label="Box height (m)",
                            value=IITB_DEFAULTS["scene_h_m"], precision=3,
                        )
                    with gr.Row():
                        calib_manual = gr.Number(
                            label="Manual CF override",
                            value=IITB_DEFAULTS["calib_factor"], precision=3,
                        )
                        density = gr.Number(
                            label="Density (kg / m³)",
                            value=IITB_DEFAULTS["density"], precision=1,
                        )
                    with gr.Row():
                        h_thresh = gr.Number(
                            label="Noise threshold (cm)",
                            value=IITB_DEFAULTS["height_thresh_cm"], precision=2,
                        )
                        floor_pct = gr.Number(
                            label="Floor percentile",
                            value=IITB_DEFAULTS["floor_pct"], precision=1,
                        )
                    with gr.Row():
                        box_thresh = gr.Slider(
                            0.05, 0.80, value=0.30, step=0.05,
                            label="Segmentation box threshold",
                        )
                        text_thresh = gr.Slider(
                            0.05, 0.80, value=0.25, step=0.05,
                            label="Segmentation text threshold",
                        )

            # ── RIGHT: visualisations ──
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### Visualisations")
                    with gr.Tabs():
                        with gr.Tab("Height heatmap"):
                            heat_out = gr.Image(
                                type="numpy", show_label=False, height=420,
                            )
                        with gr.Tab("Depth map"):
                            depth_out = gr.Image(
                                type="numpy", show_label=False, height=420,
                            )
                        with gr.Tab("Mask"):
                            seg_out = gr.Image(
                                type="numpy", show_label=False, height=420,
                            )

        # show/hide the prompt textbox depending on the mode
        def _toggle_prompt(m):
            return gr.update(visible=(m == "Prompted segmentation"))

        mode.change(fn=_toggle_prompt, inputs=mode, outputs=prompt_in)

        run_btn.click(
            fn=run_inference,
            inputs=[image_in, mode, prompt_in, cam_h, scene_w, scene_h,
                    calib_preset, calib_manual, density,
                    h_thresh, floor_pct, box_thresh, text_thresh],
            outputs=[seg_out, depth_out, heat_out, metrics_out],
        )
    return demo


if __name__ == "__main__":
    demo = _build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=os.getenv("GRADIO_SHARE", "0") == "1",
    )
