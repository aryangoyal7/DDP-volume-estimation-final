#!/usr/bin/env python3
"""Smoke test: import app.estimate_volume and run it on one
IITB image from each pile. Confirm the calibrated volumes match the
report's pipeline within reasonable tolerance.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import estimate_volume, IITB_DEFAULTS

# Reference numbers from latex_report/main.tex and evaluation/`/per_image_results.csv
REFERENCE = {
    "pile1/IMG_7975.jpg": dict(raw_L=14.376, scale=0.2696, sanity_ok=True, sanity_diff_cm=6.53),
    "pile2/IMG_7983.jpg": dict(raw_L=38.737, scale=0.2894, sanity_ok=True, sanity_diff_cm=7.82),
    "pile3/IMG_7999.jpg": dict(raw_L=12.856, scale=0.2825, sanity_ok=True, sanity_diff_cm=3.66),
}

ds_root = ROOT / "solid_waste_dataset2_iitb_site_pho"

print("=" * 78)
print(f"Smoke test: app.estimate_volume in Full-frame mode")
print(f"IITB defaults: {IITB_DEFAULTS}")
print("=" * 78)

for rel, ref in REFERENCE.items():
    img_path = ds_root / rel
    if not img_path.exists():
        print(f"\n[skip] {img_path} not found")
        continue
    img_np = np.array(Image.open(img_path).convert("RGB"))
    seg, dep, heat, md, m = estimate_volume(img_np, mode="Full-frame")

    raw_ok = abs(m["vol_raw_L"] - ref["raw_L"]) < 1.5  # tolerance of 1.5 L
    sanity_match = (m["sanity_ok"] == ref["sanity_ok"])
    scale_ok = abs(m["depth_scale"] - ref["scale"]) < 0.02
    diff_ok = abs(m["sanity_diff_cm"] - ref["sanity_diff_cm"]) < 1.5

    flag_raw    = "OK" if raw_ok    else "FAIL"
    flag_san    = "OK" if sanity_match else "FAIL"
    flag_scale  = "OK" if scale_ok  else "FAIL"
    flag_diff   = "OK" if diff_ok   else "FAIL"

    print(f"\n--- {rel} ---")
    print(f"raw    : got {m['vol_raw_L']:7.3f} L   (ref {ref['raw_L']:7.3f} L)  [{flag_raw}]")
    print(f"calib  : got {m['vol_cal_L']:7.3f} L   (CF = {m['calib_factor']})")
    print(f"scale  : got {m['depth_scale']:6.4f}   (ref {ref['scale']:6.4f})  [{flag_scale}]")
    print(f"sanity : got {m['sanity_ok']} ({m['sanity_diff_cm']:.2f} cm)   "
          f"(ref {ref['sanity_ok']}, {ref['sanity_diff_cm']:.2f} cm)  [{flag_san}/{flag_diff}]")
    print(f"runtime: {m['runtime_s']:.2f} s")
