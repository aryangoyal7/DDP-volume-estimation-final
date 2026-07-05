#!/usr/bin/env bash
# Run all evaluations — edit the calibration params before running!
set -e
cd "$(dirname "$0")/.."

# ─── CALIBRATION PARAMS — MUST FILL IN ─────────────────────────────────────
# Measure these physically from your setup:
BOX_LENGTH_M=0.40      # interior length of white box in metres
BOX_WIDTH_M=0.30       # interior width of white box in metres
CAMERA_H_M=1.00        # camera height above box floor in metres

# Auto-compute box area
BOX_AREA=$(python3 -c "print($BOX_LENGTH_M * $BOX_WIDTH_M)")
echo "Box area: ${BOX_AREA} m²  |  Camera height: ${CAMERA_H_M} m"

# ─── GPU SELECTION ──────────────────────────────────────────────────────────
# Use A6000 (cuda:1) if available, else RTX 4090 (cuda:0)
DEVICE="cuda:1"

# ─── EVAL 1: Dataset-1 (small objects, known GT) ────────────────────────────
echo ""
echo "============================================================"
echo "EVAL 1: Dataset-1 (planar depth, no segmentation)"
echo "============================================================"
python3 evaluation/eval_dataset1.py \
    --csv     dataset_1/eval_gt_volume_liters.csv \
    --img_dir dataset_1/jpg_by_object \
    --out_dir evaluation/dataset1_gpu \
    --device  $DEVICE

# ─── EVAL 2: IITB full images (auto white-box detection) ────────────────────
echo ""
echo "============================================================"
echo "EVAL 2: IITB full images (auto white-box detection)"
echo "============================================================"
python3 evaluation/eval_iitb.py \
    --img_dir         solid_waste_dataset2_iitb_site_pho \
    --out_dir         evaluation/iitb_autobox \
    --piles           pile1,pile2,pile3 \
    --box_area_m2     $BOX_AREA \
    --camera_height_m $CAMERA_H_M \
    --gt_json         evaluation/iitb_gt.json \
    --device          $DEVICE

# ─── EVAL 3: IITB pre-cropped (pile1 only) ──────────────────────────────────
echo ""
echo "============================================================"
echo "EVAL 3: IITB pre-cropped images (pile1)"
echo "============================================================"
python3 evaluation/eval_iitb.py \
    --img_dir         datasets/iitb_cropped \
    --out_dir         evaluation/iitb_cropped \
    --piles           pile1 \
    --box_area_m2     $BOX_AREA \
    --camera_height_m $CAMERA_H_M \
    --gt_json         evaluation/iitb_gt.json \
    --pre_cropped \
    --device          $DEVICE

# ─── EVAL 4: LangSAM on IITB (approach 2) ───────────────────────────────────
echo ""
echo "============================================================"
echo "EVAL 4: LangSAM segmentation + DINOv3 depth (IITB)"
echo "============================================================"
python3 evaluation/eval_langsam.py \
    --img_dir         solid_waste_dataset2_iitb_site_pho \
    --out_dir         evaluation/langsam_iitb \
    --mode            iitb \
    --prompt          "solid waste pile" \
    --scene_area_m2   $BOX_AREA \
    --camera_height_m $CAMERA_H_M \
    --gt_json         evaluation/iitb_gt.json \
    --device          $DEVICE

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "Results in: evaluation/"
echo "============================================================"
