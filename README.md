# Waste Volume Estimation from a Single RGB Image

**Dual Degree Project (Master's Thesis) — Aryan Goyal**

This repository estimates the **volume and mass of waste piles from a single RGB image**. It combines:

- **DepthAnythingV2** (Metric-Indoor-Large) — metric monocular depth estimation
- **LangSAM** (GroundingDINO + SAM 2.1) — text-prompted segmentation, so the pile of interest is selected with a natural-language prompt (e.g. `"waste pile"`, `"plastic bottles"`)
- **SAM-3D pipeline** — back-projection of the segmented depth map to a 3D point cloud, with volume computed by height integration, voxel-grid counting, and convex hull

The estimated volume is converted to mass with a material bulk density (kg/m³).

## Method overview

Given one RGB image:

1. **Depth** — DepthAnythingV2 predicts a metric depth map.
2. **Segmentation (optional)** — LangSAM produces a pile mask from a text prompt. In full-frame mode this step is skipped and the whole image is integrated.
3. **Floor fitting** — a plane is fit on the deepest pixels (the floor); the known camera height gives the metric scale `s = H_cam / median(floor depth)`.
4. **Height map** — per-pixel height above the floor plane, `h = s · (d_plane − d)`, thresholded at 1 cm to suppress noise.
5. **Volume** — Riemann sum of heights × pixel footprint area, corrected by a calibration factor (CF = 0.86, the global median derived from the ground-truth calibration set).
6. **Mass** — `m = ρ · V` for a chosen bulk density.

The SAM-3D variant (`evaluation/run_iitb_sam3d.py`) additionally exports the segmented point cloud as a `.ply` file and cross-checks the integrated volume against voxel-grid and convex-hull estimates.

## Repository structure

```
├── app.py                  # Demo portal (Gradio web app) — main entry point
├── lang_sam/               # Text-prompted segmentation package (GroundingDINO + SAM 2.1)
├── scripts/                # Batch pipelines and utilities
│   ├── conversational_vla.py           # Multi-turn CLI: prompt → mask → volume/mass
│   ├── run_langsam_dinov3_mass_volume.py  # Batch eval: LangSAM masks + depth maps
│   ├── run_dinov3_volume_batch.py      # Depth-only batch volume estimation
│   ├── run_volume_baseline.py          # Baseline (no text prompt, plane-fit floor)
│   ├── run_volume_whitebox.py          # White-box dataset evaluation
│   ├── run_mc_data_test.py             # SAM-3D + DAv2 on MRF bunker images
│   └── annotate_tool.py                # White-box corner annotation tool
├── evaluation/             # Evaluation scripts, ground truth, and run_all_evals.sh
├── apps/legacy/            # Earlier app iterations (kept for provenance)
├── configs/                # Calibration files (box_calibration.json)
├── datasets/               # Sample images (spectralwaste, zero_dataset, iitb_cropped)
├── notebooks/              # vla_inference.ipynb — single-image walkthrough
├── docs/                   # Experiment notes and demo narration script
├── reports/                # LaTeX thesis reports, DOCX report builder, slides
└── assets/                 # Example images and outputs
```

## Installation

Requirements: Python 3.10+, and an NVIDIA GPU with CUDA is strongly recommended (SAM 2.1 Large + DepthAnythingV2 Large).

```bash
git clone https://github.com/aryangoyal7/DDP-volume-estimation-final.git
cd DDP-volume-estimation-final

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If you need a specific CUDA build of PyTorch, install `torch`/`torchvision` first, then run `pip install -e .`.

Model weights (DepthAnythingV2, GroundingDINO, SAM 2.1) are downloaded automatically from Hugging Face on first run.

### Docker

```bash
docker build -t waste-volume-estimation .
docker run --gpus all -p 7862:7862 waste-volume-estimation
```

## Demo portal

The main deliverable is a Gradio web app:

```bash
python app.py
```

Open **http://localhost:7862**. Two modes are available:

| Mode | What it does |
|---|---|
| **Full-frame** | No segmentation — the whole image is integrated (matches the calibrated report pipeline). |
| **Prompted segmentation** | Type a text prompt (e.g. `waste pile`); LangSAM masks the pile and only that region is integrated. |

Adjustable inputs: camera height, scene dimensions, calibration factor (per-pile presets or the global median CF = 0.86), and bulk density for mass. The app displays the depth map, segmentation mask, height map, and the estimated volume (litres) and mass (kg).

A narration script for recording a demo video is in `docs/demo_script.txt`.

### Conversational CLI

Multi-turn prompting on a single image:

```bash
python scripts/conversational_vla.py \
  --image /path/to/image.png \
  --sam_type sam2.1_hiera_large \
  --gdino_model_id IDEA-Research/grounding-dino-base \
  --real_world_height 1.0 --real_world_width 1.0 \
  --density_kg_per_m3 180
```

Then type prompts (`mixed waste pile`, `plastic bottles`, `cardboard`); use `/save` to export and `/exit` to quit.

## Reproducing the evaluation

All quantitative results in the thesis are produced by the scripts in `evaluation/`:

```bash
bash evaluation/run_all_evals.sh
```

Individual experiments:

```bash
# DepthAnythingV2 full-frame pipeline on the IITB piles
python evaluation/run_iitb_fullframe.py --device cuda:0

# Calibration-factor derivation (calibration/held-out split)
python evaluation/run_iitb_calibration.py --device cuda:0

# SAM-3D pipeline: point cloud + 3 volume estimators + .ply export
python evaluation/run_iitb_sam3d.py --device cuda:0 --prompt "plastic waste"

# Point-cloud comparison (DAv2 vs DINOv3 depth)
python evaluation/run_iitb_pointcloud.py --device cuda:0 --model both
```

Each script writes per-image results (CSV/JSON), visualisations, and aggregate metrics (MAE, RMSE, MAPE, R²) to a subdirectory of `evaluation/`. Ground-truth volumes for the IITB piles are in `evaluation/iitb_gt.json`.

Batch evaluation with externally generated depth maps (`*_depth_raw.png`, 16-bit) is documented in `docs/langsam_dinov3_experiment.md`.

## Datasets

Small image subsets are included for out-of-the-box runs:

- `datasets/spectralwaste/` — SpectralWaste samples (conveyor-belt waste)
- `datasets/zero_dataset/` — miscellaneous waste-pile photos
- `datasets/iitb_cropped/` — cropped IITB white-box pile images

The full IITB solid-waste dataset (piles with weighed ground truth) is documented in `reports/dataset_doc_latex/`.

## Thesis material

- `reports/latex_report/` — main pipeline report (LaTeX)
- `reports/latex_report_sam3d/` — SAM-3D pipeline report
- `reports/latex_report_demo/` — demo write-up
- `reports/slides/` — presentation slides
- `reports/report_docx/` — DOCX report builder (`build_report.py`)

## Acknowledgments

Built on top of these open-source projects:

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM 2](https://github.com/facebookresearch/segment-anything-2)
- [Supervision](https://github.com/roboflow/supervision)

## License

Apache 2.0 — see [LICENSE](LICENSE).
