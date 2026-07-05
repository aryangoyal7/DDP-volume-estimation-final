#!/usr/bin/env python3
"""
Depth-only VLA demo:
- DINOv3 depth estimation from RGB
- Height-map derivation from depth
- Volume estimation using scene area and per-pixel heights
"""

import os
import time
from functools import lru_cache
from typing import Dict, Tuple

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gradio_tmp")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    register_heif_opener = None


PRESET_SCENE_AREA_M2 = float(os.getenv("SCENE_AREA_M2", "1.0"))
PRESET_REFERENCE_HEIGHT_CM = float(os.getenv("REFERENCE_HEIGHT_CM", "0.0"))
PRESET_MIN_HEIGHT_CM = float(os.getenv("MIN_HEIGHT_CM", "1.0"))
PRESET_GROUND_PERCENTILE = float(os.getenv("GROUND_PERCENTILE", "99.5"))
PRESET_PORT = int(os.getenv("DEPTH_ONLY_PORT", "7862"))

PRESET_DINOV3_GITHUB_REPO = os.getenv("DINOV3_GITHUB_REPO", "facebookresearch/dinov3")
PRESET_DINOV3_DEPTHER_WEIGHTS = os.getenv("DINOV3_DEPTHER_WEIGHTS", "SYNTHMIX")
PRESET_DINOV3_BACKBONE_WEIGHTS = os.getenv("DINOV3_BACKBONE_WEIGHTS", "LVD1689M")
PRESET_DINOV3_RESIZE = int(os.getenv("DINOV3_RESIZE", "896"))
PRESET_DINOV3_MIN_DEPTH = float(os.getenv("DINOV3_MIN_DEPTH", "0.85"))
PRESET_DINOV3_MAX_DEPTH = float(os.getenv("DINOV3_MAX_DEPTH", "1.0"))
PRESET_DINOV3_USE_CPU = os.getenv("DINOV3_USE_CPU", "0") == "1"
PRESET_DINOV3_DEVICE = os.getenv("DINOV3_DEVICE", "auto")
PRESET_DINOV3_WARMUP_ON_STARTUP = os.getenv("DINOV3_WARMUP_ON_STARTUP", "1") == "1"


def _autodetect_dinov3_repo() -> str:
    candidates = [
        os.getenv("DINOV3_REPO_DIR", ""),
        os.path.abspath(os.path.dirname(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isfile(os.path.join(candidate, "hubconf.py")) and os.path.isdir(os.path.join(candidate, "dinov3")):
            return candidate
    return ""


PRESET_DINOV3_REPO_DIR = _autodetect_dinov3_repo()


def _resolve_device(device_name: str, use_cpu: bool) -> torch.device:
    if use_cpu:
        return torch.device("cpu")
    if device_name and device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@lru_cache(maxsize=2)
def get_dinov3_depther(
    repo_dir: str,
    depther_weights: str,
    backbone_weights: str,
    min_depth: float,
    max_depth: float,
    use_cpu: bool,
    device_name: str,
):
    device = _resolve_device(device_name, use_cpu=use_cpu)
    if repo_dir:
        depther = torch.hub.load(
            repo_dir,
            "dinov3_vit7b16_dd",
            source="local",
            pretrained=False,
            weights=depther_weights,
            backbone_weights=backbone_weights,
            depth_range=(min_depth, max_depth),
        )
    else:
        depther = torch.hub.load(
            PRESET_DINOV3_GITHUB_REPO,
            "dinov3_vit7b16_dd",
            source="github",
            pretrained=False,
            weights=depther_weights,
            backbone_weights=backbone_weights,
            depth_range=(min_depth, max_depth),
        )
    depther.eval()
    depther = depther.to(device)
    return depther, device


def make_transform(resize_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _extract_file_path(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    if isinstance(uploaded_file, str):
        return uploaded_file
    if hasattr(uploaded_file, "name"):
        return str(uploaded_file.name)
    raise gr.Error("Unsupported upload object from Gradio. Please upload a valid image file.")


def load_image_rgb(path: str) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def estimate_depth_with_dinov3(image_pil: Image.Image) -> Tuple[np.ndarray, torch.device]:
    depther, device = get_dinov3_depther(
        repo_dir=PRESET_DINOV3_REPO_DIR,
        depther_weights=PRESET_DINOV3_DEPTHER_WEIGHTS,
        backbone_weights=PRESET_DINOV3_BACKBONE_WEIGHTS,
        min_depth=PRESET_DINOV3_MIN_DEPTH,
        max_depth=PRESET_DINOV3_MAX_DEPTH,
        use_cpu=PRESET_DINOV3_USE_CPU,
        device_name=PRESET_DINOV3_DEVICE,
    )
    transform = make_transform(PRESET_DINOV3_RESIZE)
    x = transform(image_pil)[None].to(device)
    out_h, out_w = image_pil.size[1], image_pil.size[0]

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                depth_pred = depther(x)
        else:
            depth_pred = depther(x)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.float(),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )

    depth_map = depth_pred[0, 0].detach().cpu().numpy()
    del x
    del depth_pred
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return depth_map, device


def preview_uploaded_image(uploaded_file):
    image_path = _extract_file_path(uploaded_file)
    if not image_path:
        return None
    image_pil = load_image_rgb(image_path)
    return np.array(image_pil)


def estimate_height_map(
    depth_map: np.ndarray,
    ground_percentile: float,
    reference_height_cm: float,
) -> Tuple[np.ndarray, float, float]:
    valid = depth_map > 0
    valid_depth = depth_map[valid]
    if valid_depth.size == 0:
        return np.zeros_like(depth_map), 0.0, 1.0

    if ground_percentile < 100.0:
        ground_depth = float(np.percentile(valid_depth, ground_percentile))
    else:
        ground_depth = float(np.max(valid_depth))

    height_map = ground_depth - depth_map
    height_map[~valid] = 0.0
    height_map = np.clip(height_map, a_min=0.0, a_max=None)

    scale = 1.0
    if reference_height_cm and reference_height_cm > 0:
        positive = height_map[height_map > 0]
        if positive.size > 0:
            peak_height = float(np.percentile(positive, 99.0))
            if peak_height > 1e-9:
                scale = (reference_height_cm / 100.0) / peak_height
                height_map = height_map * scale
                ground_depth = ground_depth * scale

    return height_map, ground_depth, scale


def compute_volume_metrics(
    height_map: np.ndarray,
    scene_area_m2: float,
    min_height_cm: float,
) -> Dict[str, float | np.ndarray]:
    h, w = height_map.shape
    area_per_pixel_m2 = float(scene_area_m2) / float(h * w)
    min_height_m = float(min_height_cm) / 100.0
    object_mask = height_map > min_height_m
    object_heights = height_map[object_mask]

    if object_heights.size == 0:
        return {
            "object_mask": object_mask,
            "coverage_percent": 0.0,
            "object_area_m2": 0.0,
            "mean_height_cm": 0.0,
            "max_height_cm": 0.0,
            "volume_m3": 0.0,
            "volume_liters": 0.0,
        }

    volume_m3 = float(np.sum(object_heights) * area_per_pixel_m2)
    coverage = float(np.mean(object_mask) * 100.0)
    object_area_m2 = float(np.mean(object_mask) * scene_area_m2)
    mean_height_cm = float(np.mean(object_heights) * 100.0)
    max_height_cm = float(np.max(object_heights) * 100.0)
    return {
        "object_mask": object_mask,
        "coverage_percent": coverage,
        "object_area_m2": object_area_m2,
        "mean_height_cm": mean_height_cm,
        "max_height_cm": max_height_cm,
        "volume_m3": volume_m3,
        "volume_liters": volume_m3 * 1000.0,
    }


def colormap_depth(depth_map: np.ndarray) -> np.ndarray:
    valid = depth_map > 0
    vis = np.zeros_like(depth_map, dtype=np.float32)
    if np.any(valid):
        depth_valid = depth_map[valid]
        d_lo = float(np.percentile(depth_valid, 2.0))
        d_hi = float(np.percentile(depth_valid, 98.0))
        denom = max(d_hi - d_lo, 1e-9)
        vis = np.clip((depth_map - d_lo) / denom, 0.0, 1.0)
    cmap = plt.get_cmap("viridis_r")
    return (cmap(vis)[..., :3] * 255).astype(np.uint8)


def colormap_height(height_map: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    vmax = float(np.max(height_map[object_mask])) if np.any(object_mask) else 1.0
    norm = np.clip(height_map / (vmax + 1e-9), 0.0, 1.0)
    cmap = plt.get_cmap("hot")
    colored = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    return np.where(object_mask[..., None], colored, 0).astype(np.uint8)


def run_app(uploaded_file, scene_area_m2, reference_height_cm, min_height_cm, ground_percentile):
    t0 = time.perf_counter()
    image_path = _extract_file_path(uploaded_file)
    if not image_path:
        raise gr.Error("Please provide an input image.")
    if scene_area_m2 is None or float(scene_area_m2) <= 0:
        raise gr.Error("Scene area must be > 0 m^2.")
    if min_height_cm is None or float(min_height_cm) < 0:
        raise gr.Error("Minimum height must be >= 0 cm.")
    if ground_percentile is None or not (90.0 <= float(ground_percentile) <= 100.0):
        raise gr.Error("Ground percentile must be between 90 and 100.")

    image_pil = load_image_rgb(image_path)
    t_load = time.perf_counter()
    depth_map, device = estimate_depth_with_dinov3(image_pil)
    t_depth = time.perf_counter()
    height_map, ground_depth, depth_scale = estimate_height_map(
        depth_map=depth_map,
        ground_percentile=float(ground_percentile),
        reference_height_cm=float(reference_height_cm),
    )
    t_height = time.perf_counter()
    metrics = compute_volume_metrics(
        height_map=height_map,
        scene_area_m2=float(scene_area_m2),
        min_height_cm=float(min_height_cm),
    )
    t_metrics = time.perf_counter()

    depth_vis = colormap_depth(depth_map)
    height_vis = colormap_height(height_map, metrics["object_mask"])
    msg = (
        "### Depth + Volume (DINOv3 only)\n"
        f"- Device: `{device}`\n"
        f"- Scene area used: `{float(scene_area_m2):.6f} m^2`\n"
        f"- Ground percentile: `{float(ground_percentile):.2f}`\n"
        f"- Reference height input: `{float(reference_height_cm):.2f} cm` (0 = no calibration)\n"
        f"- Height threshold: `{float(min_height_cm):.2f} cm`\n"
        f"- Object coverage: `{metrics['coverage_percent']:.2f}%`\n"
        f"- Object area estimate: `{metrics['object_area_m2']:.6f} m^2`\n"
        f"- Mean/Max height: `{metrics['mean_height_cm']:.2f}/{metrics['max_height_cm']:.2f} cm`\n"
        f"- Estimated volume: `{metrics['volume_liters']:.4f} L` (`{metrics['volume_m3']:.6f} m^3`)\n"
        f"- Ground depth (scaled): `{ground_depth:.4f}`\n"
        f"- Depth scale factor: `{depth_scale:.4f}`\n"
        f"- DINOv3 source: `{'local repo' if PRESET_DINOV3_REPO_DIR else PRESET_DINOV3_GITHUB_REPO}`\n"
        f"- Timing: load `{(t_load - t0):.2f}s`, depth `{(t_depth - t_load):.2f}s`, "
        f"volume `{(t_metrics - t_height):.2f}s`, total `{(t_metrics - t0):.2f}s`\n"
    )
    return depth_vis, height_vis, msg


def warmup_model_once() -> None:
    if not PRESET_DINOV3_WARMUP_ON_STARTUP:
        return

    start = time.perf_counter()
    depther, device = get_dinov3_depther(
        repo_dir=PRESET_DINOV3_REPO_DIR,
        depther_weights=PRESET_DINOV3_DEPTHER_WEIGHTS,
        backbone_weights=PRESET_DINOV3_BACKBONE_WEIGHTS,
        min_depth=PRESET_DINOV3_MIN_DEPTH,
        max_depth=PRESET_DINOV3_MAX_DEPTH,
        use_cpu=PRESET_DINOV3_USE_CPU,
        device_name=PRESET_DINOV3_DEVICE,
    )
    dummy = torch.zeros((1, 3, 224, 224), device=device, dtype=torch.float32)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = depther(dummy)
            torch.cuda.synchronize(device)
        else:
            _ = depther(dummy)
    elapsed = time.perf_counter() - start
    print(f"[startup] DINOv3 preloaded on {device} in {elapsed:.2f}s")


with gr.Blocks(title="Depth + Volume (DINOv3 only)") as blocks:
    gr.Markdown(
        f"""
# Depth + Volume (DINOv3 only)

This demo removes segmentation and uses only DINOv3 depth to estimate volume.

Inputs:
- RGB image
- Scene area in m^2
- Optional reference height in cm (for scale calibration)
- Minimum height threshold in cm (filters floor noise)
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.File(type="file", file_types=["image"], label="Input image (JPG/PNG/HEIC)")
            scene_area_input = gr.Number(value=PRESET_SCENE_AREA_M2, label="Scene area (m^2)")
            reference_height_input = gr.Number(
                value=PRESET_REFERENCE_HEIGHT_CM,
                label="Reference max height (cm, optional, 0 disables)",
            )
            min_height_input = gr.Number(value=PRESET_MIN_HEIGHT_CM, label="Minimum height threshold (cm)")
            ground_percentile_input = gr.Slider(
                minimum=90.0,
                maximum=100.0,
                value=PRESET_GROUND_PERCENTILE,
                step=0.1,
                label="Ground percentile",
            )
            run_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=1):
            preview_output = gr.Image(type="numpy", label="Uploaded image preview")
            depth_output = gr.Image(type="numpy", label="Predicted depth map")
            height_output = gr.Image(type="numpy", label="Height map used for volume")
            metrics_output = gr.Markdown(label="Metrics")

    image_input.change(fn=preview_uploaded_image, inputs=[image_input], outputs=[preview_output])

    run_btn.click(
        fn=run_app,
        inputs=[
            image_input,
            scene_area_input,
            reference_height_input,
            min_height_input,
            ground_percentile_input,
        ],
        outputs=[depth_output, height_output, metrics_output],
    )


if __name__ == "__main__":
    warmup_model_once()
    blocks.launch(server_name="0.0.0.0", server_port=PRESET_PORT, share=False)
