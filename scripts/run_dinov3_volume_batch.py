#!/usr/bin/env python3
"""
Batch depth-only volume estimation using DINOv3.

Pipeline:
1. Traverse images in --input_dir (recursive).
2. Predict depth map using DINOv3.
3. Convert depth to height-above-ground map.
4. Estimate volume from per-pixel height and known scene area.
5. Save per-image artifacts and a CSV/JSON summary.
"""

import argparse
import csv
import json
import os
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_device(device_name: str, use_cpu: bool) -> torch.device:
    if use_cpu:
        return torch.device("cpu")
    if device_name and device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_dinov3_depther(
    repo_dir: str,
    github_repo: str,
    depther_weights: str,
    backbone_weights: str,
    min_depth: float,
    max_depth: float,
    device: torch.device,
    init_on_cpu_first: bool,
    gpu_half: bool,
):
    @contextmanager
    def no_cuda_calls(enabled: bool):
        if not enabled:
            yield
            return

        original_module_cuda = torch.nn.Module.cuda
        original_tensor_cuda = torch.Tensor.cuda

        def _module_cuda_noop(self, device=None):
            return self

        def _tensor_cuda_noop(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
            return self

        torch.nn.Module.cuda = _module_cuda_noop
        torch.Tensor.cuda = _tensor_cuda_noop
        try:
            yield
        finally:
            torch.nn.Module.cuda = original_module_cuda
            torch.Tensor.cuda = original_tensor_cuda

    force_cpu_load = device.type == "cpu" or init_on_cpu_first
    with no_cuda_calls(force_cpu_load):
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
                github_repo,
                "dinov3_vit7b16_dd",
                source="github",
                pretrained=False,
                weights=depther_weights,
                backbone_weights=backbone_weights,
                depth_range=(min_depth, max_depth),
            )
    depther.eval()
    if device.type == "cuda" and gpu_half:
        depther = depther.half()
        depther.autocast_ctx = partial(torch.autocast, device_type="cuda", dtype=torch.float16, enabled=True)
    elif device.type == "cuda":
        depther.autocast_ctx = partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16, enabled=True)
    else:
        depther.autocast_ctx = partial(torch.autocast, device_type="cpu", enabled=True)
    depther = depther.to(device)
    return depther


def make_transform(resize_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def estimate_depth_with_dinov3(
    image_pil: Image.Image,
    depther: torch.nn.Module,
    device: torch.device,
    resize_size: int,
) -> np.ndarray:
    transform = make_transform(resize_size)
    model_dtype = next(depther.parameters()).dtype
    x = transform(image_pil)[None].to(device=device, dtype=model_dtype)
    out_h, out_w = image_pil.size[1], image_pil.size[0]

    with torch.inference_mode():
        if device.type == "cuda":
            autocast_dtype = torch.float16 if model_dtype == torch.float16 else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
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
    return depth_map


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

    depth_scale = 1.0
    if reference_height_cm > 0:
        positive = height_map[height_map > 0]
        if positive.size > 0:
            peak_height = float(np.percentile(positive, 99.0))
            if peak_height > 1e-9:
                depth_scale = (reference_height_cm / 100.0) / peak_height
                height_map = height_map * depth_scale
                ground_depth = ground_depth * depth_scale
    return height_map, ground_depth, depth_scale


def compute_volume_metrics(height_map: np.ndarray, scene_area_m2: float, min_height_cm: float) -> Dict[str, float]:
    h, w = height_map.shape
    area_per_pixel_m2 = float(scene_area_m2) / float(h * w)
    min_height_m = float(min_height_cm) / 100.0

    object_mask = height_map > min_height_m
    object_heights = height_map[object_mask]

    if object_heights.size == 0:
        return {
            "coverage_percent": 0.0,
            "object_area_m2": 0.0,
            "mean_height_cm": 0.0,
            "max_height_cm": 0.0,
            "volume_m3": 0.0,
            "volume_liters": 0.0,
        }

    volume_m3 = float(np.sum(object_heights) * area_per_pixel_m2)
    return {
        "coverage_percent": float(np.mean(object_mask) * 100.0),
        "object_area_m2": float(np.mean(object_mask) * scene_area_m2),
        "mean_height_cm": float(np.mean(object_heights) * 100.0),
        "max_height_cm": float(np.max(object_heights) * 100.0),
        "volume_m3": volume_m3,
        "volume_liters": volume_m3 * 1000.0,
    }


def depth_to_uint16(depth_map: np.ndarray, scale_factor: float = 10000.0) -> np.ndarray:
    clipped = np.clip(depth_map, a_min=0.0, a_max=np.iinfo(np.uint16).max / scale_factor)
    return (clipped * scale_factor).astype(np.uint16)


def colormap_depth(depth_map: np.ndarray) -> np.ndarray:
    valid = depth_map > 0
    vis = np.zeros_like(depth_map, dtype=np.float32)
    if np.any(valid):
        depth_valid = depth_map[valid]
        d_lo = float(np.percentile(depth_valid, 2.0))
        d_hi = float(np.percentile(depth_valid, 98.0))
        vis = np.clip((depth_map - d_lo) / max(d_hi - d_lo, 1e-9), 0.0, 1.0)
    cmap = plt.get_cmap("viridis_r")
    return (cmap(vis)[..., :3] * 255).astype(np.uint8)


def colormap_height(height_map: np.ndarray, min_height_cm: float) -> np.ndarray:
    mask = height_map > (min_height_cm / 100.0)
    vmax = float(np.max(height_map[mask])) if np.any(mask) else 1.0
    norm = np.clip(height_map / (vmax + 1e-9), 0.0, 1.0)
    cmap = plt.get_cmap("hot")
    colored = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    return np.where(mask[..., None], colored, 0).astype(np.uint8)


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float,
    nonzero_only: bool,
) -> np.ndarray:
    base = image_rgb.astype(np.float32)
    heat = heatmap_rgb.astype(np.float32)
    out = base.copy()
    if nonzero_only:
        mask = np.any(heatmap_rgb > 0, axis=-1)
    else:
        mask = np.ones(image_rgb.shape[:2], dtype=bool)
    out[mask] = (1.0 - alpha) * base[mask] + alpha * heat[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def find_images(input_dir: Path) -> list[Path]:
    images = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DINOv3 depth + volume estimation")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image folder (recursive)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder")
    parser.add_argument("--scene_area_m2", type=float, required=True, help="Known scene area in square meters")
    parser.add_argument("--reference_height_cm", type=float, default=0.0, help="Reference max object height in cm")
    parser.add_argument("--min_height_cm", type=float, default=1.0, help="Minimum object height threshold in cm")
    parser.add_argument("--ground_percentile", type=float, default=99.5, help="Ground percentile in [90, 100]")
    parser.add_argument("--resize", type=int, default=int(os.getenv("DINOV3_RESIZE", "896")))
    parser.add_argument("--repo_dir", type=str, default=os.getenv("DINOV3_REPO_DIR", ""))
    parser.add_argument("--github_repo", type=str, default=os.getenv("DINOV3_GITHUB_REPO", "facebookresearch/dinov3"))
    parser.add_argument("--depther_weights", type=str, default=os.getenv("DINOV3_DEPTHER_WEIGHTS", "SYNTHMIX"))
    parser.add_argument("--backbone_weights", type=str, default=os.getenv("DINOV3_BACKBONE_WEIGHTS", "LVD1689M"))
    parser.add_argument("--min_depth", type=float, default=float(os.getenv("DINOV3_MIN_DEPTH", "0.85")))
    parser.add_argument("--max_depth", type=float, default=float(os.getenv("DINOV3_MAX_DEPTH", "1.0")))
    parser.add_argument("--device", type=str, default=os.getenv("DINOV3_DEVICE", "auto"))
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help="Alpha for heatmap overlays on original image [0..1]",
    )
    parser.add_argument(
        "--init_on_cpu_first",
        type=int,
        choices=[0, 1],
        default=1,
        help="Initialize depther on CPU first to avoid GPU-OOM during construction",
    )
    parser.add_argument(
        "--gpu_half",
        type=int,
        choices=[0, 1],
        default=1,
        help="When using CUDA, cast model to float16 before moving to GPU",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    artifacts_dir = output_dir / "artifacts"
    depth_raw_dir = artifacts_dir / "depth_raw"
    depth_vis_dir = artifacts_dir / "depth_vis"
    height_vis_dir = artifacts_dir / "height_vis"
    depth_overlay_dir = artifacts_dir / "depth_overlay"
    height_overlay_dir = artifacts_dir / "height_overlay"
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_raw_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_dir.mkdir(parents=True, exist_ok=True)
    height_vis_dir.mkdir(parents=True, exist_ok=True)
    depth_overlay_dir.mkdir(parents=True, exist_ok=True)
    height_overlay_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device, use_cpu=args.use_cpu)
    print(f"[info] device={device}")
    print("[info] loading DINOv3 depther...")
    t_model_start = time.perf_counter()
    depther = get_dinov3_depther(
        repo_dir=args.repo_dir,
        github_repo=args.github_repo,
        depther_weights=args.depther_weights,
        backbone_weights=args.backbone_weights,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        device=device,
        init_on_cpu_first=bool(args.init_on_cpu_first),
        gpu_half=bool(args.gpu_half),
    )
    t_model = time.perf_counter() - t_model_start
    model_dtype = str(next(depther.parameters()).dtype)
    print(f"[info] model loaded in {t_model:.2f}s | dtype={model_dtype}")

    image_paths = find_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {input_dir}")
    print(f"[info] found {len(image_paths)} images")

    rows: list[dict] = []
    total_start = time.perf_counter()
    for idx, image_path in enumerate(image_paths, start=1):
        rel = image_path.relative_to(input_dir)
        rel_key = str(rel.with_suffix("")).replace(os.sep, "__")
        t0 = time.perf_counter()

        try:
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.array(image_pil, dtype=np.uint8)
            depth_map = estimate_depth_with_dinov3(
                image_pil=image_pil,
                depther=depther,
                device=device,
                resize_size=args.resize,
            )
            height_map, ground_depth, depth_scale = estimate_height_map(
                depth_map=depth_map,
                ground_percentile=args.ground_percentile,
                reference_height_cm=args.reference_height_cm,
            )
            metrics = compute_volume_metrics(
                height_map=height_map,
                scene_area_m2=args.scene_area_m2,
                min_height_cm=args.min_height_cm,
            )

            depth_raw_path = depth_raw_dir / f"{rel_key}_depth_raw.png"
            Image.fromarray(depth_to_uint16(depth_map)).save(depth_raw_path)

            depth_vis = colormap_depth(depth_map)
            depth_vis_path = depth_vis_dir / f"{rel_key}_depth_vis.png"
            Image.fromarray(depth_vis).save(depth_vis_path)

            height_vis = colormap_height(height_map, args.min_height_cm)
            height_vis_path = height_vis_dir / f"{rel_key}_height_vis.png"
            Image.fromarray(height_vis).save(height_vis_path)

            depth_overlay = overlay_heatmap_on_image(
                image_rgb=image_np,
                heatmap_rgb=depth_vis,
                alpha=float(args.overlay_alpha),
                nonzero_only=False,
            )
            depth_overlay_path = depth_overlay_dir / f"{rel_key}_depth_overlay.png"
            Image.fromarray(depth_overlay).save(depth_overlay_path)

            height_overlay = overlay_heatmap_on_image(
                image_rgb=image_np,
                heatmap_rgb=height_vis,
                alpha=float(args.overlay_alpha),
                nonzero_only=True,
            )
            height_overlay_path = height_overlay_dir / f"{rel_key}_height_overlay.png"
            Image.fromarray(height_overlay).save(height_overlay_path)

            elapsed = time.perf_counter() - t0
            row = {
                "image_path": str(image_path),
                "depth_raw_path": str(depth_raw_path),
                "depth_vis_path": str(depth_vis_path),
                "height_vis_path": str(height_vis_path),
                "depth_overlay_path": str(depth_overlay_path),
                "height_overlay_path": str(height_overlay_path),
                "scene_area_m2": float(args.scene_area_m2),
                "reference_height_cm": float(args.reference_height_cm),
                "min_height_cm": float(args.min_height_cm),
                "ground_percentile": float(args.ground_percentile),
                "ground_depth_scaled": float(ground_depth),
                "depth_scale": float(depth_scale),
                **metrics,
                "runtime_sec": float(elapsed),
            }
            rows.append(row)
            print(
                f"[{idx}/{len(image_paths)}] {rel} | volume={row['volume_liters']:.3f} L | "
                f"mean_h={row['mean_height_cm']:.2f} cm | t={elapsed:.2f}s"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"[{idx}/{len(image_paths)}] {rel} | ERROR after {elapsed:.2f}s: {exc}")
            rows.append(
                {
                    "image_path": str(image_path),
                    "error": str(exc),
                    "runtime_sec": float(elapsed),
                }
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - total_start
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "summary.json"

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = [r for r in rows if "error" not in r]
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_images": len(image_paths),
        "num_success": len(successful),
        "num_failed": len(rows) - len(successful),
        "device": str(device),
        "model_dtype": model_dtype,
        "scene_area_m2": float(args.scene_area_m2),
        "reference_height_cm": float(args.reference_height_cm),
        "min_height_cm": float(args.min_height_cm),
        "ground_percentile": float(args.ground_percentile),
        "model_load_sec": float(t_model),
        "total_runtime_sec": float(total_elapsed),
        "avg_runtime_sec_per_image": float(np.mean([r["runtime_sec"] for r in successful])) if successful else None,
        "mean_volume_liters": float(np.mean([r["volume_liters"] for r in successful])) if successful else None,
        "results_csv": str(csv_path),
    }
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] csv={csv_path}")
    print(f"[done] summary={json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
