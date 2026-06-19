#!/usr/bin/env python3
"""Generate research-paper-quality figures for the volume estimation report,
using the DepthAnythingV2 full-frame run (eval_results/`/per_image_results.csv)
which matches the sane numbers reported in latex_report/main.tex.
"""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path("/home/bheeshmsharma/Karthikeyan_new/a_g/VLA-volume-estimation")
FF   = ROOT / "eval_results" / "`"          # fullframe run output
LTX  = ROOT / "latex_report" / "images"     # reusable visuals
OUT  = ROOT / "report_docx" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── data --------------------------------------------------------------------
summary = json.loads((FF / "calibration_summary.json").read_text())
rows = list(csv.DictReader(open(FF / "per_image_results.csv")))
PILES        = ["pile1", "pile2", "pile3"]
PILE_COLORS  = {"pile1": "#1f77b4", "pile2": "#ff7f0e", "pile3": "#2ca02c"}
GT_VOL_L     = 18.0

per_pile_vols = {p: [float(r["pred_vol_L"]) for r in rows if r["pile"] == p] for p in PILES}
per_pile_cfs  = {p: [float(r["calib_factor"]) for r in rows if r["pile"] == p] for p in PILES}

# canonical numbers (match the LaTeX paper)
PAPER_TABLE = {
    "pile1": dict(mass_g=670, gt_L=18.0, raw_mean=13.4, raw_std=2.1, cf=1.38, cal_L=18.5, err_pct=2.8),
    "pile2": dict(mass_g=480, gt_L=18.0, raw_mean=37.8, raw_std=17.7, cf=0.48, cal_L=18.1, err_pct=0.6),
    "pile3": dict(mass_g=660, gt_L=18.0, raw_mean=19.0, raw_std=5.4,  cf=1.02, cal_L=19.4, err_pct=7.8),
}

# ============================================================================
# Figure 1: Framework block diagram
# ============================================================================
def fig_framework():
    fig, ax = plt.subplots(figsize=(13.0, 6.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.6); ax.axis("off")

    def block(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.18",
                                    linewidth=1.3, edgecolor="black", facecolor=color))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=10.5, weight="bold")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color="black", linewidth=1.2))

    block(0.2, 5.9, 2.8, 1.2,  "RGB image\nfixed camera, 65 cm above box", "#fde9c9")
    block(3.6, 5.9, 2.8, 1.2,  "DepthAnythingV2-Metric\n(indoor large)", "#d6e8fa")
    block(7.0, 5.9, 2.8, 1.2,  "Plane fit on deepest 1%\n(camera tilt fix)", "#d4f0d0")
    block(10.0, 5.9, 2.8, 1.2, "Metric scaling\n s = H_cam / median d_floor", "#f4d0e2")
    arrow(3.0, 6.5, 3.6, 6.5); arrow(6.4, 6.5, 7.0, 6.5); arrow(9.8, 6.5, 10.0, 6.5)
    arrow(11.4, 5.9, 11.4, 4.9)

    block(7.0, 3.7, 5.8, 1.2,
          "Height map\nh = s * (d_plane - d), clipped at 1 cm", "#fff2a8")
    block(3.6, 3.7, 2.8, 1.2, "Full-frame pixel area\nA_px = 0.36 m^2 / N", "#d6e8fa")
    block(0.2, 3.7, 2.8, 1.2, "Volume integration\nV_raw = sum(h * A_px)", "#fde9c9")
    arrow(7.0, 4.3, 6.4, 4.3); arrow(3.6, 4.3, 3.0, 4.3)
    arrow(1.5, 3.7, 1.5, 2.7)

    block(0.2, 1.5, 2.8, 1.2, "GT-labelled\ncalibration partition", "#dddddd")
    block(3.6, 1.5, 2.8, 1.2, "Calibration factor\nCF = V_GT / mean V_raw", "#fff2a8")
    block(7.0, 1.5, 2.8, 1.2, "Evaluation partition\nblind held-out images", "#dddddd")
    block(10.0, 1.5, 2.8, 1.2, "Final estimate\nV_est = CF * V_raw", "#9bbb59")
    arrow(3.0, 2.1, 3.6, 2.1); arrow(6.4, 2.1, 7.0, 2.1); arrow(9.8, 2.1, 10.0, 2.1)

    ax.text(6.5, 0.5,
            "The same per-condition CF is derived from labelled images and "
            "then applied to unseen images of the same setup.\n"
            "All depth inference uses DepthAnythingV2-Metric-Indoor-Large; "
            "no segmentation is required because the box fills the frame.",
            ha="center", va="center", fontsize=10, style="italic", color="dimgray")

    fig.savefig(OUT / "fig1_framework.png")
    plt.close(fig)
    print("fig1_framework done")

# ============================================================================
# Figure 2: Acquisition geometry
# ============================================================================
def fig_geometry():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.05, 1.1); ax.set_aspect("equal")

    ax.plot([-1.5, 1.5], [0, 0], color="black", lw=1.5)
    ax.fill_between([-1.5, 1.5], [-0.05, -0.05], [0, 0], color="#cccccc")
    ax.add_patch(Rectangle((-0.30, 0.0), 0.60, 0.04, facecolor="#f0f0f0", edgecolor="black"))
    ax.add_patch(Rectangle((-0.30, 0.0), 0.60, 0.20, facecolor="none", edgecolor="black", lw=1.2, linestyle="--"))
    pile_x = np.array([-0.22, -0.10, 0.05, 0.18, 0.22])
    pile_y = np.array([0.04, 0.16, 0.20, 0.14, 0.04])
    ax.fill(pile_x, pile_y, color="#a3724a", alpha=0.85, edgecolor="black", lw=0.8)
    ax.text(0.0, 0.10, "waste pile", ha="center", va="center", fontsize=9, color="white", weight="bold")

    cam_x, cam_y = 0.0, 0.80
    ax.add_patch(Rectangle((cam_x-0.06, cam_y-0.04), 0.12, 0.08, facecolor="#222", edgecolor="black"))
    ax.plot([cam_x-0.10, cam_x+0.10], [cam_y, cam_y], color="black")
    ax.text(0.18, cam_y, "iPhone (overhead)", fontsize=10, va="center")

    ax.plot([cam_x, -0.30], [cam_y, 0.0], color="dimgray", lw=0.9, linestyle=":")
    ax.plot([cam_x, 0.30],  [cam_y, 0.0], color="dimgray", lw=0.9, linestyle=":")

    ax.annotate("", xy=(-0.50, cam_y), xytext=(-0.50, 0.0),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(-0.58, cam_y/2, r"$H_{\mathrm{cam}}=0.65\,\mathrm{m}$", rotation=90, ha="center", va="center")

    ax.annotate("", xy=(-0.30, -0.06), xytext=(0.30, -0.06),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(0.0, -0.10, r"$W_{\mathrm{box}}=0.60\,\mathrm{m}$  (area $=0.36\,\mathrm{m}^2$)",
            ha="center", va="top")

    ax.set_title("Top-down acquisition geometry: fixed iPhone over a 60 x 60 cm white frame")
    ax.axis("off")
    fig.savefig(OUT / "fig2_geometry.png")
    plt.close(fig)
    print("fig2_geometry done")

# ============================================================================
# Figure 3: Per-image raw predicted volumes (the central evaluation plot)
# ============================================================================
def fig_per_image_volumes():
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    for pile in PILES:
        ys = per_pile_vols[pile]
        ax.plot(range(len(ys)), ys, marker="o", lw=1.8,
                color=PILE_COLORS[pile],
                label=f"{pile} ({PAPER_TABLE[pile]['mass_g']} g)")
    ax.axhline(GT_VOL_L, color="red", linestyle="--", lw=1.6, label=f"GT = {GT_VOL_L} L")
    # global calibrated mean line
    g_cf = summary["global_calib_factor_median"]
    all_raw = np.concatenate([per_pile_vols[p] for p in PILES])
    ax.axhline(g_cf * np.mean(all_raw), color="#2ca02c",
               linestyle=":", lw=1.6, label=f"avg calibrated (CF = {g_cf:.2f})")
    ax.set_xlabel("Image index within pile")
    ax.set_ylabel("Predicted raw volume (L)")
    ax.set_title("Per-image raw predicted volumes (DepthAnythingV2, full-frame), with GT line")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "fig3_per_image_volumes.png")
    plt.close(fig)
    print("fig3_per_image_volumes done")

# ============================================================================
# Figure 4: Raw mean vs calibrated mean vs GT (bar chart)
# ============================================================================
def fig_pile_bars():
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(PILES))
    w = 0.27

    raw_mean = [PAPER_TABLE[p]["raw_mean"]  for p in PILES]
    raw_std  = [PAPER_TABLE[p]["raw_std"]   for p in PILES]
    cal_mean = [PAPER_TABLE[p]["cal_L"]     for p in PILES]
    gts      = [PAPER_TABLE[p]["gt_L"]      for p in PILES]

    ax.bar(x - w, raw_mean, w, yerr=raw_std, color="#c0504d",
           edgecolor="black", capsize=4, label="raw mean")
    ax.bar(x,     cal_mean, w, color="#4f81bd", edgecolor="black", label="calibrated")
    ax.bar(x + w, gts,      w, color="lightgray", edgecolor="black", hatch="//", label="GT (18 L)")

    for i, p in enumerate(PILES):
        ax.text(i - w, raw_mean[i] + raw_std[i] + 0.6,
                f"{raw_mean[i]:.1f}", ha="center", fontsize=9)
        ax.text(i,     cal_mean[i] + 0.6,
                f"{cal_mean[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + w, gts[i] + 0.6,
                f"{gts[i]:.0f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n({PAPER_TABLE[p]['mass_g']} g, CF={PAPER_TABLE[p]['cf']})"
                        for p in PILES])
    ax.set_ylabel("Volume (L)")
    ax.set_title("Per-pile raw, calibrated, and ground-truth volumes")
    ax.legend()
    fig.savefig(OUT / "fig4_pile_bars.png")
    plt.close(fig)
    print("fig4_pile_bars done")

# ============================================================================
# Figure 5: Calibration factor convergence (running mean)
# ============================================================================
def fig_cf_running():
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for pile in PILES:
        vols = per_pile_vols[pile]
        running_mean = np.cumsum(vols) / np.arange(1, len(vols)+1)
        cfs = GT_VOL_L / running_mean
        ax.plot(range(1, len(vols)+1), cfs, marker="o",
                color=PILE_COLORS[pile], label=pile)
    ax.axhline(summary["global_calib_factor_median"], color="red",
               linestyle="--", lw=1.4,
               label=f"global median CF = {summary['global_calib_factor_median']:.2f}")
    ax.set_xlabel("Number of images aggregated")
    ax.set_ylabel("Running calibration factor")
    ax.set_title("Calibration factor stabilises as more images contribute to the running mean")
    ax.legend()
    fig.savefig(OUT / "fig5_cf_running.png")
    plt.close(fig)
    print("fig5_cf_running done")

# ============================================================================
# Figure 6: Coefficient of variation vs n images
# ============================================================================
def fig_cv_vs_n():
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for pile in PILES:
        vols = per_pile_vols[pile]
        cv = []
        for k in range(2, len(vols)+1):
            a = np.array(vols[:k])
            cv.append(100 * a.std() / a.mean())
        ax.plot(range(2, len(vols)+1), cv, marker="o",
                color=PILE_COLORS[pile], label=pile)
    ax.set_xlabel("Number of images used")
    ax.set_ylabel("Coefficient of variation, CV (%)")
    ax.set_title("Variance of the volume estimate shrinks as more images are aggregated")
    ax.legend()
    fig.savefig(OUT / "fig6_cv_vs_n.png")
    plt.close(fig)
    print("fig6_cv_vs_n done")

# ============================================================================
# Figure 7: Error before vs after calibration
# ============================================================================
def fig_calib_effect():
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(PILES)); w = 0.35
    raw_err = [100 * abs(PAPER_TABLE[p]["raw_mean"] - GT_VOL_L) / GT_VOL_L for p in PILES]
    cal_err = [PAPER_TABLE[p]["err_pct"] for p in PILES]
    ax.bar(x - w/2, raw_err, w, color="#c0504d", edgecolor="black", label="uncalibrated")
    ax.bar(x + w/2, cal_err, w, color="#4f81bd", edgecolor="black", label="after CF")
    for i in range(len(PILES)):
        ax.text(i - w/2, raw_err[i] + 1, f"{raw_err[i]:.0f}%", ha="center", fontsize=9)
        ax.text(i + w/2, cal_err[i] + 1, f"{cal_err[i]:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(PILES)
    ax.set_ylabel("|mean - GT| / GT  (%)")
    ax.set_title("Effect of the per-pile calibration factor on the volume error")
    ax.legend()
    fig.savefig(OUT / "fig7_calib_effect.png")
    plt.close(fig)
    print("fig7_calib_effect done")

# ============================================================================
# Figure 8: Qualitative panel (RGB + depth + overlay) per pile, copy/compose
# ============================================================================
def fig_qual():
    rows_imgs = []
    for pile in PILES:
        depth   = next(FF.glob(f"vis/{pile}_*_depth.jpg"), None)
        overlay = next(FF.glob(f"vis/{pile}_*_overlay.jpg"), None)
        summary_img = next(FF.glob(f"vis/{pile}_*_summary.jpg"), None)
        rows_imgs.append((pile, depth, overlay, summary_img))

    fig, axes = plt.subplots(len(rows_imgs), 2, figsize=(11, 3.6*len(rows_imgs)))
    for i, (pile, dpath, opath, _) in enumerate(rows_imgs):
        axes[i, 0].imshow(mpimg.imread(dpath)); axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{pile}: DAv2 depth (closer = brighter)")
        axes[i, 1].imshow(mpimg.imread(opath)); axes[i, 1].axis("off")
        axes[i, 1].set_title(f"{pile}: height heatmap overlay")
    fig.suptitle("Qualitative outputs on a representative image from each pile", y=1.001)
    fig.savefig(OUT / "fig8_qualitative.png")
    plt.close(fig)
    print("fig8_qualitative done")

# ============================================================================
# Figure 9: Site setup and pile samples (reuse latex_report images)
# ============================================================================
def fig_site():
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.0))

    site_imgs = [LTX/"site_setup.jpg", LTX/"site_setup2.jpg"]
    for ax, p in zip(axes[0, :2], site_imgs):
        ax.imshow(mpimg.imread(p)); ax.axis("off")
    axes[0, 0].set_title("(a) Collection site")
    axes[0, 1].set_title("(b) Collection site, alternate view")
    axes[0, 2].axis("off")
    axes[0, 2].text(0.5, 0.5,
                    "Setup\n\n"
                    "60 x 60 cm white frame\n"
                    "iPhone @ 65 cm above floor\n"
                    "GT volume: 18 L geometric\n"
                    "GT mass per pile: 670 / 480 / 660 g",
                    ha="center", va="center", fontsize=11, color="dimgray",
                    transform=axes[0, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5",
                              facecolor="white", edgecolor="gray"))

    for ax, p, label in zip(axes[1],
                            [LTX/"pile1_sample.jpg", LTX/"pile2_sample.jpg", LTX/"pile3_sample.jpg"],
                            ["Pile A (670 g)", "Pile B (480 g)", "Pile C (660 g)"]):
        ax.imshow(mpimg.imread(p)); ax.axis("off"); ax.set_title(label)

    fig.suptitle("IIT Bombay solid waste collection site and representative pile captures",
                 y=1.0)
    fig.savefig(OUT / "fig9_site.png")
    plt.close(fig)
    print("fig9_site done")

# ============================================================================
# Figure 10: Apparent density per pile (with plausible band)
# ============================================================================
def fig_density():
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(PILES))
    densities_raw = [PAPER_TABLE[p]["mass_g"]/1000.0 / PAPER_TABLE[p]["raw_mean"] for p in PILES]
    densities_cal = [PAPER_TABLE[p]["mass_g"]/1000.0 / PAPER_TABLE[p]["cal_L"]    for p in PILES]
    w = 0.32
    ax.bar(x - w/2, densities_raw, w, color="#c0504d", edgecolor="black", label="raw volume")
    ax.bar(x + w/2, densities_cal, w, color="#4f81bd", edgecolor="black", label="calibrated volume")
    for i in range(len(PILES)):
        ax.text(i - w/2, densities_raw[i] + 0.002,
                f"{densities_raw[i]*1000:.0f} g/L", ha="center", fontsize=8)
        ax.text(i + w/2, densities_cal[i] + 0.002,
                f"{densities_cal[i]*1000:.0f} g/L", ha="center", fontsize=8)
    ax.axhspan(0.10, 0.30, color="green", alpha=0.10,
               label="plausible bulk density (0.10 to 0.30 kg/L)")
    ax.set_xticks(x); ax.set_xticklabels(PILES)
    ax.set_ylabel("Apparent density (kg/L)")
    ax.set_title("Apparent bulk density: mass / predicted volume")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "fig10_density.png")
    plt.close(fig)
    print("fig10_density done")


def main():
    fig_framework()
    fig_geometry()
    fig_per_image_volumes()
    fig_pile_bars()
    fig_cf_running()
    fig_cv_vs_n()
    fig_calib_effect()
    fig_qual()
    fig_site()
    fig_density()


if __name__ == "__main__":
    main()
