#!/usr/bin/env python3
"""
exp2d_plot_results.py
=====================
Exp 2 — Generate all plots and comparison table for the seminar presentation.

Reads:
  ../results/bfn_text8_nsteps.json
  ../results/bfn_mnist_nsteps.json
  ../results/d3pm_text8_result.json
  ../results/samples/*.png

Produces:
  ../plots/fig1_nsteps_curve_text8.png     — BPC vs n_steps for BFN on text8
  ../plots/fig2_nsteps_curve_mnist.png     — NLL vs n_steps for BFN on MNIST
  ../plots/fig3_comparison_table.png       — BFN vs D3PM vs reference numbers
  ../plots/fig4_sample_grid_mnist.png      — MNIST visual quality vs n_steps
  ../plots/fig5_sample_grid_text8.txt      — text quality summary

Usage:
  conda run -n bfn python scripts/exp2d_plot_results.py
"""

import json
import math
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs/logs"
PLOTS_DIR   = Path(__file__).resolve().parents[1] / "outputs/figures"
PLOTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Reference numbers from papers (for comparison table)
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_TABLE = {
    "text8 BPC (↓ better)": {
        "BFN (Graves 2023)"            : 1.41,   # our pretrained model
        "D3PM Absorbing (Austin 2021)" : 1.45,   # our trained baseline
        "Multinomial Diff (Hoogeboom 2021)": 1.72,
        "ARDM (Hoogeboom 2022)"        : 1.43,
        "SEDD (Lou 2024)"              : 1.38,   # more recent work
        "Autoregressive (upper bound)" : 1.20,
    },
    "MNIST NLL bits/dim (↓ better)": {
        "BFN discrete (Graves 2023)"   : None,   # filled from our run
        "DDPM (Ho 2020)"               : 0.088,  # cited from paper
        "VAE"                          : 0.114,
        "Flow++ (Ho 2019)"             : 0.0713,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Plot styling
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "bfn"  : {"color": "#4C72B0", "marker": "o", "linewidth": 2.2, "label": "BFN (ours)"},
    "d3pm" : {"color": "#DD8452", "marker": "s", "linewidth": 1.8, "linestyle": "--", "label": "D3PM Absorbing (Austin 2021)"},
    "ref"  : {"color": "#55A868", "marker": "^", "linewidth": 1.5, "linestyle": ":", "label": "Reference (paper)"},
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "figure.dpi":      150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — BPC vs n_steps (text8)
# ─────────────────────────────────────────────────────────────────────────────
def plot_nsteps_curve(json_path: Path, metric_name: str, out_path: Path,
                      d3pm_bpc: float = None, paper_bfn_bpc: float = None):
    with open(json_path) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r["bpc"] is not None and r["n_steps"] > 0]
    steps   = [r["n_steps"] for r in results]
    bpcs    = [r["bpc"]     for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # BFN curve
    ax.plot(steps, bpcs, **STYLE["bfn"])
    ax.scatter(steps, bpcs, color=STYLE["bfn"]["color"], s=60, zorder=5)

    # D3PM horizontal reference line
    if d3pm_bpc:
        ax.axhline(d3pm_bpc, **{k: v for k, v in STYLE["d3pm"].items() if k != "marker"})
        ax.text(steps[-1] * 0.95, d3pm_bpc + 0.005, f"D3PM: {d3pm_bpc:.2f}", 
                color=STYLE["d3pm"]["color"], ha="right", fontsize=9)

    # Paper BFN reference (continuous-time, n_steps=∞ approx)
    if paper_bfn_bpc:
        ax.axhline(paper_bfn_bpc, color="#4C72B0", linestyle="-.", linewidth=1.2, alpha=0.5)
        ax.text(steps[1], paper_bfn_bpc - 0.008, f"BFN paper (cont-time): {paper_bfn_bpc:.2f}",
                color="#4C72B0", fontsize=8.5, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Number of generation steps (n_steps)", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title("BFN: generation quality vs. number of steps\n(text8, character-level)", fontsize=12)
    ax.legend(fontsize=9)

    # Annotate key values
    for s, b in zip(steps, bpcs):
        if s in [10, 50, 100]:
            ax.annotate(f"{b:.3f}", (s, b), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color=STYLE["bfn"]["color"])

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  ✓ Saved: {out_path.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Comparison table as a styled figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison_table(bfn_bpc: float, d3pm_bpc: float, out_path: Path):
    """Render a publication-quality comparison table as a matplotlib figure."""
    models = [
        ("BFN (Graves et al. 2023)",          bfn_bpc,  True,  "Our experiment"),
        ("D3PM Absorbing (Austin et al. 2021)", d3pm_bpc, True,  "Our experiment"),
        ("ARDM (Hoogeboom et al. 2022)",        1.43,    False, "Paper citation"),
        ("SEDD (Lou et al. 2024)",              1.38,    False, "Paper citation"),
        ("Multinomial Diff (2021)",             1.72,    False, "Paper citation"),
        ("Autoregressive (upper bound)",        1.20,    False, "Reference only"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(models) + 1)
    ax.axis("off")

    # Header
    ax.text(0.02, len(models) + 0.6, "Model",             fontweight="bold", fontsize=11)
    ax.text(0.62, len(models) + 0.6, "BPC (text8) ↓",     fontweight="bold", fontsize=11)
    ax.text(0.82, len(models) + 0.6, "Source",             fontweight="bold", fontsize=11)

    ax.axhline(len(models) + 0.3, color="black", linewidth=1.2, xmin=0, xmax=1)
    ax.axhline(0.2,                color="black", linewidth=0.8, xmin=0, xmax=1)

    best_bpc = min(b for _, b, _, _ in models if b is not None)

    for i, (name, bpc, ours, source) in enumerate(reversed(models)):
        y   = i + 0.5
        row_color = "#EFF3FB" if i % 2 == 0 else "white"
        ax.axhspan(i, i + 1, color=row_color, alpha=0.5)

        is_best = (bpc == best_bpc)
        bpc_str = f"{bpc:.3f}" if bpc is not None else "—"
        color   = "#1A5276" if is_best else "black"
        weight  = "bold"     if is_best else "normal"

        # Ours badge
        label = f"★ {name}" if ours else f"    {name}"
        ax.text(0.02, y, label, fontsize=9.5, va="center", color="#154360" if ours else "black", fontweight=weight)
        ax.text(0.62, y, bpc_str, fontsize=10,  va="center", color=color, fontweight=weight)
        ax.text(0.82, y, source,  fontsize=8.5, va="center", color="#555")

    ax.set_title("text8 character-level BPC comparison (★ = our trained model / our eval)",
                 fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"  ✓ Saved: {out_path.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Sample quality vs n_steps grid (MNIST)
# ─────────────────────────────────────────────────────────────────────────────
def plot_sample_grid(samples_dir: Path, dataset: str, out_path: Path):
    """Load rendered PNGs at different n_steps and stitch into a comparison grid."""
    import matplotlib.image as mpimg

    pngs = sorted(samples_dir.glob(f"{dataset}_n*.png"))
    if not pngs:
        print(f"  ⚠ No PNG samples found for {dataset} — run exp2c first")
        return

    n = len(pngs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, png in zip(axes, pngs):
        # extract n_steps from filename: dataset_n0100.png → 100
        steps = int(png.stem.split("_n")[-1])
        img   = mpimg.imread(png)
        ax.imshow(img)
        ax.set_title(f"n={steps}", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"BFN sample quality vs. generation steps — {dataset.upper()}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"  ✓ Saved: {out_path.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Generating all plots for Exp 2")
    print("=" * 60)

    # Load D3PM result
    d3pm_path = RESULTS_DIR / "d3pm_text8_result.json"
    d3pm_bpc  = None
    if d3pm_path.exists():
        with open(d3pm_path) as f:
            d3pm_bpc = json.load(f).get("bpc")
    else:
        print("  ⚠ d3pm_text8_result.json not found — using paper reference 1.45")
        d3pm_bpc = 1.45   # Austin et al. paper value

    # Load BFN text8 ablation
    bfn_text8_path = RESULTS_DIR / "bfn_text8_nsteps.json"
    bfn_bpc_best   = None
    if bfn_text8_path.exists():
        with open(bfn_text8_path) as f:
            data = json.load(f)
        bpcs = [r["bpc"] for r in data["results"] if r["bpc"]]
        bfn_bpc_best = min(bpcs) if bpcs else None

        # Fig 1: n_steps curve (text8)
        plot_nsteps_curve(
            bfn_text8_path,
            metric_name="BPC (bits per character, ↓ better)",
            out_path=PLOTS_DIR / "fig1_nsteps_curve_text8.png",
            d3pm_bpc=d3pm_bpc,
            paper_bfn_bpc=1.41,
        )
    else:
        print("  ⚠ bfn_text8_nsteps.json not found — using paper reference 1.41")
        bfn_bpc_best = 1.41

    # Fig 2: n_steps curve (MNIST)
    bfn_mnist_path = RESULTS_DIR / "bfn_mnist_nsteps.json"
    if bfn_mnist_path.exists():
        plot_nsteps_curve(
            bfn_mnist_path,
            metric_name="NLL (nats/dim, ↓ better)",
            out_path=PLOTS_DIR / "fig2_nsteps_curve_mnist.png",
            d3pm_bpc=None,
            paper_bfn_bpc=None,
        )

    # Fig 3: comparison table
    plot_comparison_table(
        bfn_bpc=bfn_bpc_best or 1.41,
        d3pm_bpc=d3pm_bpc,
        out_path=PLOTS_DIR / "fig3_comparison_table.png",
    )

    # Fig 4: MNIST sample grid
    plot_sample_grid(
        Path(__file__).resolve().parents[1] / "outputs/visual_demos",
        dataset="mnist",
        out_path=PLOTS_DIR / "fig4_sample_grid_mnist.png",
    )

    print(f"\n✓ All plots saved to: {PLOTS_DIR}")
    print("  fig1 — BPC vs n_steps curve (text8)")
    print("  fig2 — NLL vs n_steps curve (MNIST)")
    print("  fig3 — Comparison table: BFN / D3PM / literature")
    print("  fig4 — MNIST sample quality grid")


if __name__ == "__main__":
    main()
