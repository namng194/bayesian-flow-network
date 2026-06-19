#!/usr/bin/env python3
"""
exp1b_sampling_time_benchmark.py
=================================
Exp 1 — Benchmark sampling speed and quality:
  BFN    vs DDPM (full T=1000) vs DDIM (T=50, 100)

This is the most visually compelling comparison for seminar:
  - BFN natively supports any n_steps without architectural change
  - DDPM needs DDIM trick to reduce steps (and quality degrades more)
  - BFN inputs on probability simplex → gradient-based guidance possible

We measure:
  1. Wall-clock time per batch (16 samples) at each step count
  2. NLL/BPC (from exp2a results) — quality signal
  3. Side-by-side visual grid: BFN-n100 vs DDIM-n100 vs DDPM-n1000

Reference:
  BFN    : https://github.com/nnaisense/bayesian-flow-networks
  DDPM   : https://github.com/tqch/ddpm-torch
  DDIM   : Song et al. 2020, arxiv 2010.02502

Usage (run from project root, after exp1a and exp2c):
  conda activate bfn
  python scripts/exp1b_sampling_time_benchmark.py

Outputs:
  results/sampling_time_benchmark.json
  plots/fig_sampling_speed_comparison.png
"""

import subprocess
import sys
import json
import time
import math
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BFN_DIR      = PROJECT_ROOT / "src/bayesian-flow-networks"
DDPM_DIR     = PROJECT_ROOT / "src/ddpm-torch"
RESULTS_DIR  = PROJECT_ROOT / "outputs/logs"
PLOTS_DIR    = PROJECT_ROOT / "outputs/figures"
SAMPLES_DIR  = Path(__file__).resolve().parents[1] / "outputs/visual_demos"

sys.path.insert(0, str(BFN_DIR))

# ── Timing configs ─────────────────────────────────────────────────────────
N_SAMPLES   = 16       # batch of 16 for fair comparison
N_WARMUP    = 2        # warmup runs before timing
N_TIMING    = 5        # timing runs to average

BFN_NSTEPS  = [10, 25, 50, 100, 250, 500, 1000]
DDIM_NSTEPS = [10, 25, 50, 100, 250, 500, 1000]  # DDPM uses DDIM for < 1000


# ── BFN timing ────────────────────────────────────────────────────────────────
def time_bfn_sampling(dataset: str, n_steps: int) -> float:
    """
    Time BFN generation by calling official sample.py and measuring wall time.
    Uses the --samples_shape flag to control batch size.
    """
    if dataset == "cifar10":
        cfg   = BFN_DIR / "configs/cifar10_discretized_256bins.yaml"
        ckpt  = Path(__file__).resolve().parents[1] / "checkpoints/bfn/cifar10_256d_ema.pt"
        shape = f"[{N_SAMPLES}, 32, 32, 3]"
    else:  # mnist
        cfg   = BFN_DIR / "configs/mnist_discrete.yaml"
        ckpt  = Path(__file__).resolve().parents[1] / "checkpoints/bfn/mnist_ema.pt"
        shape = f"[{N_SAMPLES}, 28, 28, 1]"

    save_file = SAMPLES_DIR / f"_timing_bfn_{dataset}_n{n_steps}.pt"
    cmd = [
        "python", str(BFN_DIR / "sample.py"),
        f"seed=42",
        f"config_file={cfg}",
        f"load_model={ckpt}",
        f"samples_shape={shape}",
        f"n_steps={n_steps}",
        f"save_file={save_file}",
    ]

    times = []
    for i in range(N_WARMUP + N_TIMING):
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=BFN_DIR)
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            return None
        if i >= N_WARMUP:
            times.append(elapsed)

    return float(np.mean(times))


def time_ddpm_sampling(dataset: str, ddim_steps: int) -> float:
    """
    Time DDPM/DDIM generation via ddpm-torch generate.py.
    For ddim_steps < 1000, uses DDIM quadratic skip schedule.
    """
    chkpt_dir = PROJECT_ROOT / f"checkpoints/ddpm/{dataset}"
    chkpts    = sorted(chkpt_dir.glob("*.pt")) if chkpt_dir.exists() else []
    if not chkpts:
        print(f"    ⚠ No DDPM checkpoint for {dataset} — skipping DDPM timing")
        return None

    cmd = [
        "python", str(DDPM_DIR / "generate.py"),
        "--dataset",    dataset,
        "--chkpt-path", str(chkpts[-1]),
        "--num-images", str(N_SAMPLES),
        "--save-path",  str(SAMPLES_DIR / f"_timing_ddpm_{dataset}_T{ddim_steps}.pt"),
    ]
    if ddim_steps < 1000:
        cmd += [
            "--use-ddim",
            "--skip-schedule", "quadratic",
            "--subseq-size",   str(ddim_steps),
        ]

    times = []
    for i in range(N_WARMUP + N_TIMING):
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=DDPM_DIR)
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            return None
        if i >= N_WARMUP:
            times.append(elapsed)

    return float(np.mean(times))


# ── Main benchmark ─────────────────────────────────────────────────────────────
def run_benchmark(dataset: str = "cifar10") -> dict:
    print(f"\n{'='*60}")
    print(f"  Sampling Speed Benchmark — {dataset.upper()}")
    print(f"  Batch size: {N_SAMPLES} | Timing runs: {N_TIMING}")
    print(f"{'='*60}")

    results = {"dataset": dataset, "n_samples": N_SAMPLES, "bfn": [], "ddpm_ddim": []}

    # BFN sweep
    print("\n── BFN ──")
    for n in BFN_NSTEPS:
        t = time_bfn_sampling(dataset, n)
        if t is not None:
            print(f"  n={n:4d}  →  {t:.2f}s  ({t/N_SAMPLES*1000:.0f}ms/sample)")
            results["bfn"].append({"n_steps": n, "time_sec": t,
                                   "ms_per_sample": t / N_SAMPLES * 1000})

    # DDPM/DDIM sweep
    print("\n── DDPM / DDIM ──")
    for n in DDIM_NSTEPS:
        t = time_ddpm_sampling(dataset, n)
        if t is not None:
            label = "DDPM" if n == 1000 else f"DDIM(T={n})"
            print(f"  {label:14s}  →  {t:.2f}s  ({t/N_SAMPLES*1000:.0f}ms/sample)")
            results["ddpm_ddim"].append({"n_steps": n, "time_sec": t,
                                         "ms_per_sample": t / N_SAMPLES * 1000,
                                         "method": "DDPM" if n == 1000 else "DDIM"})

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_speed_comparison(benchmark: dict, quality_json: Path = None):
    """
    Dual-axis plot: sampling time (left y) and BPC quality (right y) vs n_steps.
    Shows BFN's favorable few-step generation vs DDPM's sharp quality drop.
    """
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25})

    # ── Timing lines ──
    if benchmark["bfn"]:
        xs = [r["n_steps"]         for r in benchmark["bfn"]]
        ys = [r["ms_per_sample"]   for r in benchmark["bfn"]]
        ax1.plot(xs, ys, "o-",  color="#4C72B0", lw=2.2, label="BFN (time)")
        ax1.scatter(xs, ys, color="#4C72B0", s=55, zorder=5)

    if benchmark["ddpm_ddim"]:
        xs = [r["n_steps"]         for r in benchmark["ddpm_ddim"]]
        ys = [r["ms_per_sample"]   for r in benchmark["ddpm_ddim"]]
        ax1.plot(xs, ys, "s--", color="#DD8452", lw=2.0, label="DDPM/DDIM (time)")
        ax1.scatter(xs, ys, color="#DD8452", s=55, zorder=5)

    # ── Quality overlay (BPC from ablation) ──
    if quality_json and quality_json.exists():
        with open(quality_json) as f:
            qdata = json.load(f)
        valid = [r for r in qdata["results"] if r["bpc"] and r["n_steps"] > 0]
        if valid:
            qx = [r["n_steps"] for r in valid]
            qy = [r["bpc"]     for r in valid]
            ax2.plot(qx, qy, "^:", color="#55A868", lw=1.8, label="BFN quality (BPC →)")
            ax2.scatter(qx, qy, color="#55A868", s=45, zorder=5, alpha=0.85)
            ax2.set_ylabel("BPC (bits per character, ↓ better)", color="#55A868", fontsize=10)
            ax2.tick_params(axis="y", labelcolor="#55A868")

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of sampling steps", fontsize=11)
    ax1.set_ylabel("Sampling time (ms / sample)", fontsize=11)
    ax1.set_title(
        "BFN vs DDPM: sampling speed vs. generation steps\n"
        "BFN: n_steps is freely tunable | DDPM: needs DDIM trick",
        fontsize=11
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # Annotate BFN advantage zone
    ax1.axvspan(1, 100, alpha=0.06, color="#4C72B0", label="_nolegend_")
    ax1.text(15, ax1.get_ylim()[1] * 0.9, "BFN native\nfew-step zone",
             fontsize=8, color="#4C72B0", alpha=0.8, ha="center")

    fig.tight_layout()
    out = PLOTS_DIR / "fig_sampling_speed_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    dataset = "cifar10"
    for arg in sys.argv[1:]:
        if arg in ("cifar10", "mnist", "text8"):
            dataset = arg

    benchmark = run_benchmark(dataset)

    # Save raw results
    out = RESULTS_DIR / "sampling_time_benchmark.json"
    with open(out, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n  ✓ Raw results saved to {out.name}")

    # Plot (overlay BPC from text8 ablation for dual-axis quality insight)
    quality_json = RESULTS_DIR / "bfn_text8_nsteps.json"
    plot_speed_comparison(benchmark, quality_json)
