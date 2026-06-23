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
import os
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BFN_DIR      = PROJECT_ROOT / "src/bayesian-flow-networks"
DDPM_DIR     = PROJECT_ROOT / "src/ddpm-torch"
RESULTS_DIR  = PROJECT_ROOT / "outputs/logs"
PLOTS_DIR    = PROJECT_ROOT / "outputs/figures"
SAMPLES_DIR  = PROJECT_ROOT / "outputs/visual_demos"

sys.path.insert(0, str(BFN_DIR))

# ── Device setup ───────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"  ✓ GPU detected: {GPU_NAME}")
else:
    GPU_NAME = None
    print("  ⚠ No GPU found — running on CPU")

GPU_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

# ── Timing configs ─────────────────────────────────────────────────────────────
N_SAMPLES  = 16
N_WARMUP   = 2
N_TIMING   = 5

BFN_NSTEPS  = [10, 25, 50, 100, 250, 500, 1000]
DDIM_NSTEPS = [10, 25, 50, 100, 250, 500, 1000]


# ── Accurate GPU timing ────────────────────────────────────────────────────────
def _timed(fn):
    """
    Time fn() with CUDA sync on both sides.
    torch.cuda.synchronize() is mandatory: without it perf_counter() measures
    kernel *launch* time, not *completion* time (CUDA ops are async by default).
    Returns (elapsed_sec, return_value_of_fn).
    """
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, result


# ── BFN: in-process timing ────────────────────────────────────────────────────
def _load_bfn(dataset: str):
    """
    Load BFN model once, move to GPU. Reused across all n_steps.
    GPU memory won't change during sample() calls — model is already resident.
    This is correct: we want to measure *inference time*, not load time.
    """
    from utils_train import make_config, make_bfn  # noqa: PLC0415

    if dataset == "cifar10":
        cfg_path  = BFN_DIR / "configs/cifar10_discretized_256bins.yaml"
        ckpt_path = PROJECT_ROOT / "checkpoints/bfn/cifar10_256d_ema.pt"
        shape     = [N_SAMPLES, 32, 32, 3]
    else:
        cfg_path  = BFN_DIR / "configs/mnist_discrete.yaml"
        ckpt_path = PROJECT_ROOT / "checkpoints/bfn/mnist_ema.pt"
        shape     = [N_SAMPLES, 28, 28, 1]

    train_cfg = make_config(str(cfg_path))
    bfn = make_bfn(train_cfg.model)
    bfn.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
    bfn.to(DEVICE)
    bfn.eval()
    return bfn, shape


def time_bfn_sampling(dataset: str, n_steps: int, bfn, shape: list) -> float | None:
    """
    Time BFN in-process: call bfn.sample() directly, no subprocess overhead.
    Model is already on GPU — memory stays constant during sampling (correct).
    """
    times = []

    def _one_run():
        with torch.no_grad():
            return bfn.sample(shape, n_steps)

    for i in range(N_WARMUP + N_TIMING):
        elapsed, _ = _timed(_one_run)
        if i >= N_WARMUP:
            times.append(elapsed)

    return float(np.mean(times))


# ── DDPM / DDIM: subprocess timing ────────────────────────────────────────────
def _find_ddpm_checkpoint(dataset: str) -> Path | None:
    """
    ddpm-torch saves checkpoints to checkpoints/ddpm/<dataset>/<dataset>/*.pt
    (the dataset dir is nested). Find the latest checkpoint.
    """
    # Try nested path first (actual layout), then flat fallback
    for candidate in [
        PROJECT_ROOT / f"checkpoints/ddpm/{dataset}/{dataset}",
        PROJECT_ROOT / f"checkpoints/ddpm/{dataset}",
    ]:
        if candidate.exists():
            chkpts = sorted(candidate.glob("*.pt"))
            if chkpts:
                return chkpts[-1]
    return None


def _ddpm_args_for_generate(dataset: str) -> dict:
    """
    generate.py uses --total-size (not --num-images) and --save-dir (not --save-path).
    Actual flags visible in the error:
      --total-size TOTAL_SIZE
      --save-dir SAVE_DIR
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    save_dir = SAMPLES_DIR / f"ddpm_{dataset}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return {
        "--total-size": str(N_SAMPLES),
        "--save-dir":   str(save_dir),
    }


def time_ddpm_sampling(dataset: str, ddim_steps: int) -> float | None:
    ckpt = _find_ddpm_checkpoint(dataset)
    if ckpt is None:
        print(f"    ⚠ No DDPM checkpoint found under "
              f"checkpoints/ddpm/{dataset}/ — skipping")
        return None

    extra = _ddpm_args_for_generate(dataset)

    cmd = [
        sys.executable, str(DDPM_DIR / "generate.py"),
        "--chkpt-path", str(ckpt),
        "--device",     DEVICE,
    ]
    for k, v in extra.items():
        cmd += [k, v]

    if ddim_steps < 1000:
        cmd += [
            "--use-ddim",
            "--skip-schedule", "quadratic",
            "--subseq-size",   str(ddim_steps),
        ]

    times = []
    for i in range(N_WARMUP + N_TIMING):
        def _run():
            return subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=DDPM_DIR, env=GPU_ENV)
        elapsed, result = _timed(_run)

        if result.returncode != 0:
            label = "DDPM" if ddim_steps == 1000 else f"DDIM(T={ddim_steps})"
            print(f"    ✗ {label} failed:\n{result.stderr[-800:]}")
            return None

        if i >= N_WARMUP:
            times.append(elapsed)

    return float(np.mean(times))


# ── Main benchmark ─────────────────────────────────────────────────────────────
def run_benchmark(dataset: str = "cifar10") -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Sampling Speed Benchmark — {dataset.upper()}")
    print(f"  Device   : {DEVICE}" + (f" ({GPU_NAME})" if DEVICE == "cuda" else ""))
    print(f"  Batch    : {N_SAMPLES} samples | Timing runs: {N_TIMING}")
    print(f"{'='*60}")

    results = {
        "dataset":   dataset,
        "device":    DEVICE,
        "gpu_name":  GPU_NAME,
        "n_samples": N_SAMPLES,
        "bfn":       [],
        "ddpm_ddim": [],
    }

    # ── BFN ──
    print("\n── BFN ── (loading model...)")
    bfn, shape = None, None
    try:
        bfn, shape = _load_bfn(dataset)
        mem_mb = torch.cuda.memory_allocated() / 1024**2 if DEVICE == "cuda" else 0
        print(f"  ✓ BFN loaded on {DEVICE}  (GPU mem allocated: {mem_mb:.0f} MB)")
    except Exception as e:
        print(f"  ✗ Failed to load BFN: {e}")

    if bfn is not None:
        for n in BFN_NSTEPS:
            t = time_bfn_sampling(dataset, n, bfn, shape)
            if t is not None:
                ms = t / N_SAMPLES * 1000
                print(f"  n={n:4d}  →  {t:.3f}s  ({ms:.1f}ms/sample)")
                results["bfn"].append({
                    "n_steps":       n,
                    "time_sec":      t,
                    "ms_per_sample": ms,
                })

    # ── DDPM / DDIM ──
    print("\n── DDPM / DDIM ──")
    ckpt = _find_ddpm_checkpoint(dataset)
    if ckpt:
        print(f"  ✓ Using checkpoint: {ckpt.name}")
    for n in DDIM_NSTEPS:
        t = time_ddpm_sampling(dataset, n)
        if t is not None:
            ms    = t / N_SAMPLES * 1000
            label = "DDPM" if n == 1000 else f"DDIM(T={n})"
            print(f"  {label:14s}  →  {t:.3f}s  ({ms:.1f}ms/sample)")
            results["ddpm_ddim"].append({
                "n_steps":       n,
                "time_sec":      t,
                "ms_per_sample": ms,
                "method":        "DDPM" if n == 1000 else "DDIM",
            })

    return results


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_speed_comparison(benchmark: dict, quality_json: Path = None):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25})

    if benchmark["bfn"]:
        xs = [r["n_steps"]       for r in benchmark["bfn"]]
        ys = [r["ms_per_sample"] for r in benchmark["bfn"]]
        ax1.plot(xs, ys, "o-",  color="#4C72B0", lw=2.2, label="BFN (time)")
        ax1.scatter(xs, ys, color="#4C72B0", s=55, zorder=5)

    if benchmark["ddpm_ddim"]:
        xs = [r["n_steps"]       for r in benchmark["ddpm_ddim"]]
        ys = [r["ms_per_sample"] for r in benchmark["ddpm_ddim"]]
        ax1.plot(xs, ys, "s--", color="#DD8452", lw=2.0, label="DDPM/DDIM (time)")
        ax1.scatter(xs, ys, color="#DD8452", s=55, zorder=5)

    if quality_json and quality_json.exists():
        with open(quality_json) as f:
            qdata = json.load(f)
        valid = [r for r in qdata.get("results", []) if r.get("bpc") and r["n_steps"] > 0]
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

    device_label = benchmark.get("gpu_name") or benchmark.get("device", "CPU")
    ax1.set_title(
        f"BFN vs DDPM: sampling speed vs. generation steps\n"
        f"Device: {device_label} | "
        f"BFN: n_steps freely tunable | DDPM: needs DDIM trick",
        fontsize=10,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ylim = ax1.get_ylim()
    ax1.axvspan(1, 100, alpha=0.06, color="#4C72B0")
    ax1.text(15, ylim[0] + (ylim[1] - ylim[0]) * 0.85,
             "BFN native\nfew-step zone",
             fontsize=8, color="#4C72B0", alpha=0.8, ha="center")

    fig.tight_layout()
    out = PLOTS_DIR / "fig_sampling_speed_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Plot saved: {out}")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = "cifar10"
    for arg in sys.argv[1:]:
        if arg in ("cifar10", "mnist", "text8"):
            dataset = arg

    benchmark = run_benchmark(dataset)

    out = RESULTS_DIR / "sampling_time_benchmark.json"
    with open(out, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"  ✓ Results saved: {out}")

    quality_json = RESULTS_DIR / "bfn_text8_nsteps.json"
    plot_speed_comparison(benchmark, quality_json)
