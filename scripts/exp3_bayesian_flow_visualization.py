#!/usr/bin/env python3
"""
exp3_bayesian_flow_visualization.py
=====================================
Exp 3 — Visualize the Bayesian Flow process itself.

This is the most visually compelling part for the seminar — show the AUDIENCE
what "Bayesian flow" actually *looks like* as it runs, not just the final result.

What we visualize:
  A. MNIST: probability flow from uniform prior → sharp image
     - At each time step t, capture the *network's mean output* (theta_hat)
     - Show it as a grid evolving over t ∈ {0, 0.1, 0.2, ..., 1.0}
     
  B. text8: probability simplex flow for a short sequence
     - At each step, show a heatmap of p(char | position) over 27 characters
     - Watch entropy collapse from uniform → peaked distribution
     
  C. BPC vs entropy plot: show how information accumulates during generation
     - Plot H(theta_t) (average entropy of output distribution) vs t
     - Compare with DDPM's noise schedule σ(t): different functional form!

Key insight to demonstrate:
  BFN accumulates information gradually via Bayesian updating of independent
  priors — this is fundamentally different from DDPM's Markov chain reversal.
  The probability simplex inputs make discrete generation naturally differentiable.

Reference:
  Official repo + bfn.gif: https://github.com/nnaisense/bayesian-flow-networks
  Paper Section 2–3: arxiv 2308.07037

Usage:
  conda activate bfn
  cd bayesian-flow-networks
  python ../scripts/exp3_bayesian_flow_visualization.py

Outputs:
  results/samples/flow_mnist_evolution.png  — MNIST flow grid (main visual)
  results/samples/flow_text8_heatmap.png    — text8 simplex heatmap
  plots/fig3_entropy_vs_t.png               — entropy collapse curve
  results/flow_intermediate_states.json     — entropy values for each t
"""

import sys
import json
import math
import numpy as np
from pathlib import Path
from copy import deepcopy
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BFN_DIR      = PROJECT_ROOT / "src/bayesian-flow-networks"
RESULTS_DIR  = PROJECT_ROOT / "outputs/logs"
PLOTS_DIR    = PROJECT_ROOT / "outputs/figures"
SAMPLES_DIR  = Path(__file__).resolve().parents[1] / "outputs/visual_demos"

for d in [RESULTS_DIR, PLOTS_DIR, SAMPLES_DIR]:
    d.mkdir(exist_ok=True)

sys.path.insert(0, str(BFN_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load BFN model (text8) for capturing intermediate states ─────────────────
TEXT8_CKPT = Path(__file__).resolve().parents[1] / "checkpoints/bfn/text8_ema.pt"
TEXT8_CFG  = BFN_DIR / "configs/text8_discrete.yaml"
MNIST_CKPT = Path(__file__).resolve().parents[1] / "checkpoints/bfn/mnist_ema.pt"
MNIST_CFG  = BFN_DIR / "configs/mnist_discrete.yaml"

TEXT8_CHARS = list(" abcdefghijklmnopqrstuvwxyz")  # 27 chars + space


# ── Part A: MNIST flow visualization ─────────────────────────────────────────
def visualize_mnist_flow(n_steps: int = 100, n_samples: int = 8):
    """
    Generate MNIST samples while saving intermediate states at N checkpoints.
    We do this by hooking into BFN's sample procedure and saving theta_hat
    (the network's predicted clean image) at each time step.

    This is the core visual: shows how a binary image "crystallizes" from noise.
    """
    print("\n── Part A: MNIST flow visualization ──")

    try:
        import omegaconf
        from model import DiscreteBFN           # from BFN repo
        from networks.unet import UNet          # from BFN repo
        from probability import DiscreteDist    # from BFN repo
        from data import batch_to_images
    except ImportError as e:
        print(f"  ✗ Import error (run from bayesian-flow-networks/): {e}")
        return None

    # Load config + model
    cfg   = omegaconf.OmegaConf.load(TEXT8_CFG)  # reuse structure, swap data
    cfg_m = omegaconf.OmegaConf.load(MNIST_CFG)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Instantiate model as done in sample.py
    state  = torch.load(MNIST_CKPT, map_location=device)
    model  = DiscreteBFN.from_config(cfg_m)
    model.load_state_dict(state)
    model = model.to(device).eval()

    # Capture states at these fractions of t
    capture_fracs = np.linspace(0.0, 1.0, 12)  # 12 snapshots
    captured      = []                           # list of (t_frac, theta_hat)

    # We hook into the model's generation loop
    # BFN discrete sampling: runs from t=0 to t=1, updating theta via Bayes
    shape = (n_samples, 28, 28, 1)

    with torch.no_grad():
        # Replicate BFN's generate() but save intermediate theta_hat
        # theta starts as uniform: p(x=0) = p(x=1) = 0.5 for binary MNIST
        theta = torch.full(
            (n_samples, 28, 28, 1, 2),  # 2 classes: {0, 1}
            fill_value=0.5,
            device=device
        )

        for i in range(n_steps):
            t     = torch.tensor(i / n_steps, device=device)
            t_frac = float(t)

            # Network predicts p_O(x | theta, t)  — the output distribution
            theta_hat = model.forward_theta(theta, t.expand(n_samples))

            # Capture at snapshot fractions
            for frac in capture_fracs:
                if abs(t_frac - frac) < (0.5 / n_steps):
                    # theta_hat shape: (B, H, W, C, K) — take mean image
                    mean_img = theta_hat[..., 1].cpu()  # p(x=1) for binary
                    captured.append((t_frac, mean_img.clone()))
                    break

            # Bayesian update: receive noisy sample, update theta
            theta = model.bayesian_update(theta, theta_hat, t)

    # Render flow grid: rows = samples, cols = time steps
    if captured:
        n_cols = len(captured)
        n_rows = min(n_samples, 4)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.4))

        for col, (t_frac, imgs) in enumerate(captured):
            for row in range(n_rows):
                ax  = axes[row, col]
                img = imgs[row, :, :, 0].numpy()
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"t={t_frac:.1f}", fontsize=8)

        fig.suptitle(
            "BFN: Binary MNIST — Bayesian flow over time t ∈ [0, 1]\n"
            "Each column shows p(x=1 | θ_t) — image crystallizing from uniform prior",
            fontsize=10, y=1.02
        )
        fig.tight_layout()
        out = SAMPLES_DIR / "flow_mnist_evolution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved MNIST flow grid: {out.name}")
        plt.close(fig)
        return captured

    return None


# ── Part B: text8 simplex heatmap ────────────────────────────────────────────
def visualize_text8_simplex(n_steps: int = 100, seq_len: int = 32):
    """
    For a text8 generation run, capture the full categorical distribution
    theta ∈ Δ^{27-1} (probability simplex) at each step.

    Renders a heatmap: x-axis = position in sequence, y-axis = character,
    color = probability. Shows entropy collapsing from uniform to peaked.
    """
    print("\n── Part B: text8 simplex heatmap ──")

    try:
        import omegaconf
        from model import DiscreteBFN
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return None

    cfg    = omegaconf.OmegaConf.load(TEXT8_CFG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state  = torch.load(TEXT8_CKPT, map_location=device)
    model  = DiscreteBFN.from_config(cfg)
    model.load_state_dict(state)
    model  = model.to(device).eval()

    K       = 27   # alphabet size (space + a-z)
    shape   = (1, seq_len)    # 1 sequence, 32 chars
    capture_steps = [0, 5, 10, 20, 35, 50, 70, 90, 99]   # step indices

    captured_heatmaps = []
    entropy_trace     = []

    with torch.no_grad():
        # Uniform prior over 27 characters
        theta = torch.full((1, seq_len, K), fill_value=1.0 / K, device=device)

        for i in range(n_steps):
            t = torch.tensor(i / n_steps, device=device)

            # Get network output distribution
            theta_hat = model.forward_theta(theta, t.expand(1))
            # theta_hat: (1, seq_len, K) — categorical probs

            # Entropy at this step
            eps  = 1e-8
            H    = -(theta_hat * (theta_hat + eps).log()).sum(-1).mean().item()
            H_bits = H / math.log(2)
            entropy_trace.append({"step": i, "t": float(t), "entropy_bits": H_bits})

            if i in capture_steps:
                hmap = theta_hat[0, :, :].cpu().numpy()  # (seq_len, K)
                captured_heatmaps.append((i, float(t), hmap))

            # Bayesian update
            theta = model.bayesian_update(theta, theta_hat, t)

    # Render heatmap grid
    n_panels = len(captured_heatmaps)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 2.5, 5),
                             sharey=True)

    for ax, (step_i, t_frac, hmap) in zip(axes, captured_heatmaps):
        # hmap: (seq_len, K) — transpose for display: K on y-axis
        im = ax.imshow(
            hmap.T, aspect="auto", cmap="Blues",
            vmin=0, vmax=1, origin="lower",
            interpolation="nearest"
        )
        ax.set_title(f"step={step_i}\nt={t_frac:.2f}", fontsize=8)
        ax.set_xlabel("Position", fontsize=7)
        if ax is axes[0]:
            ax.set_yticks(range(K))
            ax.set_yticklabels(TEXT8_CHARS, fontsize=6)
            ax.set_ylabel("Character", fontsize=8)

    fig.suptitle(
        "BFN text8: probability simplex θ_t over generation steps\n"
        "Blue = high p(char|pos). Uniform → peaked = information gain via Bayes",
        fontsize=10, y=1.03
    )
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.7)
    cbar.set_label("p(char | position, t)", fontsize=8)

    fig.tight_layout()
    out = SAMPLES_DIR / "flow_text8_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved text8 simplex heatmap: {out.name}")
    plt.close(fig)

    return entropy_trace


# ── Part C: Entropy collapse vs DDPM noise schedule ──────────────────────────
def plot_entropy_vs_noise(entropy_trace: list):
    """
    Compare BFN's entropy collapse H(θ_t) with DDPM's noise schedule σ(t).
    This makes the mathematical difference concrete and visual.

    BFN  : H(θ_t) decreases monotonically as Bayesian updates accumulate
    DDPM : σ²(t) decreases according to a fixed schedule (linear/cosine)
           BUT the model has no explicit probability simplex representation
    """
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: BFN entropy ──
    ax = axes[0]
    if entropy_trace:
        ts   = [e["t"]            for e in entropy_trace]
        Hbits = [e["entropy_bits"] for e in entropy_trace]
        ax.plot(ts, Hbits, color="#4C72B0", lw=2.2, label="BFN H(θ_t) [bits]")
        # Mark uniform entropy: log2(27) ≈ 4.75 bits
        ax.axhline(math.log2(27), color="#4C72B0", linestyle=":", alpha=0.5,
                   label=f"Uniform prior: log₂(27) ≈ {math.log2(27):.2f} bits")
        ax.fill_between(ts, 0, Hbits, alpha=0.12, color="#4C72B0")

    ax.set_xlabel("Generation time t ∈ [0, 1]", fontsize=11)
    ax.set_ylabel("Output entropy H(θ_t)  [bits]", fontsize=11)
    ax.set_title("BFN: entropy collapse during generation\n(text8, 27-char alphabet)", fontsize=10)
    ax.legend(fontsize=9)

    # ── Right: DDPM noise schedule comparison ──
    ax2 = axes[1]
    t_arr = np.linspace(0, 1, 1000)

    # DDPM linear beta schedule (Ho 2020): β_t = β_min + t*(β_max - β_min)
    beta_min, beta_max = 1e-4, 0.02
    betas  = beta_min + t_arr * (beta_max - beta_min)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    sigma_ddpm = np.sqrt(1 - alpha_bars)   # noise std at each t

    # BFN accuracy schedule σ₁²(t) = 1 - (1 - σ_min²)^t
    # For discrete data, the input precision β(t) grows from 0 to ∞
    # We show its normalized form for comparison
    sigma_min = 0.001
    bfn_precision = 1 - (1 - sigma_min ** 2) ** t_arr   # increasing precision

    ax2.plot(t_arr, sigma_ddpm,    color="#DD8452", lw=2.0,
             label="DDPM σ(t) — noise level")
    ax2.plot(t_arr, 1 - bfn_precision, color="#4C72B0", lw=2.0, linestyle="--",
             label="BFN 1 - β(t) — uncertainty (approx)")

    ax2.set_xlabel("t ∈ [0, 1]", fontsize=11)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title(
        "DDPM noise schedule vs BFN accuracy schedule\n"
        "Different functional form → different inductive bias",
        fontsize=10
    )
    ax2.legend(fontsize=9)

    # Annotation box
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.4)
    ax2.text(0.05, 0.95,
             "Key difference:\n"
             "DDPM: fixed Markov chain\n"
             "BFN: Bayesian update of θ\n"
             "  → discrete data is natural",
             transform=ax2.transAxes, fontsize=8.5,
             verticalalignment="top", bbox=props)

    fig.tight_layout()
    out = PLOTS_DIR / "fig3_entropy_vs_t.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved entropy plot: {out.name}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Exp 3 — Bayesian Flow Visualization")
    print("=" * 60)

    results = {}

    # Part A: MNIST flow evolution
    # (requires model internals — fallback if hooks not available)
    try:
        captured = visualize_mnist_flow(n_steps=100, n_samples=8)
        results["mnist_flow"] = "success" if captured else "skipped"
    except Exception as e:
        print(f"  ⚠ MNIST flow skipped: {e}")
        results["mnist_flow"] = f"error: {str(e)}"

    # Part B + C: text8 simplex + entropy
    try:
        entropy_trace = visualize_text8_simplex(n_steps=100, seq_len=32)
        if entropy_trace:
            plot_entropy_vs_noise(entropy_trace)
            results["entropy_trace_steps"] = len(entropy_trace)

            # Save trace for potential reuse
            out = RESULTS_DIR / "flow_intermediate_states.json"
            with open(out, "w") as f:
                json.dump({"text8_entropy": entropy_trace}, f, indent=2)
            print(f"  ✓ Entropy trace saved: {out.name}")
    except Exception as e:
        print(f"  ⚠ text8 simplex skipped: {e}")
        # Fallback: generate entropy plot using theoretical curves only
        print("  → Generating theoretical comparison plot (no model needed)")
        plot_entropy_vs_noise(None)
        results["text8_flow"] = f"theoretical only: {str(e)}"

    print("\n✓ Exp 3 complete. Outputs:")
    print(f"  {SAMPLES_DIR / 'flow_mnist_evolution.png'}")
    print(f"  {SAMPLES_DIR / 'flow_text8_heatmap.png'}")
    print(f"  {PLOTS_DIR   / 'fig3_entropy_vs_t.png'}")

    return results


if __name__ == "__main__":
    main()
