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
import torch
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BFN_DIR      = PROJECT_ROOT / "src/bayesian-flow-networks"
RESULTS_DIR  = PROJECT_ROOT / "outputs/logs"
PLOTS_DIR    = PROJECT_ROOT / "outputs/figures"
SAMPLES_DIR  = Path(__file__).resolve().parents[1] / "outputs/visual_demos"
 
for d in [RESULTS_DIR, PLOTS_DIR, SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
 
sys.path.insert(0, str(BFN_DIR))
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
 
TEXT8_CKPT = Path(__file__).resolve().parents[1] / "checkpoints/bfn/text8_ema.pt"
TEXT8_CFG  = BFN_DIR / "configs/text8_discrete.yaml"
MNIST_CKPT = Path(__file__).resolve().parents[1] / "checkpoints/bfn/mnist_ema.pt"
MNIST_CFG  = BFN_DIR / "configs/mnist_discrete.yaml"
 
# Real char table used by the repo's data.py (index 0 = "_" for space, not " ")
TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")
 
 
def _load_bfn(config_file: Path, checkpoint_file: Path, device: str):
    """Build a BFN exactly the way sample.py does: read the training config,
    construct the matching net + BayesianFlow + Loss via make_bfn, then load
    the checkpoint weights. Returns the eval-mode model on `device`."""
    from utils_train import make_config, make_bfn
 
    train_cfg = make_config(str(config_file))
    bfn = make_bfn(train_cfg.model)
    state = torch.load(checkpoint_file, map_location=device, weights_only=True)
    bfn.load_state_dict(state)
    return bfn.to(device).eval()
 
 
# ── Part A: MNIST flow visualization ─────────────────────────────────────────
def visualize_mnist_flow(n_steps: int = 100, n_samples: int = 8):
    """
    Generate MNIST samples while saving intermediate states at N checkpoints.
    This replicates BFN.sample()'s loop (model.py) manually so we can snapshot
    the network's predicted output distribution (theta_hat) at each time step,
    instead of just the final sample.
    """
    print("\n── Part A: MNIST flow visualization ──")
 
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")
        bfn = _load_bfn(MNIST_CFG, MNIST_CKPT, device)
    except Exception as e:
        print(f"  ✗ Could not build/load MNIST model (run from bayesian-flow-networks/): {e}")
        return None
 
    bayesian_flow = bfn.bayesian_flow                 # DiscreteBayesianFlow, n_classes=2
    distribution_factory = bfn.loss.distribution_factory
 
    data_shape = (n_samples, 28, 28, 1)
    # 12 evenly spaced snapshot steps in [1, n_steps]
    capture_idx = set(int(x) for x in np.linspace(1, n_steps, 12))
    captured = []  # list of (t_frac, mean_img [n_samples,28,28,1])
 
    with torch.no_grad():
        input_params = bayesian_flow.get_prior_input_params(data_shape, device)
 
        for i in range(1, n_steps + 1):
            t = torch.ones(*data_shape, device=device) * (i - 1) / n_steps
            net_inputs = bayesian_flow.params_to_net_inputs(input_params)
            output_params = bfn.net(net_inputs, t)
            output_dist = distribution_factory.get_dist(output_params, input_params, t)
 
            if i in capture_idx:
                # theta_hat: network's predicted p(x=1) for each pixel.
                # BernoulliFactory does logits.squeeze(-1) internally, so probs
                # comes back as (n_samples, 28, 28, 2) — no trailing size-1 dim.
                mean_img = output_dist.probs[..., 1].detach().cpu()  # (n_samples, 28, 28)
                captured.append(((i - 1) / n_steps, mean_img.clone()))
 
            # Actual Bayesian update, same as BFN.sample()
            output_sample = output_dist.sample().reshape(*data_shape)
            alpha = bayesian_flow.get_alpha(i, n_steps)
            y = bayesian_flow.get_sender_dist(output_sample, alpha).sample()
            input_params = bayesian_flow.update_input_params(input_params, y, alpha)
 
    if not captured:
        return None
 
    n_cols = len(captured)
    n_rows = min(n_samples, 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.4))
 
    for col, (t_frac, imgs) in enumerate(captured):
        for row in range(n_rows):
            ax = axes[row, col]
            img = imgs[row, :, :].numpy()  # imgs is (n_samples, 28, 28)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={t_frac:.1f}", fontsize=8)
 
    fig.suptitle(
        "BFN: Binary MNIST — Bayesian flow over time t ∈ [0, 1]\n"
        "Each column shows the network's predicted p(x=1 | θ_t)",
        fontsize=10, y=1.02
    )
    fig.tight_layout()
    out = SAMPLES_DIR / "flow_mnist_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved MNIST flow grid: {out.name}")
    plt.close(fig)
    return captured
 
 
# ── Part B: text8 simplex heatmap ────────────────────────────────────────────
def visualize_text8_simplex(n_steps: int = 100, seq_len: int = 256, display_len: int = 48):
    """
    For a text8 generation run, capture the full categorical distribution
    (network's predicted probs) at each step, replicating BFN.sample()'s loop
    manually so intermediate states can be snapshotted.
 
    seq_len must be 256: the GPT net's TextInputAdapter registers a fixed
    positional-embedding buffer sized to configs/text8_discrete.yaml's
    data.seq_len (256) and adds it elementwise with no slicing, so any other
    length breaks (mismatched tensor sizes in the addition).
    display_len only controls how many of the 256 positions are plotted in
    the heatmap, purely for readability; entropy is still computed over all
    256 positions.
    """
    print("\n── Part B: text8 simplex heatmap ──")
    if seq_len != 256:
        print("  ⚠ Overriding seq_len to 256 — the checkpoint's positional embedding is fixed-size.")
        seq_len = 256
 
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bfn = _load_bfn(TEXT8_CFG, TEXT8_CKPT, device)
    except Exception as e:
        print(f"  ✗ Could not build/load text8 model: {e}")
        return None
 
    bayesian_flow = bfn.bayesian_flow                 # DiscreteBayesianFlow, n_classes=27
    distribution_factory = bfn.loss.distribution_factory
    K = bayesian_flow.n_classes
 
    data_shape = (1, seq_len)
    capture_idx = sorted(set([1, 6, 11, 21, 36, 51, 71, 91, n_steps]))  # step i values (t=(i-1)/n_steps)
 
    captured_heatmaps = []
    entropy_trace = []
 
    with torch.no_grad():
        input_params = bayesian_flow.get_prior_input_params(data_shape, device)
 
        for i in range(1, n_steps + 1):
            t = torch.ones(*data_shape, device=device) * (i - 1) / n_steps
            net_inputs = bayesian_flow.params_to_net_inputs(input_params)
            output_params = bfn.net(net_inputs, t)
            output_dist = distribution_factory.get_dist(output_params, input_params, t)
            probs = output_dist.probs  # (1, seq_len, K) — theta_hat
 
            eps = 1e-8
            H = -(probs * (probs + eps).log()).sum(-1).mean().item()
            H_bits = H / math.log(2)
            entropy_trace.append({"step": i - 1, "t": (i - 1) / n_steps, "entropy_bits": H_bits})
 
            if i in capture_idx:
                # Only keep the first `display_len` positions for a readable plot
                captured_heatmaps.append(
                    (i - 1, (i - 1) / n_steps, probs[0, :display_len, :].detach().cpu().numpy())
                )
 
            output_sample = output_dist.sample().reshape(*data_shape)
            alpha = bayesian_flow.get_alpha(i, n_steps)
            y = bayesian_flow.get_sender_dist(output_sample, alpha).sample()
            input_params = bayesian_flow.update_input_params(input_params, y, alpha)
 
    n_panels = len(captured_heatmaps)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 2.5, 5), sharey=True)
 
    im = None
    for ax, (step_i, t_frac, hmap) in zip(axes, captured_heatmaps):
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
        f"BFN text8: probability simplex θ_t over generation steps (first {display_len} of {seq_len} positions)\n"
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
    Unchanged from the original draft — this part never touched the broken API.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
 
    ax = axes[0]
    if entropy_trace:
        ts = [e["t"] for e in entropy_trace]
        Hbits = [e["entropy_bits"] for e in entropy_trace]
        ax.plot(ts, Hbits, color="#4C72B0", lw=2.2, label="BFN H(θ_t) [bits]")
        ax.axhline(math.log2(27), color="#4C72B0", linestyle=":", alpha=0.5,
                   label=f"Uniform prior: log₂(27) ≈ {math.log2(27):.2f} bits")
        ax.fill_between(ts, 0, Hbits, alpha=0.12, color="#4C72B0")
 
    ax.set_xlabel("Generation time t ∈ [0, 1]", fontsize=11)
    ax.set_ylabel("Output entropy H(θ_t)  [bits]", fontsize=11)
    ax.set_title("BFN: entropy collapse during generation\n(text8, 27-char alphabet)", fontsize=10)
    ax.legend(fontsize=9)
 
    ax2 = axes[1]
    t_arr = np.linspace(0, 1, 1000)
 
    beta_min, beta_max = 1e-4, 0.02
    betas = beta_min + t_arr * (beta_max - beta_min)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    sigma_ddpm = np.sqrt(1 - alpha_bars)
 
    sigma_min = 0.001
    bfn_precision = 1 - (1 - sigma_min ** 2) ** t_arr
 
    ax2.plot(t_arr, sigma_ddpm, color="#DD8452", lw=2.0, label="DDPM σ(t) — noise level")
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
 
    try:
        captured = visualize_mnist_flow(n_steps=100, n_samples=8)
        results["mnist_flow"] = "success" if captured else "skipped"
    except Exception as e:
        print(f"  ⚠ MNIST flow skipped: {e}")
        results["mnist_flow"] = f"error: {str(e)}"
 
    try:
        entropy_trace = visualize_text8_simplex(n_steps=100, seq_len=256, display_len=48)
        if entropy_trace:
            plot_entropy_vs_noise(entropy_trace)
            results["entropy_trace_steps"] = len(entropy_trace)
 
            out = RESULTS_DIR / "flow_intermediate_states.json"
            with open(out, "w") as f:
                json.dump({"text8_entropy": entropy_trace}, f, indent=2)
            print(f"  ✓ Entropy trace saved: {out.name}")
    except Exception as e:
        print(f"  ⚠ text8 simplex skipped: {e}")
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