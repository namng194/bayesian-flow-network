#!/usr/bin/env python3
"""
exp2a_nsteps_ablation.py
========================
Exp 2 — Core ablation: sweep n_steps for BFN pretrained text8 model.

Calls the official BFN test.py with varying n_steps, collects BPC at each
step count, and saves results to JSON for plotting.

Reference:
  Official repo : https://github.com/nnaisense/bayesian-flow-networks
  Paper         : Graves et al. 2023 — arxiv 2308.07037

Usage (run from inside bayesian-flow-networks/ with conda env bfn active):
  python ../scripts/exp2a_nsteps_ablation.py

Outputs:
  ../results/bfn_text8_nsteps.json
  ../results/bfn_mnist_nsteps.json
"""

import subprocess
import json
import math
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BFN_DIR      = Path(__file__).resolve().parents[1] / "src/bayesian-flow-networks"
RESULTS_DIR  = Path(__file__).resolve().parents[1] / "outputs/logs"
RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINTS = {
    "text8": Path(__file__).resolve().parents[1] / "checkpoints/bfn/text8_ema.pt",
    "mnist": Path(__file__).resolve().parents[1] / "checkpoints/bfn/mnist_ema.pt",
}
CONFIGS = {
    "text8": BFN_DIR / "configs/text8_discrete.yaml",
    "mnist": BFN_DIR / "configs/mnist_discrete.yaml",
}

# n_steps=0 → continuous-time loss (BFN special, no discrete-time equiv)
# n_steps=784 → maximum for MNIST (28×28 pixels)
NSTEPS_TEXT8 = [0, 5, 10, 25, 50, 100, 250, 500]   # 0=continuous-time
NSTEPS_MNIST = [10, 25, 50, 100, 250, 500, 784]

N_REPEATS_TEXT8 = 5   # raise to 50+ for final results (slower but lower variance)
N_REPEATS_MNIST = 50  # paper uses 2000 — reduce for speed

# ── Helpers ───────────────────────────────────────────────────────────────────
def nats_to_bpc(nats: float) -> float:
    """Convert nats-per-dim to bits-per-character (divide by ln 2)."""
    return nats / math.log(2)


def run_bfn_test(dataset: str, n_steps: int, n_repeats: int, seed: int = 1) -> dict:
    """
    Call official test.py, parse stdout for the loss value.

    test.py prints a line like:
        Loss: 0.9765 nats  (or similar)
    We capture it and convert to BPC.
    """
    cmd = [
        "python", str(BFN_DIR / "test.py"),
        f"seed={seed}",
        f"config_file={CONFIGS[dataset]}",
        f"load_model={CHECKPOINTS[dataset]}",
        f"n_steps={n_steps}",
        f"n_repeats={n_repeats}",
    ]
    print(f"  Running: n_steps={n_steps}, dataset={dataset} …", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, cwd=BFN_DIR
    )
    if result.returncode != 0:
        print(f"  ✗ STDERR:\n{result.stderr[-800:]}")
        return {"n_steps": n_steps, "loss_nats": None, "bpc": None, "error": True}

    # Parse loss from stdout — official test.py prints "Loss: X.XXXX nats"
    loss_nats = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if "loss" in line.lower() and "nats" in line.lower():
            try:
                # e.g. "Loss: 0.9765 nats" or "Test loss: 0.9765 nats"
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.replace(".", "").replace("-", "").isdigit():
                        loss_nats = float(p)
                        break
            except (ValueError, IndexError):
                pass

    if loss_nats is None:
        # fallback: try last float on last non-empty line
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if lines:
            toks = lines[-1].split()
            for tok in reversed(toks):
                try:
                    loss_nats = float(tok)
                    break
                except ValueError:
                    pass

    bpc = nats_to_bpc(loss_nats) if loss_nats is not None else None
    print(f"    → loss={loss_nats:.4f} nats  /  BPC={bpc:.4f}" if bpc else "    → parse failed")
    return {
        "n_steps": n_steps,
        "loss_nats": loss_nats,
        "bpc": bpc,
        "stdout": result.stdout[-500:],  # keep tail for debugging
    }


# ── Main sweep ────────────────────────────────────────────────────────────────
def sweep(dataset: str, nsteps_list: list, n_repeats: int):
    print(f"\n{'='*60}")
    print(f"  BFN n_steps ablation — {dataset.upper()}")
    print(f"{'='*60}")
    results = []
    for n in nsteps_list:
        r = run_bfn_test(dataset, n, n_repeats)
        results.append(r)

    out_path = RESULTS_DIR / f"bfn_{dataset}_nsteps.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset, "model": "BFN (pretrained)", "results": results}, f, indent=2)
    print(f"\n  ✓ Saved to {out_path}")
    return results


if __name__ == "__main__":
    sweep("text8", NSTEPS_TEXT8, N_REPEATS_TEXT8)
    sweep("mnist", NSTEPS_MNIST, N_REPEATS_MNIST)
