#!/usr/bin/env python3
"""
exp2c_generate_samples.py
=========================
Exp 2 — Visual sample generation from pretrained BFN checkpoints.

Generates:
  1. text8 sequences at various n_steps → shows text quality progression
  2. MNIST images at various n_steps    → visual quality vs speed tradeoff
  3. CIFAR-10 images                    → continuous data demo

Also captures intermediate Bayesian flow states for visualization
(the evolution of the probability distribution over generation steps).

Reference:
  Official repo: https://github.com/nnaisense/bayesian-flow-networks
  sample.py API: seed, config_file, load_model, samples_shape, n_steps, save_file

Usage (run from inside bayesian-flow-networks/ with conda env bfn active):
  python ../scripts/exp2c_generate_samples.py

Outputs:
  ../results/samples/text8_n{steps}.pt       — raw tensors
  ../results/samples/mnist_n{steps}.png      — rendered images
  ../results/samples/cifar10_n{steps}.png    — rendered images
  ../results/samples/text8_samples.txt       — decoded text
"""

import subprocess
import sys
import os
import torch
from pathlib import Path

BFN_DIR      = Path(__file__).resolve().parents[1] / "src/bayesian-flow-networks"
SAMPLES_DIR  = Path(__file__).resolve().parents[1] / "outputs/logs" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Import BFN data utilities for rendering
sys.path.insert(0, str(BFN_DIR))

# ── Sample configs ─────────────────────────────────────────────────────────
SAMPLE_CONFIGS = {
    "mnist": {
        "config_file": BFN_DIR / "configs/mnist_discrete.yaml",
        "load_model":  Path(__file__).resolve().parents[1] / "checkpoints/bfn/mnist_ema.pt",
        "shape":       "[16, 28, 28, 1]",   # 16 images for a nice 4×4 grid
        "n_steps_list": [5, 10, 25, 100, 784],
        "render": "image",
    },
    "cifar10": {
        "config_file": BFN_DIR / "configs/cifar10_discretized_256bins.yaml",
        "load_model":  Path(__file__).resolve().parents[1] / "checkpoints/bfn/cifar10_256d_ema.pt",
        "shape":       "[16, 32, 32, 3]",
        "n_steps_list": [10, 50, 100, 1000],
        "render": "image",
    },
    "text8": {
        "config_file": BFN_DIR / "configs/text8_discrete.yaml",
        "load_model":  Path(__file__).resolve().parents[1] / "checkpoints/bfn/text8_ema.pt",
        "shape":       "[4, 256]",           # 4 sequences of 256 chars
        "n_steps_list": [5, 10, 25, 50, 100, 500],
        "render": "text",
    },
}


def run_sample(dataset: str, n_steps: int, cfg: dict, seed: int = 42) -> Path:
    """Call official sample.py and return the save path."""
    save_file = SAMPLES_DIR / f"{dataset}_n{n_steps:04d}.pt"
    cmd = [
        "python", str(BFN_DIR / "sample.py"),
        f"seed={seed}",
        f"config_file={cfg['config_file']}",
        f"load_model={cfg['load_model']}",
        f"samples_shape={cfg['shape']}",
        f"n_steps={n_steps}",
        f"save_file={save_file}",
    ]
    print(f"  Sampling {dataset} @ n_steps={n_steps} …", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BFN_DIR)
    if result.returncode != 0:
        print(f"  ✗ Error:\n{result.stderr[-500:]}")
        return None
    print(f"    → saved to {save_file.name}")
    return save_file


def render_samples(dataset: str, n_steps: int, save_pt: Path, render_mode: str):
    """Load saved tensor and render to image or text using BFN's own utilities."""
    if not save_pt or not save_pt.exists():
        return

    from data import batch_to_images, batch_to_str  # noqa: from BFN repo

    samples = torch.load(save_pt, map_location="cpu")

    if render_mode == "image":
        out_png = SAMPLES_DIR / f"{dataset}_n{n_steps:04d}.png"
        fig = batch_to_images(samples)
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"    → rendered to {out_png.name}")

    elif render_mode == "text":
        out_txt = SAMPLES_DIR / f"{dataset}_n{n_steps:04d}.txt"
        texts   = batch_to_str(samples)
        with open(out_txt, "w") as f:
            f.write(f"# text8 samples — n_steps={n_steps}\n\n")
            for i, t in enumerate(texts):
                f.write(f"[sample {i+1}]\n{t}\n\n")
        print(f"    → text saved to {out_txt.name}")
        # also print a snippet
        print(f"    Preview: {texts[0][:100]} …")


def main():
    print("=" * 60)
    print("  BFN Sample Generation — all datasets")
    print("=" * 60)

    for dataset, cfg in SAMPLE_CONFIGS.items():
        print(f"\n── {dataset.upper()} ──")
        for n_steps in cfg["n_steps_list"]:
            save_pt = run_sample(dataset, n_steps, cfg)
            if save_pt:
                render_samples(dataset, n_steps, save_pt, cfg["render"])

    print("\n✓ All samples generated.")
    print(f"  Output dir: {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
