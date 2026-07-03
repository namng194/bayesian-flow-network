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

import argparse
import gc
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

BFN_DIR = Path(__file__).resolve().parents[1] / "src/bayesian-flow-networks"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "outputs/logs/samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BFN_DIR))

from data import batch_to_images, batch_to_str


SAMPLE_CONFIGS = {
    "mnist": {
        "config_file": BFN_DIR / "configs/mnist_discrete.yaml",
        "load_model": Path(__file__).resolve().parents[1] / "checkpoints/bfn/mnist_ema.pt",
        "shape": "[16,28,28,1]",
        "n_steps_list": [5, 10, 25, 100, 784],
        "render": "image",
    },
    "cifar10": {
        "config_file": BFN_DIR / "configs/cifar10_discretized_256bins.yaml",
        "load_model": Path(__file__).resolve().parents[1] / "checkpoints/bfn/cifar10_256d_ema.pt",
        "shape": "[16,32,32,3]",
        "n_steps_list": [10, 50, 100, 1000],
        "render": "image",
    },
    "text8": {
        "config_file": BFN_DIR / "configs/text8_discrete.yaml",
        "load_model": Path(__file__).resolve().parents[1] / "checkpoints/bfn/text8_ema.pt",
        "shape": "[4,256]",
        "n_steps_list": [5, 10, 25, 50, 100, 500],
        "render": "text",
    },
}


def cleanup():
    gc.collect()
    plt.close("all")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sample(dataset, cfg, n_steps, seed=42):

    save_file = SAMPLES_DIR / f"{dataset}_n{n_steps:04d}.pt"

    cmd = [
        sys.executable,
        str(BFN_DIR / "sample.py"),
        f"seed={seed}",
        f"config_file={cfg['config_file']}",
        f"load_model={cfg['load_model']}",
        f"samples_shape={cfg['shape']}",
        f"n_steps={n_steps}",
        f"save_file={save_file}",
    ]

    print(f"\n===== {dataset} | n_steps={n_steps} =====")

    proc = subprocess.Popen(
        cmd,
        cwd=BFN_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="")

    proc.wait()

    if proc.returncode != 0:
        print("Sampling failed.")
        cleanup()
        return None

    return save_file


def render(dataset, mode, n_steps, pt_file):

    samples = torch.load(pt_file, map_location="cpu")

    if mode == "image":

        fig = batch_to_images(samples)

        out_png = SAMPLES_DIR / f"{dataset}_n{n_steps:04d}.png"

        fig.savefig(
            out_png,
            dpi=120,
            bbox_inches="tight",
        )

        plt.close(fig)

    else:

        texts = batch_to_str(samples)

        out_txt = SAMPLES_DIR / f"{dataset}_n{n_steps:04d}.txt"

        with open(out_txt, "w") as f:

            for i, t in enumerate(texts):
                f.write(f"[sample {i+1}]\n")
                f.write(t)
                f.write("\n\n")

    del samples

    cleanup()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10", "text8"],
        required=True,
    )

    args = parser.parse_args()

    cfg = SAMPLE_CONFIGS[args.dataset]

    print("=" * 60)
    print(args.dataset)
    print("=" * 60)

    for n_steps in cfg["n_steps_list"]:

        pt = run_sample(
            args.dataset,
            cfg,
            n_steps,
        )

        if pt is not None:
            render(
                args.dataset,
                cfg["render"],
                n_steps,
                pt,
            )

    cleanup()

    print("\nDone.")


if __name__ == "__main__":
    main()