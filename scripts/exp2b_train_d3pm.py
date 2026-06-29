#!/usr/bin/env python3
"""
exp2b_train_d3pm.py
===================
Exp 2 — Train D3PM Absorbing baseline on text8 using nanoDD.

nanoDD is a clean, scalable PyTorch re-implementation of D3PM Absorbing
by the same author (rupspace) who also released the BFN pretrained checkpoints.
This makes it directly comparable: same HuggingFace ecosystem, same dataset.

Reference:
  nanoDD repo : https://github.com/flukeskywalker/nanoDD
  D3PM paper  : Austin et al. 2021 — NeurIPS — "Structured Denoising Diffusion
                Models in Discrete State Spaces"
  BFN baseline: Graves et al. report D3PM absorbing = 1.45 BPC on text8 (Table 3)

Usage (run from inside nanoDD/ with conda env bfn active):
  python ../scripts/exp2b_train_d3pm.py [--quick]

Args:
  --quick  : reduced model (6 layers) + shorter training, good for prototyping
             Expected result: ~1.45–1.60 BPC (vs 1.37 for full nanoDD model)

Outputs (in nanoDD/out/):
  ckpt.pt   — best checkpoint by val loss
  log.csv   — training curve (step, train_loss, val_loss)

Time estimates (single RTX PRO Blackwell 6000, fp16):
  --quick  : ~3–5 hours   → sufficient for comparison table
  full     : ~20–30 hours → replicates nanoDD's 1.37 BPC
"""

import subprocess
import sys
import json
import csv
import os
from pathlib import Path
from datetime import datetime

NANOD_DIR   = Path(__file__).resolve().parents[1] / "src/nanoDD"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs/logs"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Quick-mode config (override nanoDD defaults for faster training) ────────
# Full nanoDD text8 config: 12 layers, ~70M params, T=1000 diffusion steps
# Quick config: 6 layers, ~25M params — enough to reach ~1.50 BPC in 3–5h
QUICK_OVERRIDES = {
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 512,
    "max_iters": 50_000,       # full = ~200k
    "eval_interval": 2_000,
    "eval_iters": 100,
    "batch_size": 128,
    "block_size": 256,          # text8 sequence length from BFN paper
    "dropout": 0.0,
    "learning_rate": 3e-4,
    "warmup_iters": 2_000,
    "diffusion_steps": 1000,    # T in D3PM — absorbing variant
    "out_dir": str(Path(__file__).resolve().parents[1] / "checkpoints/d3pm/quick"),
}


# def build_train_cmd(quick: bool = False) -> list:
#     """Build the nanoDD train.py command."""
#     cmd = ["python", str(NANOD_DIR / "train.py"), "d3pm_text8"]
#     if quick:
#         for k, v in QUICK_OVERRIDES.items():
#             cmd.append(f"--{k}={v}")
#     return cmd
def build_train_cmd(quick: bool = False) -> list:

    if quick:
        print(
            "[WARN] nanoDD does not support quick-mode overrides. "
            "Falling back to author's d3pm_text8 config."
        )

    return [
        "python",
        str(NANOD_DIR / "train.py"),
        "d3pm_text8",
    ]


def stream_training(cmd: list, log_path: Path):
    """Run training, stream output, and save log."""
    print(f"\n{'='*60}")
    print("  Training D3PM Absorbing on text8")
    print(f"  Command: {' '.join(cmd[:5])} ...")
    print(f"  Log   : {log_path}")
    print(f"  Start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    with open(log_path, "w") as logf:
        logf.write("timestamp,step,train_loss,val_loss\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=NANOD_DIR,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            # nanoDD logs lines like: "step 5000: train loss 3.421, val loss 3.315"
            if "step" in line and "loss" in line:
                try:
                    parts = line.strip().split()
                    step = int(parts[1].rstrip(":"))
                    train_loss = float(parts[parts.index("train") + 2].rstrip(","))
                    val_loss   = float(parts[parts.index("val")   + 2])
                    ts = datetime.now().strftime("%H:%M:%S")
                    logf.write(f"{ts},{step},{train_loss:.6f},{val_loss:.6f}\n")
                    logf.flush()
                except (ValueError, IndexError):
                    pass
        process.wait()
    return process.returncode


def evaluate_d3pm(ckpt_path: Path) -> dict:
    """
    Run nanoDD evaluate.py on the test split.
    Returns dict with bpc and raw loss (nats-per-token for text8 = BPC directly
    since alphabet size=27, but nanoDD reports bits-per-character natively).
    """
    print(f"\n  Evaluating checkpoint: {ckpt_path}")
    cmd = [
        "python", str(NANOD_DIR / "evaluate.py"),
        str(ckpt_path),
        "--split", "test",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=NANOD_DIR)
    print(result.stdout)
    # nanoDD evaluate.py prints "Test loss (BPC): X.XXXX"
    bpc = None
    for line in result.stdout.splitlines():
        if "bpc" in line.lower() or "bits" in line.lower():
            try:
                bpc = float(line.strip().split()[-1])
            except ValueError:
                pass
    return {"model": "D3PM Absorbing (nanoDD)", "dataset": "text8", "bpc": bpc}


if __name__ == "__main__":
    quick = "--quick" in sys.argv

    cmd      = build_train_cmd(quick=quick)
    log_path = RESULTS_DIR / f"d3pm_text8_train_{'quick' if quick else 'full'}.csv"
    rc       = stream_training(cmd, log_path)
 
    if rc != 0:
        print(f"\n✗ Training exited with code {rc}")
        sys.exit(rc)

    # Evaluate best checkpoint
    out_dir   = Path(__file__).resolve().parents[1] / "checkpoints/d3pm" / ("quick" if quick else "full")
    ckpt_path = out_dir / "ckpt.pt"
    eval_res  = evaluate_d3pm(ckpt_path)

    out_path = RESULTS_DIR / "d3pm_text8_result.json"
    with open(out_path, "w") as f:
        json.dump(eval_res, f, indent=2)

    print(f"\n✓ D3PM result saved to {out_path}")
    print(f"  BPC = {eval_res['bpc']}")
    print(f"\n  Paper reference (Austin et al. 2021): 1.45 BPC")
    print(f"  BFN reference  (Graves et al. 2023) : 1.41 BPC")
