#!/usr/bin/env python3

"""
Exp 2A
BFN n_steps ablation on pretrained checkpoints.

CRASH RESILIENCE:
    - Results are saved to disk after every single (dataset, n_steps) run.
    - Already-completed runs are skipped on restart (resume from last good result).
    - GPU health is checked between runs; experiment halts cleanly if GPU is faulting.

MEMORY CONSTRAINT FIX (Kubeflow 8GB RAM pod):
    - Instead of calling test.py once with n_repeats=N (loads all N samples into RAM),
      we call test.py multiple times with n_repeats=CHUNK_SIZE, then aggregate stats.
    - Total number of evaluations is preserved → variance stays low → curve stays smooth.
    - Each subprocess only holds CHUNK_SIZE samples in RAM at a time → no OOM.

Default configuration is intentionally MINIMAL.

To scale toward paper-quality experiments:
    - increase N_REPEATS_TEXT8
    - increase N_REPEATS_MNIST
    - add more seeds
    - add more n_steps values
"""

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path


# =============================================================================
# MINIMAL CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

VENV_PYTHON = ROOT / ".venv/bin/python"

BFN_DIR = ROOT / "src/bayesian-flow-networks"

OUTPUT_DIR = ROOT / "outputs/logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1 # [1,2,3,4,5]

CHECKPOINTS = {
    "text8": ROOT / "checkpoints/bfn/text8_ema.pt",
    "mnist": ROOT / "checkpoints/bfn/mnist_ema.pt",
}

CONFIGS = {
    "text8": BFN_DIR / "configs/text8_discrete.yaml",
    "mnist": BFN_DIR / "configs/mnist_discrete.yaml",
}

# None = continuous-time BFN loss (passed as n_steps=0 to upstream test.py)
# See: https://github.com/nnaisense/bayesian-flow-networks README
NSTEPS_TEXT8 = [
    # None,   # → n_steps=0 (continuous-time)
    5,
    10,
    25,
    50,
    100,
    250,
    500,
]

NSTEPS_MNIST = [
    10,
    25,
    50,
    100,
    250,
    500,
    784,    
]

# Total evaluations — same scientific intent as original.
# These are split across multiple subprocess calls (see CHUNK_SIZE below),
# so RAM usage per call stays within the 8GB pod limit.
N_REPEATS_TEXT8 = 2 # 50
N_REPEATS_MNIST = 5 # 200

# Max n_repeats per single test.py subprocess call.
# Tune this if you still OOM: lower = less RAM per call.
# text8 sequences are longer (256 chars) so needs a smaller chunk than MNIST.
CHUNK_SIZE_TEXT8 = 2
CHUNK_SIZE_MNIST = 5

# Seconds to sleep between dataset runs
INTER_RUN_SLEEP_S = 10

def nats_to_bits(nats: float) -> float:
    return nats / math.log(2.0)

# =============================================================================
# GPU HEALTH
# =============================================================================

def check_gpu_health() -> bool:
    """
    Returns True if GPU is healthy, False if it is in an error/reset state.
    Checks for ERR! in nvidia-smi output — the signature of a Blackwell GSP crash.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name,power.draw,temperature.gpu,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    if result.returncode != 0:
        print(f"[ERROR] nvidia-smi failed: {result.stderr.strip()}")
        return False

    output = result.stdout
    if "ERR!" in output or "[N/A]" in output.upper():
        print(f"[ERROR] GPU appears to be in a fault state:\n{output}")
        return False

    print(f"[GPU] {output.strip()}")
    return True


# =============================================================================
# CUDA CHECK
# =============================================================================

def verify_cuda():

    result = subprocess.run(
        [
            str(VENV_PYTHON),
            "-c",
            (
                "import torch;"
                "print(torch.cuda.is_available());"
                "print(torch.cuda.device_count())"
            ),
        ],
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().splitlines()

    if len(lines) < 2:
        raise RuntimeError("Failed CUDA verification")

    cuda_available = lines[0].strip() == "True"
    device_count = int(lines[1])

    if not cuda_available:
        raise RuntimeError("CUDA unavailable")

    if device_count == 0:
        raise RuntimeError("No CUDA device detected")

    print(f"[OK] CUDA devices = {device_count}")


# =============================================================================
# PARSERS
# =============================================================================

LOSS_RE = re.compile(
    r"Loss is\s+([0-9eE+\-.]+)\s+\+-\s+([0-9eE+\-.]+)"
)

RECON_RE = re.compile(
    r"Reconstruction Loss is\s+([0-9eE+\-.]+)\s+\+-\s+([0-9eE+\-.]+)"
)

# Handles both "Total loss mean = 1.234" (fixed test.py)
# and "Total loss mean = tensor(1.234)" (unfixed upstream test.py)
TOTAL_RE = re.compile(
    r"Total loss mean\s*=\s*(?:tensor\()?([0-9eE+\-.]+)\)?"
)


def parse_metrics(stdout: str):

    loss_mean = None
    loss_err = None
    recon_mean = None
    recon_err = None
    total_loss = None

    m = LOSS_RE.search(stdout)
    if m:
        loss_mean = float(m.group(1))
        loss_err = float(m.group(2))

    m = RECON_RE.search(stdout)
    if m:
        recon_mean = float(m.group(1))
        recon_err = float(m.group(2))

    m = TOTAL_RE.search(stdout)
    if m:
        total_loss = float(m.group(1))

    return {
        "loss_mean": loss_mean,
        "loss_err": loss_err,
        "recon_mean": recon_mean,
        "recon_err": recon_err,
        "total_loss": total_loss,
    }


# =============================================================================
# SINGLE SUBPROCESS CALL
# Runs test.py with a fixed n_repeats (the chunk size).
# Returns raw metrics dict, or error dict.
# =============================================================================

def run_single_chunk(
    dataset: str,
    n_steps,        # Python-level sentinel: None = continuous-time
    n_repeats: int, # chunk size — kept small to fit in 8GB RAM
    seed: int,
) -> dict:
    """
    One subprocess call to test.py with n_repeats = chunk_size.
    RAM usage is bounded by chunk_size, not total N_REPEATS.
    """
    # BUG 1 FIX: upstream test.py uses n_steps=0 for continuous-time loss,
    # not omitting the flag.  Omitting it causes OmegaConf MissingMandatoryValue.
    cli_n_steps = 0 if n_steps is None else n_steps

    cmd = [
        str(VENV_PYTHON),
        "test.py",
        f"seed={seed}",
        f"config_file={CONFIGS[dataset]}",
        f"load_model={CHECKPOINTS[dataset]}",
        f"n_repeats={n_repeats}",
        f"n_steps={cli_n_steps}",   # always explicit; never omit
    ]

    result = subprocess.run(
        cmd,
        cwd=BFN_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {
            "error": True,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }

    metrics = parse_metrics(result.stdout)
    return {"error": False, **metrics}


# =============================================================================
# CHUNKED RUN
#
# Core fix for the 8GB RAM OOM problem.
#
# Instead of:
#   test.py n_repeats=50   ← loads 50 samples into RAM at once → OOM
#
# We do:
#   test.py n_repeats=1    ← 1 sample in RAM
#   test.py n_repeats=1    ← 1 sample in RAM
#   ... × 50 times
#   aggregate mean + stderr across all 50 chunk results
#
# Scientific validity:
#   - Total evaluations = N_REPEATS (unchanged)
#   - Each chunk uses a different seed → independent samples
#   - Final mean/stderr computed from all chunk means → equivalent to
#     running n_repeats=50 in one call, by the law of large numbers
#   - Error bars remain meaningful for the BPC vs n_steps curve
# =============================================================================

def run_chunked(
    dataset: str,
    n_steps,
    n_repeats: int,
    chunk_size: int,
    seed: int,
) -> dict:
    """
    Splits n_repeats into ceil(n_repeats / chunk_size) subprocess calls.
    Aggregates total_loss mean and standard error across chunks.
    """
    n_chunks = math.ceil(n_repeats / chunk_size)
    chunk_losses = []
    errors = []

    print(f"  [chunked] {n_chunks} chunks × up to {chunk_size} repeats each")

    for i in range(n_chunks):
        actual_repeats = min(chunk_size, n_repeats - i * chunk_size)
        chunk_seed = seed + i  # different seed per chunk → independent samples

        chunk_result = run_single_chunk(
            dataset=dataset,
            n_steps=n_steps,
            n_repeats=actual_repeats,
            seed=chunk_seed,
        )

        if chunk_result.get("error"):
            print(f"    chunk {i+1}/{n_chunks}: ERROR (rc={chunk_result.get('returncode')})")
            errors.append(chunk_result)
            continue

        loss = chunk_result.get("total_loss")
        if loss is None:
            print(f"    chunk {i+1}/{n_chunks}: could not parse total_loss, skipping")
            continue

        chunk_losses.append(loss)
        bits = nats_to_bits(loss)
        print(f"    chunk {i+1}/{n_chunks}: loss={loss:.4f}  bits={bits:.4f}")

    if not chunk_losses:
        return {
            "dataset": dataset,
            "seed": seed,
            "n_steps": n_steps,
            "error": True,
            "error_detail": "all chunks failed or returned no parseable loss",
            "chunk_errors": errors,
        }

    # Aggregate: mean and standard error of the mean across chunks
    n = len(chunk_losses)
    mean_loss = sum(chunk_losses) / n
    if n > 1:
        variance = sum((x - mean_loss) ** 2 for x in chunk_losses) / (n - 1)
        stderr = math.sqrt(variance / n)
    else:
        stderr = float("nan")

    mean_bits = nats_to_bits(mean_loss)

    print(f"  [aggregated] n_chunks_ok={n}  loss={mean_loss:.4f} ± {stderr:.4f}  "
          f"bits={mean_bits:.4f}")

    return {
        "dataset": dataset,
        "seed": seed,
        "n_steps": n_steps,
        # Aggregated stats — drop-in replacement for original single-run fields
        "loss_mean": mean_loss,
        "loss_err": stderr,
        "recon_mean": None,   # not aggregated across chunks; extend if needed
        "recon_err": None,
        "total_loss": mean_loss,
        # WARNING: assumes total_loss is normalized nats.
        # Verify model implementation before calling it BPC.
        "bits_estimate": mean_bits,
        "bits_stderr": nats_to_bits(stderr) if not math.isnan(stderr) else None,
        # Provenance
        "n_repeats_requested": n_repeats,
        "n_chunks_ok": n,
        "chunk_size": chunk_size,
        "chunk_losses_nats": chunk_losses,
        "error": False,
    }


# =============================================================================
# RESULT PERSISTENCE
#
# Each (dataset, n_steps) result is saved immediately after it completes.
# This means a crash only loses the run that was in-progress, not all prior work.
#
# Layout on disk:
#   outputs/logs/bfn_text8_nsteps/
#       step_None.json
#       step_5.json
#       step_10.json
#       ...
#   outputs/logs/bfn_mnist_nsteps/
#       step_10.json
#       ...
# =============================================================================

def result_dir(dataset: str) -> Path:
    d = OUTPUT_DIR / f"bfn_{dataset}_nsteps"
    d.mkdir(parents=True, exist_ok=True)
    return d


def result_path(dataset: str, n_steps) -> Path:
    key = str(n_steps)   # "None", "5", "10", …
    return result_dir(dataset) / f"step_{key}.json"


def save_result(result: dict):
    path = result_path(result["dataset"], result["n_steps"])
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> saved: {path}")


def load_result(dataset: str, n_steps) -> dict | None:
    path = result_path(dataset, n_steps)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def already_done(dataset: str, n_steps) -> bool:
    r = load_result(dataset, n_steps)
    if r is None:
        return False
    if r.get("error"):
        return False   # retry errored runs
    return True


def assemble_summary(dataset: str, nsteps_list, n_repeats) -> dict:
    """Collect all per-step results into a single summary JSON (matches original format)."""
    results = []
    for n_steps in nsteps_list:
        r = load_result(dataset, n_steps)
        if r is not None:
            results.append(r)
    return {
        "experiment": "BFN n_steps ablation",
        "dataset": dataset,
        "seed": SEED,
        "n_repeats": n_repeats,
        "checkpoint": str(CHECKPOINTS[dataset]),
        "config": str(CONFIGS[dataset]),
        "results": results,
    }


# =============================================================================
# SWEEP
# =============================================================================

def sweep(
    dataset: str,
    nsteps_list,
    n_repeats: int,
    chunk_size: int,
):
    for n_steps in nsteps_list:

        # --- Resume: skip already-completed runs ---
        if already_done(dataset, n_steps):
            print(f"[SKIP] {dataset} n_steps={n_steps} — already done")
            continue

        # --- GPU health gate ---
        if not check_gpu_health():
            print()
            print("=" * 80)
            print("ABORTING: GPU is in a fault state before run.")
            print("The RTX PRO 6000 Blackwell requires a full PSU power cycle")
            print("to recover from a GSP firmware crash (Xid 62/119/154).")
            print("After power cycle, restart this script — it will resume.")
            print("=" * 80)
            sys.exit(1)

        print()
        print("=" * 80)
        print(f"{dataset}  n_steps={n_steps}  "
              f"(n_repeats={n_repeats}, chunk_size={chunk_size})")
        print("=" * 80)

        # --- Chunked run ---
        result = run_chunked(
            dataset=dataset,
            n_steps=n_steps,
            n_repeats=n_repeats,
            chunk_size=chunk_size,
            seed=SEED,
        )

        # --- Save immediately, regardless of error ---
        save_result(result)

        if result.get("error"):
            print(f"[WARN] Run errored; result saved, continuing to next n_steps.")

        # --- Brief cooldown between runs ---
        if INTER_RUN_SLEEP_S > 0:
            print(f"  [cooldown {INTER_RUN_SLEEP_S}s]")
            time.sleep(INTER_RUN_SLEEP_S)

    # --- Write consolidated summary JSON (same format as original) ---
    summary = assemble_summary(dataset, nsteps_list, n_repeats)
    out_file = OUTPUT_DIR / f"bfn_{dataset}_nsteps.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {out_file}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    verify_cuda()

    sweep(
        dataset="text8",
        nsteps_list=NSTEPS_TEXT8,
        n_repeats=N_REPEATS_TEXT8,
        chunk_size=CHUNK_SIZE_TEXT8,
    )

    sweep(
        dataset="mnist",
        nsteps_list=NSTEPS_MNIST,
        n_repeats=N_REPEATS_MNIST,
        chunk_size=CHUNK_SIZE_MNIST,
    )