#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh — Master automation script for BFN Seminar
#
# Environment target: uv + .venv
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  BFN vs Baselines (D3PM, DDPM) — Full Seminar Pipeline   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Ensure virtual environment exists ────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "⚠️  .venv not found."
    echo "Run scripts/00_setup.sh first."
    exit 1
fi

# Activate venv
source .venv/bin/activate
## If no python in .venv, reinstall uv and python 3.11
# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.local/bin/env
# uv python install 3.11
# ls -l /home/jovyan/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11

echo "✓ Using Python: $(which python)"
python --version

# ── 0. Setup Phase ────────────────────────────────────────────────────────────
if [ ! -d "src/bayesian-flow-networks" ]; then
    echo "[00] First-time setup and downloading weights..."
    bash scripts/00_setup.sh
else
    echo "[00] Repos and checkpoints already exist — skipping setup."
fi

# ── 1. Exp 1: Continuous Image Domain (DDPM vs BFN) ──────────────────────────
echo ""
echo "=== [Exp 1a] Training DDPM baseline on CIFAR-10 (Quick mode: ~3h) ==="
# nohup .venv/bin/python scripts/exp1a_train_ddpm.py --dataset cifar10 --quick > exp1a_train_ddpm_cifar10.log 2>&1 &
.venv/bin/python scripts/exp1a_train_ddpm.py --dataset cifar10 --quick

echo ""
echo "=== [Exp 1b] Sampling speed benchmark: BFN vs DDPM/DDIM ==="
.venv/bin/python scripts/exp1b_sampling_time_benchmark.py

# ── 2. Exp 2: Discrete Text Domain (D3PM vs BFN) ─────────────────────────────
echo ""
echo "=== [Exp 2a] BFN n_steps ablation sweep (text8 + MNIST) (~1h) ==="
.venv/bin/python scripts/exp2a_nsteps_ablation.py 
# nohup .venv/bin/python -u scripts/exp2a_nsteps_ablation.py > exp2a_nsteps_ablation.log 2>&1 &
# Note: edit code at bayesian-flow-network/src/bayesian-flow-networks/test.py line 39

echo ""
echo "=== [Exp 2b] Training D3PM Absorbing baseline on text8 (Quick mode: ~5h) ==="
.venv/bin/python scripts/exp2b_train_d3pm.py --quick
# nohup .venv/bin/python scripts/exp2b_train_d3pm.py --quick > exp2b_train_d3pm.log 2>&1 &
# Note: edit code at ./nanoDD/train.py line 246
# cat > outputs/logs/d3pm_text8_result.json

echo ""
echo "=== [Exp 2c] Generating visual samples for qualitative analysis ==="
.venv/bin/python scripts/exp2c_generate_samples.py --dataset mnist
.venv/bin/python scripts/exp2c_generate_samples.py --dataset cifar10
.venv/bin/python scripts/exp2c_generate_samples.py --dataset text8

echo ""
echo "=== [Exp 2d] Generating intermediate plot results ==="
.venv/bin/python scripts/exp2d_plot_results.py
# nohup .venv/bin/python scripts/exp2a_nsteps_ablation.py > exp2a_nsteps_ablation.log 2>&1 &

# ── 3. Exp 3: Visualization & Intuition (Seminar Core) ───────────────────────
echo ""
echo "=== [Exp 3] Generating Bayesian flow process visualizations ==="
.venv/bin/python scripts/exp3_bayesian_flow_visualization.py
# nohup .venv/bin/python scripts/exp2a_nsteps_ablation.py > exp2a_nsteps_ablation.log 2>&1 &

# ── 4. Final Aggregation ──────────────────────────────────────────────────────
echo ""
echo "=== [FINAL] Consolidating all metrics into presentation-ready figures ==="
.venv/bin/python scripts/exp_final_plots.py
# nohup .venv/bin/python scripts/exp2a_nsteps_ablation.py > exp2a_nsteps_ablation.log 2>&1 &

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE! Key slide assets generated:          ║"
echo "║                                                          ║"
echo "║  outputs/figures/FINAL_fig1_comparison_table.png         ║"
echo "║  outputs/figures/FINAL_fig2_speed_quality.png            ║"
echo "║  outputs/figures/FINAL_fig3_entropy_collapse.png         ║"
echo "║  outputs/figures/FINAL_fig4_sample_grid.png              ║"
echo "║  outputs/visual_demos/mnist_crystallization.gif          ║"
echo "╚══════════════════════════════════════════════════════════╝"
