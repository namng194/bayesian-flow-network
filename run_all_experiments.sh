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
python scripts/exp1a_train_ddpm.py --dataset cifar10 --quick

echo ""
echo "=== [Exp 1b] Sampling speed benchmark: BFN vs DDPM/DDIM ==="
python scripts/exp1b_sampling_time_benchmark.py

# ── 2. Exp 2: Discrete Text Domain (D3PM vs BFN) ─────────────────────────────
echo ""
echo "=== [Exp 2a] BFN n_steps ablation sweep (text8 + MNIST) (~1h) ==="
python scripts/exp2a_nsteps_ablation.py

echo ""
echo "=== [Exp 2b] Training D3PM Absorbing baseline on text8 (Quick mode: ~5h) ==="
python scripts/exp2b_train_d3pm.py --quick

echo ""
echo "=== [Exp 2c] Generating visual samples for qualitative analysis ==="
python scripts/exp2c_generate_samples.py

echo ""
echo "=== [Exp 2d] Generating intermediate plot results ==="
python scripts/exp2d_plot_results.py

# ── 3. Exp 3: Visualization & Intuition (Seminar Core) ───────────────────────
echo ""
echo "=== [Exp 3] Generating Bayesian flow process visualizations ==="
python scripts/exp3_bayesian_flow_visualization.py

# ── 4. Final Aggregation ──────────────────────────────────────────────────────
echo ""
echo "=== [FINAL] Consolidating all metrics into presentation-ready figures ==="
python scripts/exp_final_plots.py

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
