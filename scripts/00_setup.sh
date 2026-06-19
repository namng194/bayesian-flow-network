#!/usr/bin/env bash
# =============================================================================
# 00_setup.sh — Automated environment initialization & data fetching
# =============================================================================
set -euo pipefail

# Lấy đường dẫn tuyệt đối của thư mục gốc project (ngay bên ngoài thư mục scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [1/5] Creating project directories ==="
mkdir -p src checkpoints/{bfn,d3pm,ddpm} data/text8 outputs/{logs,figures,visual_demos}

echo "=== [2/5] Cloning external repositories ==="
cd src
if [ ! -d "bayesian-flow-networks" ]; then
    echo "Cloning official BFN repo..."
    git clone https://github.com/nnaisense/bayesian-flow-networks.git
fi
if [ ! -d "nanoDD" ]; then
    echo "Cloning D3PM Absorbing baseline (nanoDD)..."
    git clone https://github.com/flukeskywalker/nanoDD.git
fi
if [ ! -d "ddpm-torch" ]; then
    echo "Cloning Continuous DDPM baseline..."
    git clone https://github.com/tqch/ddpm-torch.git
fi
cd "$PROJECT_ROOT"

echo "=== [3/5] Setting up Conda Environment ==="
# Khởi tạo env `bfn` từ repo gốc
conda env create -f src/bayesian-flow-networks/env.yml 2>/dev/null || echo "Conda env 'bfn' already exists."

# Cài đặt thêm các package dùng chung cho DDPM, D3PM và Visualization
conda run -n bfn pip install diffusers neptune matplotlib seaborn || true
if [ -f "requirements.txt" ]; then
    conda run -n bfn pip install -r requirements.txt || true
fi

echo "=== [4/5] Downloading pretrained checkpoints (HuggingFace) ==="
cd checkpoints
git lfs install

# Tải BFN Checkpoints
if [ ! -f "bfn/text8_ema.pt" ]; then
    echo "Downloading BFN checkpoints..."
    git clone https://huggingface.co/rupspace/pretrained-BFNs temp_bfn
    mv temp_bfn/*.pt bfn/
    rm -rf temp_bfn
fi

# Tải D3PM / nanoDD Checkpoint
if [ ! -f "d3pm/nanodd_text8_best.pt" ]; then
    echo "Downloading D3PM nanoDD checkpoint..."
    git clone https://huggingface.co/rupspace/nanoDD-D3PM-text8 temp_d3pm
    mv temp_d3pm/ckpt.pt d3pm/nanodd_text8_best.pt
    rm -rf temp_d3pm
fi
cd "$PROJECT_ROOT"

echo "=== [5/5] Preparing data ==="
# nanoDD có script tải và tách train/test text8. Ta sẽ chạy nó rồi link/copy qua thư mục data chung.
conda run -n bfn python src/nanoDD/data/prepare_text8.py
if [ -d "src/nanoDD/data/text8" ] && [ -z "$(ls -A data/text8 2>/dev/null)" ]; then
    cp -r src/nanoDD/data/text8/* data/text8/
fi

echo ""
echo "✓ Setup complete!"
echo "  Repos cloned into    : ./src/"
echo "  Checkpoints saved to : ./checkpoints/"
echo "  Data extracted to    : ./data/text8/"