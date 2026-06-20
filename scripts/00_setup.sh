#!/usr/bin/env bash
# =============================================================================
# 00_setup.sh — Automated environment initialization & data fetching
# Uses uv + .venv (NO CONDA)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [1/6] Creating project directories ==="
mkdir -p \
    src \
    checkpoints/{bfn,d3pm,ddpm} \
    data/text8 \
    outputs/{logs,figures,visual_demos}

echo "=== [2/6] Creating uv virtual environment ==="

if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.11 --seed
fi

source .venv/bin/activate

echo "Python executable: $(which python)"
python --version

echo "=== [3/6] Cloning external repositories ==="

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

echo "=== [4/6] Installing Python dependencies ==="

uv pip install --upgrade pip setuptools wheel

# PyTorch mặc định.
# Không ép CUDA, không cài wheel CUDA riêng.
uv pip install torch torchvision torchaudio

uv pip install \
    diffusers \
    transformers \
    accelerate \
    datasets \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    pillow \
    tqdm \
    einops \
    scikit-learn \
    neptune \
    huggingface_hub \
    tensorboard \
    imageio \
    imageio-ffmpeg \
    gitpython

if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
fi

# Optional repo-specific requirements
if [ -f "src/bayesian-flow-networks/requirements.txt" ]; then
    uv pip install -r src/bayesian-flow-networks/requirements.txt || true
fi

if [ -f "src/ddpm-torch/requirements.txt" ]; then
    uv pip install -r src/ddpm-torch/requirements.txt || true
fi

if [ -f "src/nanoDD/requirements.txt" ]; then
    uv pip install -r src/nanoDD/requirements.txt || true
fi

echo "=== [5/6] Downloading pretrained checkpoints ==="

cd checkpoints

git lfs install

if [ ! -f "bfn/text8_ema.pt" ]; then
    echo "Downloading BFN checkpoints..."
    git clone https://huggingface.co/rupspace/pretrained-BFNs temp_bfn
    mv temp_bfn/*.pt bfn/
    rm -rf temp_bfn
fi

if [ ! -f "d3pm/nanodd_text8_best.pt" ]; then
    echo "Downloading D3PM checkpoint..."
    git clone https://huggingface.co/rupspace/nanoDD-D3PM-text8 temp_d3pm
    mv temp_d3pm/ckpt.pt d3pm/nanodd_text8_best.pt
    rm -rf temp_d3pm
fi

cd "$PROJECT_ROOT"

echo "=== [6/6] Preparing text8 dataset ==="

if [ -f "src/nanoDD/data/prepare_text8.py" ]; then
    python src/nanoDD/data/prepare_text8.py
fi

if [ -d "src/nanoDD/data/text8" ] && [ -z "$(ls -A data/text8 2>/dev/null)" ]; then
    cp -r src/nanoDD/data/text8/* data/text8/
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "Environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Repos:"
echo "  ./src/"
echo ""
echo "Checkpoints:"
echo "  ./checkpoints/"
echo ""
echo "Dataset:"
echo "  ./data/text8/"
