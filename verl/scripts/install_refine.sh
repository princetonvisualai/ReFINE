#!/bin/bash

set -e
export MAX_JOBS=32

echo "=== Stage 1: Install PyTorch and pre-built wheels ==="
echo "After this script completes, run: pip install -r requirements.txt"
echo ""

# Step 1: Install PyTorch (must be installed before packages in requirements.txt)
echo "1. Installing PyTorch..."
pip install --no-cache-dir "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

# Step 2: Install FlashAttention and FlashInfer from pre-built wheels
echo "2. Installing FlashAttention and FlashInfer..."
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl

echo ""
echo "=== Stage 1 complete ==="
echo "Now run: pip install -r requirements.txt"
