#!/usr/bin/env bash
# ============================================================
# Part 2 Setup Script
# Clone external repositories and download pretrained weights.
# Usage:  bash setup.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTERNAL_DIR="$SCRIPT_DIR/external"
mkdir -p "$EXTERNAL_DIR"

echo "============================================"
echo " 1. Cloning external repositories"
echo "============================================"

# --- VGGT4D ---
if [ ! -d "$EXTERNAL_DIR/VGGT4D" ]; then
    echo "[+] Cloning VGGT4D ..."
    git clone https://github.com/3DAgentWorld/VGGT4D.git "$EXTERNAL_DIR/VGGT4D"
else
    echo "[=] VGGT4D already exists, skipping."
fi

# --- SAM 2 ---
if [ ! -d "$EXTERNAL_DIR/sam2" ]; then
    echo "[+] Cloning SAM 2 ..."
    git clone https://github.com/facebookresearch/sam2.git "$EXTERNAL_DIR/sam2"
else
    echo "[=] sam2 already exists, skipping."
fi

# --- ProPainter ---
if [ ! -d "$EXTERNAL_DIR/ProPainter" ]; then
    echo "[+] Cloning ProPainter ..."
    git clone https://github.com/sczhou/ProPainter.git "$EXTERNAL_DIR/ProPainter"
else
    echo "[=] ProPainter already exists, skipping."
fi

echo ""
echo "============================================"
echo " 2. Installing SAM 2 as editable package"
echo "============================================"
cd "$EXTERNAL_DIR/sam2"
pip install -e . 2>&1 | tail -5
cd "$SCRIPT_DIR"

echo ""
echo "============================================"
echo " 3. Installing ProPainter dependencies"
echo "============================================"
pip install -r "$EXTERNAL_DIR/ProPainter/requirements.txt" 2>&1 | tail -5

echo ""
echo "============================================"
echo " 4. Installing VGGT4D dependencies"
echo "============================================"
pip install -r "$EXTERNAL_DIR/VGGT4D/requirements.txt" 2>&1 | tail -5

echo ""
echo "============================================"
echo " 5. Downloading pretrained model weights"
echo "============================================"

# --- VGGT4D checkpoint ---
VGGT4D_CKPT="$EXTERNAL_DIR/VGGT4D/ckpts/model_tracker_fixed_e20.pt"
if [ ! -f "$VGGT4D_CKPT" ]; then
    echo "[+] Downloading VGGT4D checkpoint ..."
    mkdir -p "$EXTERNAL_DIR/VGGT4D/ckpts"
    wget -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt?download=true" \
         -O "$VGGT4D_CKPT"
else
    echo "[=] VGGT4D checkpoint exists."
fi

# --- SAM 2.1 Large checkpoint ---
SAM2_CKPT="$EXTERNAL_DIR/sam2/checkpoints/sam2.1_hiera_large.pt"
if [ ! -f "$SAM2_CKPT" ]; then
    echo "[+] Downloading SAM 2.1 Hiera Large checkpoint ..."
    mkdir -p "$EXTERNAL_DIR/sam2/checkpoints"
    wget -c "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
         -O "$SAM2_CKPT"
else
    echo "[=] SAM 2.1 checkpoint exists."
fi

# --- ProPainter weights (auto-download fallback, but explicit is better) ---
PP_WEIGHTS_DIR="$EXTERNAL_DIR/ProPainter/weights"
mkdir -p "$PP_WEIGHTS_DIR"

PP_WEIGHT="$PP_WEIGHTS_DIR/ProPainter.pth"
if [ ! -f "$PP_WEIGHT" ]; then
    echo "[+] Downloading ProPainter.pth ..."
    wget -c "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth" \
         -O "$PP_WEIGHT"
else
    echo "[=] ProPainter.pth exists."
fi

RFC_WEIGHT="$PP_WEIGHTS_DIR/recurrent_flow_completion.pth"
if [ ! -f "$RFC_WEIGHT" ]; then
    echo "[+] Downloading recurrent_flow_completion.pth ..."
    wget -c "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth" \
         -O "$RFC_WEIGHT"
else
    echo "[=] recurrent_flow_completion.pth exists."
fi

RAFT_WEIGHT="$PP_WEIGHTS_DIR/raft-things.pth"
if [ ! -f "$RAFT_WEIGHT" ]; then
    echo "[+] Downloading raft-things.pth ..."
    wget -c "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth" \
         -O "$RAFT_WEIGHT"
else
    echo "[=] raft-things.pth exists."
fi

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo "External repos: $EXTERNAL_DIR"
echo "  VGGT4D  checkpoint: $VGGT4D_CKPT"
echo "  SAM 2.1 checkpoint: $SAM2_CKPT"
echo "  ProPainter weights: $PP_WEIGHTS_DIR"
