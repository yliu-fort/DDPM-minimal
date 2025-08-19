#!/usr/bin/env bash
set -euo pipefail

# === Disk Management ===
#sudo mkfs -t ext4 /dev/nvme2n1
#sudo mkdir /data
#sudo mount /dev/nvme2n1 /data
#sudo chown ubuntu:ubuntu /data

# ===  ===
PY_VER="3.10"
ENV_NAME="diffusion"
REPO_URL="https://github.com/yliu-fort/DDPM-minimal.git"
REPO_DIR="/data/DDPM-minimal"

source /opt/pytorch/bin/activate

echo "[*] Updating packages..."
sudo apt-get update -y
sudo apt-get install -y git wget curl tmux htop unzip


echo "[*] Installing PyTorch (CUDA 12.1 wheels) ..."
pip install --upgrade pip
#pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo "[*] Cloning repo..."
rm -rf "$REPO_DIR"
git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR" || { echo " : $REPO_DIR"; exit 1; }

echo "[*] Installing project requirements..."
if [[ -f requirements.txt ]]; then
pip install -r requirements.txt
fi

echo "[*] Running unittests..."
python -m unittest discover -s ./tests -v || true



echo
echo "[] Init done."
echo "To start working:"
echo "cd $REPO_DIR"
echo "source /opt/pytorch/bin/activate"
echo "PYTHONPATH="$PWD/src" python src/diffusion_sandbox/train.py --config configs/cifar10_uncond.yaml"
echo "tensorboard --logdir runs"