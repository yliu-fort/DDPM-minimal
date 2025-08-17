#!/usr/bin/env bash
set -euo pipefail

# === 可自定义变量 ===
PY_VER="3.10"
ENV_NAME="diffusion"
REPO_URL="https://github.com/yliu-fort/DDPM-minimal.git"
REPO_DIR="DDPM-minimal"

echo "[*] Updating packages..."
sudo apt-get update -y
sudo apt-get install -y git wget curl tmux htop unzip

# 检查是否已有 NVIDIA 驱动（DLAMI 通常自带）
if [[ -e /proc/driver/nvidia/version ]]; then
  echo "[*] NVIDIA driver detected."
else
  echo "[!] NVIDIA driver not detected. If you are on plain Ubuntu, consider using AWS Deep Learning AMI or install drivers manually:"
  echo "    sudo apt-get install -y ubuntu-drivers-common && sudo ubuntu-drivers autoinstall && sudo reboot"
  echo "    After reboot, re-run this script."
fi

# 安装 Miniconda（若已安装则跳过）
if ! command -v conda >/dev/null 2>&1; then
  echo "[*] Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$HOME/miniconda"
  rm miniconda.sh
  echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> "$HOME/.bashrc"
  source "$HOME/.bashrc"
fi

echo "[*] Creating conda env ($ENV_NAME, python=$PY_VER)..."
conda deactivate || true
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
conda create -n "$ENV_NAME" "python=${PY_VER}" -y
source "$HOME/miniconda/bin/activate" "$ENV_NAME"

echo "[*] Installing PyTorch (CUDA 12.1 wheels) ..."
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 获取项目代码
if [[ -n "$REPO_URL" ]]; then
  echo "[*] Cloning repo: $REPO_URL"
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
elif [[ -n "$ZIP_PATH" ]]; then
  echo "[*] Unzipping project from: $ZIP_PATH"
  rm -rf "$REPO_DIR"
  mkdir -p "$REPO_DIR"
  unzip -q "$ZIP_PATH" -d "$REPO_DIR"
else
  echo "[!] No REPO_URL or ZIP_PATH provided. Put your project under $REPO_DIR and re-run 'pip install -r requirements.txt'."
  mkdir -p "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "[*] Installing project requirements..."
if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
fi

echo "[*] Running unittests..."
python -m unittest -v || true

echo
echo "[✓] Init done."
echo "To start working:"
echo "  source \$HOME/miniconda/bin/activate $ENV_NAME"
echo "  python -m diffusion_sandbox.train --config configs/cifar10_class.yaml"
