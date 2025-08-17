#!/usr/bin/env bash
set -euo pipefail

# === 可自定义变量 ===
PY_VER="3.10"
ENV_NAME="diffusion"
REPO_URL="https://github.com/yliu-fort/DDPM-minimal.git"
REPO_DIR="$HOME/DDPM-minimal"

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
  source $HOME/.bashrc
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
fi

# -------------------------------------
# 2) 初始化 conda（保证 activate 可用）
# -------------------------------------
# 有些非交互 shell 下直接 activate 会报 "Run 'conda init' before 'conda activate'"
# 这里做两步： (a) conda init bash (b) 在当前 shell 注入 hook
echo "[*] Initializing conda for bash..."
"$HOME/miniconda/bin/conda" init bash || true
# 让当前脚本会话获得 conda 的函数（而不必重新登录）
# 方式A：使用 conda 官方 hook
eval "$("$HOME/miniconda/bin/conda" shell.bash hook)" || {
  # 方式B：fallback 到 profile.d（某些版本可用）
  # shellcheck disable=SC1091
  source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || true
}

# -------------------------------------
# 3) 创建/重建环境
# -------------------------------------
# 若你想保留已有环境，把下一行删除即可
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true

echo "[*] Creating conda env ($ENV_NAME, python=$PY_VER)..."
conda create -n "$ENV_NAME" "python=${PY_VER}" -y

# --- 路径1：标准方式（activate） ---
if conda activate "$ENV_NAME" 2>/dev/null; then
  echo "[*] Activated env via 'conda activate $ENV_NAME'."

  echo "[*] Installing PyTorch (CUDA 12.1 wheels) ..."
  pip install --upgrade pip
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

  echo "[*] Cloning repo..."
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR" || { echo "❌ 进入目录失败: $REPO_DIR"; exit 1; }

  echo "[*] Installing project requirements..."
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt
  fi

  echo "[*] Running unittests..."
  python -m unittest discover -s ./tests -v || true

else
  # --- 路径2：备选方式（不依赖 activate，使用 conda run） ---
  echo "[!] 'conda activate' not available in this shell; proceeding with 'conda run' path."

  echo "[*] Installing PyTorch (CUDA 12.1 wheels) ..."
  "$HOME/miniconda/bin/conda" run -n "$ENV_NAME" python -m pip install --upgrade pip
  "$HOME/miniconda/bin/conda" run -n "$ENV_NAME" python -m pip install \
      --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

  echo "[*] Cloning repo..."
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR" || { echo "❌ 进入目录失败: $REPO_DIR"; exit 1; }

  echo "[*] Installing project requirements..."
  if [[ -f requirements.txt ]]; then
    "$HOME/miniconda/bin/conda" run -n "$ENV_NAME" python -m pip install -r requirements.txt
  fi

  echo "[*] Running unittests..."
  "$HOME/miniconda/bin/conda" run -n "$ENV_NAME" python -m unittest discover -s ./tests -v || true
fi

echo
echo "[✓] Init done."
echo "To start working:"
echo "  source \$HOME/miniconda/bin/activate $ENV_NAME"
echo "  PYTHONPATH="$PWD/src" python src/diffusion_sandbox/train.py --config configs/cifar10_uncond.yaml"
echo "  tensorboard --logdir runs"