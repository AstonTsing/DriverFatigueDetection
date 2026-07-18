#!/usr/bin/env bash
# 在 WSL2 Ubuntu 中运行：安装 TensorFlow GPU（CUDA 由 pip 包自带）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> 创建虚拟环境 ~/venvs/drowsiness-wsl"
python3 -m venv ~/venvs/drowsiness-wsl
# shellcheck disable=SC1090
source ~/venvs/drowsiness-wsl/bin/activate

python -m pip install --upgrade pip
python -m pip install "tensorflow[and-cuda]" scikit-learn matplotlib ipykernel jupyter

echo "==> 注册 Jupyter 内核"
python -m ipykernel install --user --name drowsiness-wsl --display-name "Python (WSL CUDA GPU)"

echo "==> 验证 GPU"
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU'))"

echo ""
echo "完成。在 WSL 中启动 Jupyter："
echo "  source ~/venvs/drowsiness-wsl/bin/activate"
echo "  cd \"$PROJECT_DIR\""
echo "  jupyter notebook --no-browser"
