#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-dd}"
KERNEL_NAME="${KERNEL_NAME:-dd}"

if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda。请先安装 Miniconda 或 Anaconda。"
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda 环境 '$ENV_NAME' 已存在，正在根据 environment.yml 更新。"
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  conda env create -n "$ENV_NAME" -f environment.yml
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

python -m ipykernel install \
  --user \
  --name "$KERNEL_NAME" \
  --display-name "Python ($KERNEL_NAME)"

echo "环境配置完成。"
echo "激活环境: conda activate $ENV_NAME"
echo "启动 Jupyter: jupyter notebook"
