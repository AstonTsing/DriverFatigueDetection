# Windows GPU 训练说明（RTX 4060）

## 为什么当前环境看不到 GPU？

你的显卡 **NVIDIA RTX 4060** 和驱动（592.27）正常，但 conda 环境 `drowsiness` 里是 **TensorFlow 2.21**。

从 **TensorFlow 2.11 起，官方不再在 Windows 原生系统上支持 CUDA GPU**，因此会出现：

```text
TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11
GPUs: []
```

这不是显卡坏了，而是 **Windows + 新版 TensorFlow 的组合限制**。

---

## 方案 A（推荐，已配置）：DirectML 环境 `drowsiness-gpu`

在 **不装 WSL** 的前提下，用 Microsoft **DirectML** 插件让 TensorFlow 2.10 走 GPU（你的 RTX 4060 已验证可识别）。

### 1. 环境已创建时

在 Anaconda Prompt 或 PowerShell：

```powershell
conda activate drowsiness-gpu
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

应看到 `2.10.0` 且至少 1 个 `GPU` 设备。

### 2. Jupyter 切换内核

打开 `Drowsiness_Detection.ipynb` → **Kernel / 选择内核** → 选：

**`Python (drowsiness-gpu GPU)`**

然后 **Restart Kernel**，从头运行所有单元格。

### 3. 笔记本显卡模式与 `DML_VISIBLE_DEVICES`

| 模式 | DirectML 可见设备 | 应设置 |
|------|-------------------|--------|
| **混合模式**（核显 + 独显） | 通常 2 个，0=核显、1=独显 | `DML_VISIBLE_DEVICES=1` |
| **独显直连**（仅 RTX 4060） | 通常 **只有 1 个**，索引为 **0** | **不要设为 `1`**，用 `0` 或不设置 |

若切换 BIOS/OMEN 为「独显直连」后，第一个单元格显示 `GPU 设备: 无`，原因是旧代码写死了 `DML_VISIBLE_DEVICES='1'`，在只剩 1 块 GPU 时索引 1 不存在。请在 import tensorflow **之前**改为：

```python
os.environ['DML_VISIBLE_DEVICES'] = '0'   # 独显直连
# 或删除/注释该行，让 DirectML 自动选唯一 GPU
```

混合模式下若 GPU 利用率为 0，再改回 `'1'`。

### 4. 重新创建环境（可选）

```powershell
cd "C:\Users\Administrator\Desktop\Drowsiness-Detection-in-Drivers-using-Deep-Learning-master"
conda env create -f environment-gpu.yml
conda activate drowsiness-gpu
python -m ipykernel install --user --name drowsiness-gpu --display-name "Python (drowsiness-gpu GPU)"
```

---

## 方案 B（长期最佳）：WSL2 + CUDA TensorFlow

适合需要 **TensorFlow 2.21+** 和 **完整 CUDA** 性能的场景。

1. **以管理员身份**打开 PowerShell，执行：

   ```powershell
   wsl --install -d Ubuntu
   ```

   按提示重启电脑，完成 Ubuntu 初始化（设置用户名/密码）。

2. 在 **Ubuntu (WSL)** 终端：

   ```bash
   cd /mnt/c/Users/Administrator/Desktop/Drowsiness-Detection-in-Drivers-using-Deep-Learning-master
   bash scripts/setup_wsl_gpu.sh
   ```

3. 在 WSL 里用 Jupyter 或 VS Code「连接到 WSL」打开同一项目笔记本。

WSL 内 **不要** 安装 Linux 版 NVIDIA 驱动，只用 Windows 已装的驱动即可。

---

## 训练速度参考（本机实测，VGG16 冻结骨干 + 分类头）

| 环境 | 纯计算 `train_on_batch`（batch=32） | 说明 |
|------|-------------------------------------|------|
| `drowsiness` CPU（TF 2.21 + oneDNN） | **~1.8 s/step** | Windows 上当前**更快** |
| `drowsiness-gpu` DirectML（TF 2.10） | **~3.6 s/step** | 有 GPU 但**更慢** |
| WSL2 + `tensorflow[and-cuda]` | 通常 **< 1 s/step** | 真 CUDA，应优先采用 |

**结论**：在 Windows 原生环境下，DirectML **不等于** NVIDIA CUDA。对本项目的 VGG 迁移学习，继续用 `drowsiness`（CPU）往往比 `drowsiness-gpu` 更省时；要真正吃满 RTX 4060，请用 **WSL2 + CUDA**（方案 B）。

训练时在任务管理器查看 **GPU 利用率**；若仍为 0%，检查是否选对了 Jupyter 内核。

---

## 常见问题

**Q: 能否在 `drowsiness` 里 pip install tensorflow-gpu？**  
A: 2.11+ 在 Windows 上无效，请换 `drowsiness-gpu` 或 WSL。

**Q: DirectML 和 CUDA 哪个好？**  
A: CUDA（WSL）通常更快、更新；DirectML 是 Windows 上不装 WSL 时的实用折中。

**Q: 保存的模型格式？**  
A: GPU 环境（TF 2.10）请用 `.h5`；CPU 环境（TF 2.21）可用 `.keras`。
