# 便携式部署方案

本文档说明如何创建完全便携的Python环境，让整个项目可以拷贝到U盘在任何电脑运行。

## 方案概述

使用 **Python Embedded** (嵌入式版本) + 虚拟环境，将所有依赖安装到项目目录下。

## 目录结构

```
系统实践小程序/
├── python_env/           # 便携式Python环境(将创建)
│   ├── python.exe
│   ├── Lib/
│   ├── Scripts/
│   └── ...
├── run_gui.bat          # 一键启动脚本
├── setup_portable.bat   # 环境配置脚本
├── predictor.py
├── detection_gui.py
├── DriverFatigueDetection/
├── drowsiness_detection/
└── Introduction_to_AI-Project_chenbh/
```

## 部署步骤

### 第一步：下载Python嵌入式版本

1. 访问: https://www.python.org/downloads/windows/
2. 下载: **Windows embeddable package (64-bit)** 
   - 推荐版本: Python 3.10.x
   - 文件名如: `python-3.10.11-embed-amd64.zip`

3. 解压到项目目录:
```bash
解压 python-3.10.11-embed-amd64.zip
重命名文件夹为: python_env
移动到: E:\Disk-G\大二春季学期\人工智能导论\AI系统实践\系统实践小程序\python_env
```

### 第二步：配置pip

嵌入式Python默认不包含pip，需要手动安装:

```bash
# 下载 get-pip.py
# 访问: https://bootstrap.pypa.io/get-pip.py
# 保存到项目根目录

# 安装pip
python_env\python.exe get-pip.py
```

### 第三步：解除路径限制

编辑 `python_env\python310._pth` (版本号可能不同):
```
python310.zip
.
Lib
Lib\site-packages

# 取消下面这行的注释:
import site
```

### 第四步：安装所有依赖

```bash
python_env\python.exe -m pip install -r requirements_gui.txt
```

### 第五步：测试运行

```bash
run_gui.bat
```

## 便携式部署的优点

✅ **完全独立**: 不依赖系统Python
✅ **即插即用**: U盘拷贝后直接运行
✅ **环境隔离**: 不影响其他Python程序
✅ **版本锁定**: 确保依赖版本一致

## 便携式部署的注意事项

⚠️ **体积较大**: 完整环境约 2-3 GB
⚠️ **首次配置**: 需要联网下载依赖
⚠️ **路径限制**: 避免中文路径(如有问题)
⚠️ **权限要求**: 某些电脑可能需要管理员权限

## 打包分发

配置完成后，整个目录可以:
1. 压缩为zip
2. 拷贝到U盘
3. 在其他电脑解压
4. 双击 `run_gui.bat` 运行

## 轻量化方案(可选)

如果体积太大，可以考虑:
1. 使用 `requirements_minimal.txt` (仅必需包)
2. 删除不用的模型
3. 使用 PyInstaller 打包为单个exe (详见 SUMMARY.md)

---

**推荐**: 使用便携式Python方案，完整、可靠、易于维护！
