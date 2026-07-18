# 驾驶员疲劳检测项目合集

本目录汇总了 3 个驾驶员疲劳/困倦检测相关项目，均围绕二分类任务展开：判断驾驶员图像或视频帧属于 `drowsy`（疲劳/困倦）还是 `notdrowsy`（清醒/非疲劳）。

## 🎯 快速开始 - 图形界面应用

**当前状态：已完成。** 根目录已生成 [detection.exe](detection.exe)，便携式 Python 环境 [python_env/](python_env/) 已安装依赖，5 种检测方法均已识别并全部验证通过。已修复 Qt platform plugin 启动错误、MediaPipe DLL 初始化错误、TensorFlow 中文路径读图问题，以及 MobileNetV2 旧版 Keras 模型兼容问题。GUI 打开后会启动常驻 [prediction_server.py](prediction_server.py) 预测服务并后台预加载 5 种模型，后续单模型检测和五模型对比检测会复用已加载模型；YOLO 会在 PyTorch 可用 CUDA 时自动使用 GPU。你现在可以直接双击 [detection.exe](detection.exe) 打开可视化界面并检测图片。

### 直接使用

1. 双击根目录下的 [detection.exe](detection.exe)。
2. 在左侧选择检测功能：
   - **单模型检测**：选择 1 种检测方法，对单张图片进行检测。
   - **五模型对比检测**：对同一张图片依次调用现有 5 种检测方法，并在右侧汇总对比结果。
3. 若使用“单模型检测”，再选择具体检测方法。
4. 点击“选择图片”，默认可从 [DriverFatigueDetection/dataset_split/test/](DriverFatigueDetection/dataset_split/test/) 选择测试图片。
5. 点击“开始单模型检测”或“开始五模型对比检测”，右侧会显示图片、检测结果、置信度/概率、关键特征、对比表和完整 JSON。

### 便携式拷贝

整个 [系统实践小程序/](./) 文件夹可以直接复制到 U 盘或移动硬盘。到另一台 Windows 电脑后，保持目录结构不变，双击 [detection.exe](detection.exe) 即可运行；无需另行安装 Python。

> 说明：[detection.exe](detection.exe) 是轻量启动器，会调用本目录下的 [python_env/pythonw.exe](python_env/pythonw.exe) 启动 [detection_gui.py](detection_gui.py)。因此复制时必须保留整个文件夹，而不是只复制单个 exe。

### 命令行备用启动

如果双击 exe 被安全软件拦截，也可以双击 [run_gui.bat](run_gui.bat)，或在终端运行：

```bat
python_env\python.exe detection_gui.py
```

### 📚 完整文档

- **[FINAL_STATUS.md](FINAL_STATUS.md)** ⭐ - 当前最终可运行状态
- **[PORTABLE_QUICKSTART.md](PORTABLE_QUICKSTART.md)** ⭐ - 便携式部署快速指南
- **[PORTABLE_REPORT.md](PORTABLE_REPORT.md)** ⭐ - 便携式部署完成报告
- **[SUMMARY.md](SUMMARY.md)** ⭐ - 项目完成总结和使用指南
- **[USAGE.md](USAGE.md)** - 详细的安装使用文档
- **[PORTABLE.md](PORTABLE.md)** - 便携式部署原理说明
- **[README.md](README.md)** - 本文档(项目总览)

### 功能特性

1. **检测功能切换**:
   - 单模型检测：选择 1 种方法检测一张图片。
   - 五模型对比检测：同一张图片同时用 5 种方法检测，并输出横向对比表和多数投票结论。

2. **多方法支持**: 支持 5 种检测方法
   - DriverFatigueDetection-v0 (SVM) - 准确率~68.7%
   - DriverFatigueDetection-v1 (HistGradientBoosting) - 准确率~92.4%
   - YOLOv11 Classification - 端到端深度学习
   - TensorFlow Baseline CNN - 准确率~91.2%
   - TensorFlow MobileNetV2 - 准确率~96.6%

3. **图片选择**: 从数据集选择测试图片

4. **结果可视化**:
   - 图片预览
   - 单模型检测结果(疲劳/清醒)
   - 五模型横向对比表
   - 多数投票结论
   - 置信度/概率分布
   - 提取的特征
   - 完整 JSON 结果

5. **异步检测**: 后台线程,界面流畅

### 新增文件

**便携式部署相关** ⭐:
- [detection.exe](detection.exe) - 双击启动图形界面的根目录入口
- [python_env/](python_env/) - 已安装依赖的便携式 Python 环境
- [setup_portable.bat](setup_portable.bat) - 便携式环境自动配置脚本
- [run_gui.bat](run_gui.bat) - 便携式启动脚本
- [PORTABLE_QUICKSTART.md](PORTABLE_QUICKSTART.md) - 便携式部署快速指南
- [PORTABLE.md](PORTABLE.md) - 便携式部署详细说明
- [requirements_minimal.txt](requirements_minimal.txt) - 最小依赖列表(体积优化)

**GUI应用核心文件**:
- [detection_gui.py](detection_gui.py) - PyQt5 图形界面主程序
- [predictor.py](predictor.py) - 统一的预测接口模块
- [requirements_gui.txt](requirements_gui.txt) - 完整依赖列表

**其他辅助文件**:
- [build_exe.bat](build_exe.bat) - PyInstaller打包脚本
- [fix_indent.bat](fix_indent.bat) - 缩进修复辅助脚本
- [USAGE.md](USAGE.md) - 详细使用指南
- [SUMMARY.md](SUMMARY.md) - 项目完成总结
- [Introduction_to_AI-Project_chenbh/train_mobilenet.py](Introduction_to_AI-Project_chenbh/train_mobilenet.py) - MobileNetV2 训练脚本

---

三个项目的技术路线并不相同：

| 项目 | 主要路线 | 核心模型/方法 | 主要入口 | 适合用途 |
|---|---|---|---|---|
| [DriverFatigueDetection](DriverFatigueDetection/) | MediaPipe 人脸关键点 + 传统机器学习 | EAR/MAR 规则、SVM、HistGradientBoosting | `split_dataset.py`、v0/v1 的 `train.py`、`evaluate.py`、`predict.py` | 可解释传统 ML baseline、特征工程对比 |
| [drowsiness_detection](drowsiness_detection/) | YOLOv11 图像分类 | `YOLO("yolo11n-cls.pt")` / `best.pt` | `train.py`、`run_train.sh`、notebook | 端到端深度学习分类训练、TFLite 导出 |
| [Introduction_to_AI-Project_chenbh](Introduction_to_AI-Project_chenbh/) | TensorFlow/Keras CNN 与迁移学习 | Baseline CNN、MobileNetV2；原 README 还说明 VGG16 思路 | `Drowsiness_Detection.ipynb` | 课程 notebook 实验、CNN 与迁移学习对比 |

> 注意：这些项目主要用于课程实践、模型对比和实验复现，不是车载真实安全系统的完整实现。真实驾驶安全系统还需要视频时序判断、连续帧平滑、低光/遮挡鲁棒性、实时性、报警策略和严格测试验证。

## 目录结构

```text
系统实践小程序/
├── README.md
├── DriverFatigueDetection/
│   ├── README.md
│   ├── split_dataset.py
│   ├── dataset_split/
│   ├── DriverFatigueDetection-v0/
│   └── DriverFatigueDetection-v1/
├── drowsiness_detection/
│   ├── README.md
│   ├── train.py
│   ├── run_train.sh
│   ├── plot_curves.py
│   ├── environment.yml
│   ├── Deep_Learning_for_Safer_Roads_YOLOv11_Driver_Drowsiness_Detection.ipynb
│   └── outputs/
└── Introduction_to_AI-Project_chenbh/
    ├── README.md
    ├── Drowsiness_Detection.ipynb
    ├── GPU_SETUP_CN.md
    ├── environment-gpu.yml
    ├── model_baseline.h5
    ├── scripts/
    └── Model & Training Results/
```

## 统一任务说明

三个项目都处理驾驶员疲劳检测，但输入、训练方式和实现重点不同。

- **输入形式**：本质上都使用从视频或数据集中获得的驾驶员面部图像/帧。
- **输出类别**：通常为 `drowsy` 和 `notdrowsy` 两类。
- **核心差异**：
  - [DriverFatigueDetection](DriverFatigueDetection/) 更强调人脸关键点、几何特征和传统机器学习的可解释性。
  - [drowsiness_detection](drowsiness_detection/) 使用 YOLOv11 classification，把问题作为图像分类任务处理。
  - [Introduction_to_AI-Project_chenbh](Introduction_to_AI-Project_chenbh/) 使用 TensorFlow/Keras notebook，重点展示 CNN、迁移学习和训练评估流程。

## 1. DriverFatigueDetection

路径：[DriverFatigueDetection](DriverFatigueDetection/)

该项目是一个传统机器学习路线的驾驶员疲劳检测仓库，包含一个数据集划分脚本和两个版本：

- [DriverFatigueDetection-v0](DriverFatigueDetection/DriverFatigueDetection-v0/)：基线版本。
- [DriverFatigueDetection-v1](DriverFatigueDetection/DriverFatigueDetection-v1/)：改进版本。

### 技术路线

项目使用 **MediaPipe Face Mesh** 检测人脸关键点，再从关键点中提取人工特征，最后用规则或传统机器学习分类器判断是否疲劳。

该项目不训练 CNN、RNN、Transformer 等深度学习模型。MediaPipe 在这里只用于人脸关键点提取，最终分类器是传统机器学习方法。

### v0：EAR/MAR + 规则或 SVM

v0 主要提取 4 个特征：

- `ear_left`：左眼 Eye Aspect Ratio。
- `ear_right`：右眼 Eye Aspect Ratio。
- `ear_mean`：左右眼 EAR 平均值。
- `mar`：Mouth Aspect Ratio。

支持两种方式：

1. **规则阈值法**：通过 [rule_config.json](DriverFatigueDetection/DriverFatigueDetection-v0/config/rule_config.json) 中的 EAR/MAR 阈值直接判断。
2. **SVM 分类器**：使用 `StandardScaler + SVC` 训练二分类模型。

主要文件：

- [DriverFatigueDetection-v0/train.py](DriverFatigueDetection/DriverFatigueDetection-v0/train.py)：训练 SVM。
- [DriverFatigueDetection-v0/evaluate.py](DriverFatigueDetection/DriverFatigueDetection-v0/evaluate.py)：评估规则或 SVM。
- [DriverFatigueDetection-v0/predict.py](DriverFatigueDetection/DriverFatigueDetection-v0/predict.py)：单张图片预测。
- [DriverFatigueDetection-v0/src/landmark_features.py](DriverFatigueDetection/DriverFatigueDetection-v0/src/landmark_features.py)：MediaPipe 关键点与 EAR/MAR 特征提取。
- [DriverFatigueDetection-v0/src/rule_detector.py](DriverFatigueDetection/DriverFatigueDetection-v0/src/rule_detector.py)：规则阈值分类器。

### v1：丰富人工特征 + HistGradientBoosting

v1 相比 v0 提取了更多特征，覆盖：

- 眼睛开合程度、左右眼差异。
- 嘴部开合程度。
- 人脸宽高比例。
- 鼻尖、下巴、额头等头部姿态代理特征。
- 眉眼距离。
- 图像亮度、对比度、模糊度。

分类器使用：

```text
SimpleImputer(strategy="median") + HistGradientBoostingClassifier
```

v1 还增加了特征缓存机制，避免每次训练/评估都重复运行耗时的 MediaPipe 特征提取。

主要文件：

- [DriverFatigueDetection-v1/train.py](DriverFatigueDetection/DriverFatigueDetection-v1/train.py)：训练 HistGradientBoosting 模型。
- [DriverFatigueDetection-v1/evaluate.py](DriverFatigueDetection/DriverFatigueDetection-v1/evaluate.py)：测试集评估。
- [DriverFatigueDetection-v1/predict.py](DriverFatigueDetection/DriverFatigueDetection-v1/predict.py)：单张图片预测。
- [DriverFatigueDetection-v1/src/feature_extractor.py](DriverFatigueDetection/DriverFatigueDetection-v1/src/feature_extractor.py)：v1 特征提取。
- [DriverFatigueDetection-v1/src/feature_cache.py](DriverFatigueDetection/DriverFatigueDetection-v1/src/feature_cache.py)：特征缓存。
- [DriverFatigueDetection-v1/config/model_config.json](DriverFatigueDetection/DriverFatigueDetection-v1/config/model_config.json)：模型与 MediaPipe 参数配置。

### 数据集结构

默认使用 [DriverFatigueDetection/dataset_split](DriverFatigueDetection/dataset_split/)：

```text
dataset_split/
├── train/
│   ├── drowsy/
│   └── notdrowsy/
├── val/
│   ├── drowsy/
│   └── notdrowsy/
└── test/
    ├── drowsy/
    └── notdrowsy/
```

如需从原始数据重新划分，可在 [DriverFatigueDetection](DriverFatigueDetection/) 下运行：

```bat
python split_dataset.py --source train --output dataset_split --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

如果确认要覆盖已有划分：

```bat
python split_dataset.py --source train --output dataset_split --overwrite
```

### 环境与运行

推荐使用 Conda：

```bat
conda create -n driver python=3.12
conda activate driver
```

安装依赖：

```bat
cd DriverFatigueDetection\DriverFatigueDetection-v0
pip install -r requirements.txt

cd ..\DriverFatigueDetection-v1
pip install -r requirements.txt
```

v0 示例：

```bat
cd DriverFatigueDetection\DriverFatigueDetection-v0
python evaluate.py --test-dir ../dataset_split/test --output reports/rule_eval_metrics.json
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/svm_ear_mar.joblib --metrics reports/train_metrics.json
python evaluate.py --test-dir ../dataset_split/test --model models/svm_ear_mar.joblib --output reports/svm_eval_metrics.json
```

v1 示例：

```bat
cd DriverFatigueDetection\DriverFatigueDetection-v1
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/hgb_fatigue.joblib --metrics reports/train_metrics.json
python evaluate.py --test-dir ../dataset_split/test --model models/hgb_fatigue.joblib --output reports/test_metrics.json
```

### 已记录评估结果

该仓库 README 中记录了当前测试集结果：测试集共 9,980 张图片，其中 9,888 张检测到人脸并参与有效评估。

| 版本 | 方法 | 准确率 | 说明 |
|---|---|---:|---|
| v0 | 规则阈值 | 59.74% | 基于 EAR/MAR 阈值 |
| v0 | SVM | 68.72% | 使用 4 个 EAR/MAR 特征 |
| v1 | HistGradientBoosting | 92.39% | 使用 35 个手工特征 |

## 2. drowsiness_detection

路径：[drowsiness_detection](drowsiness_detection/)

该项目使用 **YOLOv11 classification** 完成驾驶员困倦二分类。虽然项目名包含 detection，但当前核心流程是图像分类，而不是目标检测框定位。

### 技术路线

- 类别：`drowsy`、`notdrowsy`。
- 初始模型：`yolo11n-cls.pt`。
- 训练入口：命令行脚本 [train.py](drowsiness_detection/train.py)。
- 启动脚本：[run_train.sh](drowsiness_detection/run_train.sh)。
- 交互式流程：[Deep_Learning_for_Safer_Roads_YOLOv11_Driver_Drowsiness_Detection.ipynb](drowsiness_detection/Deep_Learning_for_Safer_Roads_YOLOv11_Driver_Drowsiness_Detection.ipynb)。
- 训练曲线工具：[plot_curves.py](drowsiness_detection/plot_curves.py)。

### train.py 行为

[train.py](drowsiness_detection/train.py) 的主要流程：

1. 检查数据集目录。
2. 检查训练集和验证集类别是否一致。
3. 加载 YOLOv11 分类模型。
4. 关闭 Ultralytics 内置完整验证，改用抽样验证。
5. 每隔 `eval-freq` 个 epoch 在验证集抽样图片上评估成功率。
6. 当抽样成功率提升时，将 checkpoint 复制到 `outputs/<task_name>/best.pt`。
7. 将最佳 epoch、最佳成功率和历史记录写入 `outputs/<task_name>/metrics.json`。
8. 训练结束后清理中间 `.pt` 文件，只保留主要 best checkpoint。

### 数据集要求

默认数据集路径在 Linux/WSL 风格目录：

```text
/home/hebu/dd_dataset
```

需要具备以下结构：

```text
dd_dataset/
├── train/
│   ├── drowsy/
│   └── notdrowsy/
└── val/ 或 valid/
    ├── drowsy/
    └── notdrowsy/
```

数据集本身不包含在仓库中，需要本地准备。

### 环境

环境文件：[environment.yml](drowsiness_detection/environment.yml)

核心依赖包括：

- Python 3.10
- PyTorch `2.6.0+cu124`
- TorchVision `0.21.0+cu124`
- Ultralytics `8.3.40`
- TensorFlow `2.15.1`
- OpenCV、Pillow、Matplotlib、Notebook 等

创建环境：

```bash
cd drowsiness_detection
bash setup_env.sh
```

或直接使用 conda 环境文件：

```bash
conda env create -f environment.yml
conda activate dd
```

### 运行训练

默认运行：

```bash
cd drowsiness_detection
bash run_train.sh
```

覆盖参数示例：

```bash
TASK_NAME=exp_50ep EPOCHS=50 BATCH_SIZE=16 EVAL_FREQ=2 NUM_EVAL=200 DEVICE=0 bash run_train.sh
```

默认关键参数来自 [run_train.sh](drowsiness_detection/run_train.sh)：

```text
TASK_NAME=test1
DATA_DIR=/home/hebu/dd_dataset
MODEL=yolo11n-cls.pt
EPOCHS=20
IMG_SIZE=224
BATCH_SIZE=16
EVAL_FREQ=2
NUM_EVAL=100
DEVICE=0
```

### 输出

训练输出位于：

```text
drowsiness_detection/outputs/<task_name>/
```

典型文件：

- `best.pt`：抽样验证表现最好的 checkpoint。
- `metrics.json`：最佳 epoch、成功率和历史记录。
- `train/results.csv`：Ultralytics 训练日志。
- `curves.png`、`lr_curve.png`：由 [plot_curves.py](drowsiness_detection/plot_curves.py) 生成的曲线。

仓库中已存在一个示例输出目录 [outputs/test1](drowsiness_detection/outputs/test1/)，可作为训练结果参考。

### 注意事项

- 该项目是 YOLOv11 分类流程，不是传统 bounding-box 目标检测流程。
- 默认路径 `/home/hebu/dd_dataset` 更适合 Linux/WSL；Windows 原生运行时需要改 `DATA_DIR`。
- 环境中 PyTorch CUDA wheel 固定到 `cu124`，不同 CUDA/CPU 环境可能需要调整。
- README 中列出的 TensorFlow 版本与 [environment.yml](drowsiness_detection/environment.yml) 中版本存在差异，应以实际环境文件为准。

## 3. Introduction_to_AI-Project_chenbh

路径：[Introduction_to_AI-Project_chenbh](Introduction_to_AI-Project_chenbh/)

该项目是一个基于 TensorFlow/Keras 的驾驶员困倦检测 notebook 项目，重点展示 CNN 与迁移学习模型在二分类任务上的训练和评估。

### 原项目说明

该项目 README 描述的任务是：从视频数据中抽取帧，使用面部特征训练 CNN 模型，检测驾驶员疲劳状态。

原 README 提到的数据集为：

- NTHU Drivers' Drowsiness Detection Dataset

原 README 还介绍了两个模型：

1. **Baseline Model**：从零训练的标准 CNN。
2. **Final Model**：基于 VGG16 ImageNet 预训练层的迁移学习模型，冻结前部卷积层，再训练后续分类层。

原 README 中记录的结果：

| 模型 | Accuracy |
|---|---:|
| Baseline Model | 68.6% |
| Final Model | 73.2% |

### 当前 notebook 内容

当前 [Drowsiness_Detection.ipynb](Introduction_to_AI-Project_chenbh/Drowsiness_Detection.ipynb) 已经调整为中文说明，并使用 `dataset_split` 结构进行训练与评估。

notebook 中的主要流程：

1. 设置路径和超参数。
2. 从 `dataset_split/train`、`dataset_split/val`、`dataset_split/test` 加载图片。
3. 使用 `ImageDataGenerator` 做 rescale 和数据增强。
4. 计算类别权重，缓解 `drowsy` 与 `notdrowsy` 类别不平衡。
5. 训练 Baseline CNN。
6. 训练 MobileNetV2 迁移学习模型。
7. 绘制训练/验证准确率和 loss 曲线。
8. 在测试集上输出分类报告和混淆矩阵。

notebook 中记录的数据规模：

| split | drowsy | notdrowsy | total |
|---|---:|---:|---:|
| train | 25,221 | 21,343 | 46,564 |
| val | 5,404 | 4,573 | 9,977 |
| test | 5,405 | 4,575 | 9,980 |

### 当前 notebook 模型

Baseline CNN：

- 3 层卷积/池化。
- Flatten。
- Dense + Dropout。
- 2 类 softmax。
- 保存为 `model_baseline.h5`。

MobileNetV2：

- 使用 ImageNet 预训练的 `MobileNetV2(include_top=False)`。
- 冻结 MobileNetV2 backbone。
- 接 Flatten、Dense(1024)、Dense(512)、Dense(2)。
- 保存为 `model_mobilenetv2.h5`。

当前 notebook 测试集上记录的 MobileNetV2 结果：

```text
accuracy: 0.9661
macro avg f1-score: 0.9659
weighted avg f1-score: 0.9661
```

混淆矩阵：

```text
[[5268  137]
 [ 201 4374]]
```

### GPU/环境说明

项目包含中文 GPU 说明：[GPU_SETUP_CN.md](Introduction_to_AI-Project_chenbh/GPU_SETUP_CN.md)

其中说明：

- TensorFlow 2.11 起，Windows 原生系统不再官方支持 CUDA GPU。
- Windows 原生可使用 DirectML 折中方案，但该项目实测 DirectML 不一定比 CPU 快。
- 若要真正使用 RTX 4060 等 NVIDIA GPU 的 CUDA 性能，推荐 WSL2 + CUDA TensorFlow。

DirectML 环境文件：[environment-gpu.yml](Introduction_to_AI-Project_chenbh/environment-gpu.yml)

```bash
conda env create -f environment-gpu.yml
conda activate drowsiness-gpu
python -m ipykernel install --user --name drowsiness-gpu --display-name "Python (drowsiness-gpu GPU)"
```

辅助脚本：

- [scripts/benchmark_train_step.py](Introduction_to_AI-Project_chenbh/scripts/benchmark_train_step.py)：对比 CPU 与 DirectML GPU 的单步训练耗时。
- [scripts/setup_wsl_gpu.sh](Introduction_to_AI-Project_chenbh/scripts/setup_wsl_gpu.sh)：WSL GPU 环境设置脚本。

### 注意事项

- 当前 notebook 使用 `dataset_split` 目录，若仓库内没有该目录，需要先准备数据。
- README 的原始说明、notebook 当前实现和 GPU 文档之间存在一定历史差异：原 README 强调 VGG16，当前 notebook 使用 MobileNetV2 迁移学习。
- `.h5` 模型文件适合 TensorFlow 2.10/DirectML 环境；较新的 TensorFlow 也可考虑 `.keras` 格式。

## 三个项目对比

| 维度 | DriverFatigueDetection | drowsiness_detection | Introduction_to_AI-Project_chenbh |
|---|---|---|---|
| 方法类型 | 传统机器学习 | YOLOv11 深度学习分类 | TensorFlow/Keras CNN 与迁移学习 |
| 是否依赖人脸关键点 | 是，MediaPipe Face Mesh | 否，直接图像分类 | 否，直接图像分类 |
| 特征来源 | EAR/MAR、头部几何、图像质量等人工特征 | YOLO 模型自动学习图像特征 | CNN/MobileNetV2 自动学习图像特征 |
| 可解释性 | 较强 | 较弱 | 较弱 |
| 训练入口 | Python 脚本 | Python 脚本 + shell + notebook | Jupyter Notebook |
| 数据路径 | `dataset_split` | 默认 `/home/hebu/dd_dataset` | `dataset_split` |
| 输出模型 | `.joblib` | `.pt`，可导出 TFLite | `.h5` |
| 适合平台 | Windows/Conda 较友好 | Linux/WSL/GPU 更自然 | Jupyter/Conda，Windows GPU 需注意 TF 限制 |

## 推荐阅读顺序

如果目的是课程汇报或理解实现差异，建议按以下顺序阅读：

1. [DriverFatigueDetection/README.md](DriverFatigueDetection/README.md)：先理解传统 ML baseline 和数据集结构。
2. [DriverFatigueDetection/DriverFatigueDetection-v0](DriverFatigueDetection/DriverFatigueDetection-v0/)：理解 EAR/MAR 与规则阈值。
3. [DriverFatigueDetection/DriverFatigueDetection-v1](DriverFatigueDetection/DriverFatigueDetection-v1/)：理解更丰富特征如何提升传统 ML 表现。
4. [drowsiness_detection/README.md](drowsiness_detection/README.md)：理解 YOLOv11 分类训练流程。
5. [drowsiness_detection/train.py](drowsiness_detection/train.py)：重点看抽样验证、最佳 checkpoint 保存逻辑。
6. [Introduction_to_AI-Project_chenbh/Drowsiness_Detection.ipynb](Introduction_to_AI-Project_chenbh/Drowsiness_Detection.ipynb)：理解 CNN/MobileNetV2 notebook 实验流程。
7. [Introduction_to_AI-Project_chenbh/GPU_SETUP_CN.md](Introduction_to_AI-Project_chenbh/GPU_SETUP_CN.md)：如果需要 GPU 训练，再看 Windows/DirectML/WSL 的环境说明。

## 复现实验时的共同注意事项

1. **数据集授权**：NTHU DDD 等数据集可能有授权要求，请按数据集许可获取和使用。
2. **数据泄漏风险**：如果同一驾驶员、同一视频片段或相邻帧同时出现在 train/val/test 中，评估准确率可能偏乐观。
3. **图片质量影响明显**：遮挡、侧脸、模糊、低光照会影响 MediaPipe 或 CNN 分类效果。
4. **Windows 路径问题**：本目录包含中文路径，运行脚本时建议使用引号包裹路径；Python 读图代码也需要注意非 ASCII 路径兼容性。
5. **GPU 环境差异**：PyTorch、TensorFlow、CUDA、DirectML、WSL2 的兼容性差异较大，复现前应先确认当前环境能否识别 GPU。
6. **输出文件区分**：`models/`、`reports/`、`cache/`、`outputs/`、`runs/` 等多为训练生成文件；修改代码时应区分源代码和实验产物。

## 总结

这三个项目可以形成一条由浅入深的对比线：

1. [DriverFatigueDetection](DriverFatigueDetection/) 展示了传统可解释方法：从人脸关键点提取人工特征，再使用规则、SVM 或梯度提升分类。
2. [drowsiness_detection](drowsiness_detection/) 展示了基于 YOLOv11 的现代深度学习图像分类训练流程，并包含 checkpoint 管理和 TFLite 导出思路。
3. [Introduction_to_AI-Project_chenbh](Introduction_to_AI-Project_chenbh/) 展示了 TensorFlow/Keras notebook 形式的 CNN 与迁移学习实验，适合课程展示和训练曲线分析。

如果用于课程报告，可以从“传统特征工程 baseline → 深度学习分类 → 迁移学习效果与环境部署”这一主线组织说明。
