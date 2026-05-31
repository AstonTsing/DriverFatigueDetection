# DriverFatigueDetection 驾驶员疲劳检测

本仓库包含两个基于传统机器学习的驾驶员疲劳检测项目，以及一个数据集划分脚本。项目使用 MediaPipe Face Mesh 提取人脸关键点，再基于眼睛、嘴部、头部几何等人工特征进行二分类，判断驾驶员是否处于疲劳/困倦状态。

> 本项目不训练 CNN、RNN、Transformer 等深度学习模型。MediaPipe 仅用于人脸关键点提取，最终分类器均为传统机器学习方法。

## 项目结构

```text
DriverFatigueDetection/
├── DriverFatigueDetection-v0/     # 基线版本：EAR/MAR + 规则阈值或 SVM
├── DriverFatigueDetection-v1/     # 改进版本：更丰富的人工特征 + HistGradientBoosting
├── dataset_split/                 # 已划分好的训练/验证/测试数据集
├── split_dataset.py               # 数据集划分脚本
├── CLAUDE.md                      # 项目开发说明
└── README.md                      # 根目录说明文档
```

## 两个版本说明

### DriverFatigueDetection-v0

v0 是基线项目，主要使用 MediaPipe Face Mesh 提取人脸关键点，并计算：

- EAR：Eye Aspect Ratio，眼睛纵横比
- MAR：Mouth Aspect Ratio，嘴部纵横比

支持两种检测方式：

1. **规则阈值法**：根据 EAR/MAR 阈值直接判断是否疲劳。
2. **SVM 分类器**：使用 `[ear_left, ear_right, ear_mean, mar]` 四个特征训练 SVM 二分类模型。

目录：[DriverFatigueDetection-v0/](DriverFatigueDetection-v0/)

### DriverFatigueDetection-v1

v1 是改进版本，仍使用 MediaPipe Face Mesh 提取关键点，但特征更丰富，包括：

- 眼睛开合程度和左右眼差异
- 嘴部开合程度
- 人脸宽高比例
- 鼻尖、下巴、额头等头部姿态代理特征
- 眉眼距离
- 图像亮度、对比度、模糊度

分类器使用 scikit-learn 的：

```text
SimpleImputer(strategy="median") + HistGradientBoostingClassifier
```

v1 支持特征缓存，可以避免重复运行耗时的 MediaPipe 特征提取。

目录：[DriverFatigueDetection-v1/](DriverFatigueDetection-v1/)

## 数据集结构

默认数据集目录为根目录下的 [dataset_split/](dataset_split/)，结构如下：

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

类别说明：

| 类别目录 | 标签 | 含义 |
|---|---:|---|
| `notdrowsy` | 0 | 非疲劳/清醒 |
| `drowsy` | 1 | 疲劳/困倦 |

类别目录下可以继续包含子目录，程序会递归查找图片。

## 环境配置

推荐使用 Conda 环境：

```bat
conda create -n driver python=3.12
conda activate driver
```

进入对应版本目录安装依赖。

v0：

```bat
cd DriverFatigueDetection-v0
pip install -r requirements.txt
```

v1：

```bat
cd DriverFatigueDetection-v1
pip install -r requirements.txt
```

注意：`mediapipe` 固定为 `0.10.21`，因为当前代码使用 legacy API：

```python
mp.solutions.face_mesh.FaceMesh
```

## 数据集划分

如果需要从原始带标签图片目录重新划分数据集，可在仓库根目录运行：

```bat
python split_dataset.py --source train --output dataset_split --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

如果目标目录已存在并确认要覆盖：

```bat
python split_dataset.py --source train --output dataset_split --overwrite
```

## v0 使用方法

以下命令均在 [DriverFatigueDetection-v0/](DriverFatigueDetection-v0/) 目录下运行。

### 规则阈值评估

```bat
python evaluate.py --test-dir ../dataset_split/test --output reports/rule_eval_metrics.json
```

### 训练 SVM 模型

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/svm_ear_mar.joblib --metrics reports/train_metrics.json
```

### 评估 SVM 模型

```bat
python evaluate.py --test-dir ../dataset_split/test --model models/svm_ear_mar.joblib --output reports/svm_eval_metrics.json
```

### 单张图片预测

规则阈值法：

```bat
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg
```

SVM 模型：

```bat
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg --model models/svm_ear_mar.joblib
```

## v1 使用方法

以下命令均在 [DriverFatigueDetection-v1/](DriverFatigueDetection-v1/) 目录下运行。

### 训练模型

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/hgb_fatigue.joblib --metrics reports/train_metrics.json
```

如果特征代码或数据集内容发生变化，建议强制重建缓存：

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/hgb_fatigue.joblib --metrics reports/train_metrics.json --rebuild-cache
```

### 评估模型

```bat
python evaluate.py --test-dir ../dataset_split/test --model models/hgb_fatigue.joblib --output reports/test_metrics.json
```

### 单张图片预测

```bat
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg --model models/hgb_fatigue.joblib
```

## 当前测试集评估结果

当前评估使用 [dataset_split/test/](dataset_split/test/) 测试集，总计 9,980 张图片，其中 9,888 张检测到人脸并参与有效评估。

| 版本 | 方法 | 准确率 | 说明 |
|---|---|---:|---|
| v0 | 规则阈值 | 59.74% | 基于 EAR/MAR 阈值 |
| v0 | SVM | 68.72% | 使用 4 个 EAR/MAR 特征 |
| v1 | HistGradientBoosting | 92.39% | 使用 35 个手工特征 |

v1 相比 v0 SVM 准确率提升约 **23.68 个百分点**，相比 v0 规则阈值提升约 **32.66 个百分点**。

对应报告文件：

- [DriverFatigueDetection-v0/reports/rule_eval_metrics.json](DriverFatigueDetection-v0/reports/rule_eval_metrics.json)
- [DriverFatigueDetection-v0/reports/svm_eval_metrics.json](DriverFatigueDetection-v0/reports/svm_eval_metrics.json)
- [DriverFatigueDetection-v1/reports/test_metrics.json](DriverFatigueDetection-v1/reports/test_metrics.json)

## 编译检查

可在对应项目目录运行：

```bat
python -m compileall .
```

## 注意事项

- MediaPipe 对人脸清晰度、角度、遮挡较敏感，未检测到人脸的图片会被跳过或返回 `unknown`。
- v1 不使用文件名或路径中的场景名称作为模型特征，以避免标签泄漏。
- 如果相邻帧、同一驾驶员或同一视频片段同时出现在训练/验证/测试集中，测试准确率可能偏乐观。
- 本项目主要用于课程实践和传统机器学习 baseline 对比，不是车载真实安全系统的完整实现。
