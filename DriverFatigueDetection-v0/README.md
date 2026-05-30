# DriverFatigueDetection-v0

这是一个不依赖深度学习训练的驾驶员疲劳检测基线项目。项目使用 MediaPipe Face Mesh 提取人脸关键点，并计算眼睛纵横比 EAR（Eye Aspect Ratio）和嘴部纵横比 MAR（Mouth Aspect Ratio），支持两种检测方式：

1. 规则阈值法：根据 EAR/MAR 阈值直接判断是否疲劳。
2. 传统机器学习法：使用 EAR/MAR 特征训练 SVM 二分类器。

## 目录结构

```text
DriverFatigueDetection-v0/
├── config/
│   └── rule_config.json       # EAR/MAR 阈值与 MediaPipe 检测参数
├── models/                    # 训练后模型输出目录
├── reports/                   # 评估指标输出目录
├── src/
│   ├── __init__.py
│   ├── dataset.py             # 数据集读取工具
│   ├── landmark_features.py   # 人脸关键点、EAR、MAR 特征提取
│   └── rule_detector.py       # 规则阈值疲劳分类器
├── evaluate.py                # 测试集评估脚本
├── predict.py                 # 单张图片预测脚本
├── requirements.txt           # Python 依赖
└── train.py                   # SVM 训练脚本
```

## 数据集要求

默认使用项目同级目录下的 `dataset_split` 数据集，结构如下：

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

类别含义：

- `drowsy`：疲劳/困倦样本，标签为 1。
- `notdrowsy`：非疲劳样本，标签为 0。

## 安装依赖

本项目使用名为 `driver` 的 Conda 虚拟环境。请先创建并激活该环境，再安装依赖：

```bash
conda create -n driver python=3.12
conda activate driver
pip install -r requirements.txt
```

## 规则阈值法评估

在 `DriverFatigueDetection-v0` 目录下运行：

```bash
python evaluate.py --test-dir ../dataset_split/test --output reports/rule_eval_metrics.json
```

输出文件包含准确率、混淆矩阵、分类报告、逐图预测详情，以及未检测到人脸的图片列表。

## 训练 SVM 传统分类器

```bash
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/svm_ear_mar.joblib --metrics reports/train_metrics.json
```

训练脚本会：

1. 对训练集和验证集图片提取 `[ear_left, ear_right, ear_mean, mar]` 特征。
2. 跳过未检测到人脸的图片。
3. 使用 `StandardScaler + SVC` 训练传统机器学习二分类器。
4. 保存模型到 `models/svm_ear_mar.joblib`。
5. 保存训练/验证指标到 `reports/train_metrics.json`。

## 使用 SVM 模型评估

```bash
python evaluate.py --test-dir ../dataset_split/test --model models/svm_ear_mar.joblib --output reports/svm_eval_metrics.json
```

## 单张图片预测

测试集图片通常位于类别目录下的子目录中，例如：

```text
../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg
```

规则阈值法：

```bash
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg
```

SVM 模型：

```bash
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg --model models/svm_ear_mar.joblib
```

如果需要查找可用图片路径，可以在 Windows cmd 中运行：

```bat
dir ..\dataset_split\test\drowsy /s /b
dir ..\dataset_split\test\notdrowsy /s /b
```

预测结果以 JSON 输出，包含人脸特征和预测标签。

## 阈值配置

阈值位于 `config/rule_config.json`：

```json
{
  "ear_threshold": 0.23,
  "mar_threshold": 0.62,
  "eye_closed_score_weight": 1.0,
  "mouth_open_score_weight": 0.55,
  "fatigue_score_threshold": 1.0,
  "min_detection_confidence": 0.5,
  "min_tracking_confidence": 0.5
}
```

默认逻辑：

- 当 `ear_mean < ear_threshold` 时，认为眼睛闭合。
- 当 `mar > mar_threshold` 时，认为嘴部张开/打哈欠。
- 疲劳分数大于等于 `fatigue_score_threshold` 时，判定为 `drowsy`。

如果测试集效果不理想，可以根据验证集统计结果调整 EAR/MAR 阈值。

## 注意事项

- MediaPipe 对图片中的人脸清晰度、角度、遮挡较敏感；未检测到人脸的图片不会参与训练和有效评估。
- 规则阈值法适合作为可解释 baseline，但泛化能力通常弱于经过标注数据训练的分类器。
- SVM 只使用 4 个几何特征，训练速度快，但仍受人脸关键点质量影响。
