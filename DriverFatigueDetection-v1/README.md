# DriverFatigueDetection-v1

这是一个不训练深度学习模型的驾驶员疲劳检测传统机器学习项目。相比 `DriverFatigueDetection-v0`，v1 仍然使用 MediaPipe Face Mesh 提取人脸关键点，但不再只依赖 EAR/MAR 四个特征，而是提取更丰富的眼部、嘴部、头部姿态代理、眉眼距离和图像质量特征，并使用 scikit-learn 的 `HistGradientBoostingClassifier` 进行二分类。

## 与 v0 的主要区别

- v0：规则阈值法或 `StandardScaler + SVC`，输入特征为 `[ear_left, ear_right, ear_mean, mar]`。
- v1：MediaPipe 关键点 + 多组人工几何特征 + `SimpleImputer + HistGradientBoostingClassifier`。
- v1 会缓存特征，避免每次训练/评估都重复跑完整 MediaPipe 特征提取。

本项目不训练 CNN、RNN、Transformer 等深度学习模型。MediaPipe 只用于人脸关键点提取。

## 数据集要求

默认使用项目同级目录下的 `dataset_split` 数据集：

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

标签固定为：

- `notdrowsy`：0
- `drowsy`：1

类别目录下允许有子目录，程序会递归查找图片。

## 安装依赖

使用名为 `driver` 的 Conda 环境：

```bat
conda create -n driver python=3.12
conda activate driver
cd DriverFatigueDetection-v1
pip install -r requirements.txt
```

`mediapipe` 固定为 `0.10.21`，因为当前代码使用 `mp.solutions.face_mesh.FaceMesh`。

## 训练模型

在 `DriverFatigueDetection-v1` 目录下运行：

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/hgb_fatigue.joblib --metrics reports/train_metrics.json
```

训练会：

1. 提取训练集和验证集的 v1 人工特征。
2. 跳过未检测到人脸的图片。
3. 使用类别平衡样本权重训练 `HistGradientBoostingClassifier`。
4. 保存模型到 `models/hgb_fatigue.joblib`。
5. 保存训练/验证指标到 `reports/train_metrics.json`。

第一次运行会比较慢，因为需要提取特征。后续默认复用 `cache/` 下的特征缓存。

强制重建缓存：

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/hgb_fatigue.joblib --metrics reports/train_metrics.json --rebuild-cache
```

禁用缓存：

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --no-cache
```

## 评估模型

```bat
python evaluate.py --test-dir ../dataset_split/test --model models/hgb_fatigue.joblib --output reports/test_metrics.json
```

输出文件包含：

- 准确率
- 平衡准确率
- macro/weighted F1
- ROC AUC
- 混淆矩阵
- 分类报告
- 跳过的人脸未检出图片
- 每张图片的预测详情和特征值

## 单张图片预测

测试集图片通常位于类别目录或其子目录下，例如：

```text
../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg
```

预测命令：

```bat
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg --model models/hgb_fatigue.joblib
```

如果需要查找可用图片路径，可以在 Windows cmd 中运行：

```bat
dir ..\dataset_split\test\drowsy /s /b
dir ..\dataset_split\test\notdrowsy /s /b
```

预测结果以 JSON 输出，包含：

- 是否检测到人脸
- v1 特征值
- 预测标签
- `prob_drowsy`
- 各类别概率

## 特征缓存

默认缓存目录为：

```text
cache/
```

缓存会记录：

- 特征 schema 版本
- 特征名顺序
- split 路径
- 图片路径列表
- 标签列表
- 特征矩阵
- 未检测到人脸的图片列表

如果特征 schema、图片列表或标签发生变化，缓存不会被复用。

## 配置

模型和 MediaPipe 参数位于：

```text
config/model_config.json
```

默认模型为：

```text
SimpleImputer(strategy="median") + HistGradientBoostingClassifier
```

## 注意事项

- v1 目标是提高传统机器学习 baseline 的准确率，不是端到端图像深度学习方案。
- 如果数据集中相邻帧或同一驾驶员同时出现在训练/验证/测试集中，评估结果可能偏乐观。
- v1 不使用文件名或路径中的场景名称作为模型特征，以避免标签泄漏。
