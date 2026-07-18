# 驾驶员疲劳检测GUI应用 - 安装和使用指南

## 当前状态

已完成:
- ✅ 预测模块 (`predictor.py`) - 集成5种检测方法
- ✅ GUI界面 (`detection_gui.py`) - PyQt5图形界面
- ✅ 打包脚本 (`build_exe.bat`)
- ✅ 依赖列表 (`requirements_gui.txt`)
- ✅ MobileNetV2训练脚本 (`Introduction_to_AI-Project_chenbh/train_mobilenet.py`)

待修复:
- ⚠️ Python文件缩进问题(由AI生成时产生)

## 快速修复缩进问题

由于AI工具在生成长文件时可能产生缩进错误,建议使用IDE自动修复:

### 使用VSCode修复

1. 打开 `predictor.py` 和 `detection_gui.py`
2. 全选 (Ctrl+A)
3. 格式化文档 (Shift+Alt+F)
4. 保存

### 使用PyCharm修复

1. 打开文件
2. Code -> Reformat Code
3. 保存

### 使用命令行工具

```bash
# 安装 autopep8
pip install autopep8

# 自动修复
cd "E:\Disk-G\大二春季学期\人工智能导论\AI系统实践\系统实践小程序"
autopep8 --in-place --aggressive predictor.py
autopep8 --in-place --aggressive detection_gui.py
```

## 安装依赖

```bash
cd "E:\Disk-G\大二春季学期\人工智能导论\AI系统实践\系统实践小程序"

# 方式1: 使用requirements文件
pip install -r requirements_gui.txt

# 方式2: 手动安装核心包
pip install PyQt5 tensorflow torch torchvision ultralytics opencv-python mediapipe scikit-learn numpy joblib pillow
```

## 运行GUI应用

```bash
python detection_gui.py
```

## 可用的检测方法

GUI会自动检测以下方法的可用性(基于模型文件是否存在):

1. **DriverFatigueDetection-v0 (SVM)** 
   - 模型: `DriverFatigueDetection/DriverFatigueDetection-v0/models/svm_ear_mar.joblib`
   - 特点: 基于EAR/MAR特征的SVM分类

2. **DriverFatigueDetection-v1 (HistGradientBoosting)**
   - 模型: `DriverFatigueDetection/DriverFatigueDetection-v1/models/hgb_fatigue.joblib`
   - 特点: 35个手工特征 + 梯度提升

3. **YOLOv11 Classification**
   - 模型: `drowsiness_detection/outputs/test1/best.pt`
   - 特点: 端到端深度学习分类

4. **TensorFlow Baseline CNN**
   - 模型: `Introduction_to_AI-Project_chenbh/model_baseline.h5`
   - 特点: 简单CNN架构

5. **TensorFlow MobileNetV2**
   - 模型: `Introduction_to_AI-Project_chenbh/model_mobilenetv2.h5`
   - 特点: 迁移学习,准确率最高(96.6%)
   - ⚠️ 需要先训练

## 训练MobileNetV2模型(可选)

```bash
cd Introduction_to_AI-Project_chenbh
python train_mobilenet.py
```

训练时间: 约2小时(取决于硬件)

## 使用方法

1. 启动应用后,左侧选择检测方法
2. 点击"选择图片",从数据集中选择测试图片
   - 推荐路径: `DriverFatigueDetection/dataset_split/test/`
3. 点击"开始检测"
4. 右侧查看:
   - 图片预览
   - 检测结果(疲劳/清醒)
   - 置信度/概率
   - 提取的特征
   - 完整JSON结果

## 打包为EXE(可选)

```bash
# 安装PyInstaller
pip install pyinstaller

# 运行打包脚本
build_exe.bat

# 生成的exe位置
dist\detection.exe
```

## 常见问题

### Q: 提示"未找到可用模型"
A: 检查模型文件是否存在于对应路径,至少需要一个模型文件

### Q: 检测失败:"未检测到人脸"
A: 该图片可能人脸不清晰或角度问题,尝试其他图片

### Q: 导入错误
A: 确保所有依赖已安装,特别是tensorflow/torch/ultralytics

### Q: MobileNetV2选项不可用
A: 该模型需要先训练,运行 `train_mobilenet.py` 或暂时使用其他4种方法

## 技术架构

```
detection_gui.py (GUI主程序)
  ├─ PyQt5 (界面框架)
  ├─ PredThread (异步预测线程)
  └─ predictor.py (预测模块)
       ├─ predict_v0_svm()
     ├─ predict_v1_hgb()
    ├─ predict_yolo()
       ├─ predict_tf_baseline()
       └─ predict_tf_mobilenet()
```

## 项目文件

- `predictor.py` - 统一预测接口
- `detection_gui.py` - PyQt5图形界面
- `requirements_gui.txt` - Python依赖
- `build_exe.bat` - PyInstaller打包脚本
- `USAGE.md` - 本文档

## 联系和反馈

如遇问题,请检查:
1. Python版本(推荐3.8-3.10)
2. 依赖包版本兼容性
3. 模型文件完整性
4. 数据集路径正确性
