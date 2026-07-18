# 最终状态：可双击运行

当前目录已完成便携式部署，可以直接双击根目录下的 `detection.exe` 打开驾驶员疲劳检测图形界面。

## 已完成

- 已安装便携式 Python 环境：`python_env/`
- 已安装所有运行依赖：PyQt5、OpenCV、MediaPipe、scikit-learn、Ultralytics、PyTorch、TensorFlow 等
- 已确认 5 种检测方法模型均存在：
  - DriverFatigueDetection-v0 (SVM)
  - DriverFatigueDetection-v1 (HistGradientBoosting)
  - YOLOv11 Classification
  - TensorFlow Baseline CNN
  - TensorFlow MobileNetV2
- 已修复 `predictor.py` 和 `detection_gui.py` 的语法/缩进问题
- 已修复 v0/v1 两个项目 `src` 包导入冲突
- 已修复中文路径下 MediaPipe 资源加载问题
- 已修复 PyQt5 的 Qt platform plugin (`qwindows.dll`) 路径问题
- 已修复 GUI / `pythonw.exe` 环境下 MediaPipe `_framework_bindings` DLL 初始化失败问题
- 已将 GUI 检测改为独立 `predict_cli.py` 子进程预测，隔离 PyQt5 与 MediaPipe/TensorFlow/PyTorch 的 DLL 冲突
- 已修复 TensorFlow Baseline CNN 在中文路径下无法读取图片的问题
- 已修复 TensorFlow MobileNetV2 旧版 Keras 模型在新版 TensorFlow 中的 `DepthwiseConv2D(groups=1)` 兼容问题：当前通过重建架构并加载权重进行推理
- 已新增两种可切换功能：`单模型检测` 和 `五模型对比检测`
- `五模型对比检测` 会对同一张图片依次调用 5 种方法，输出横向对比表、类别概率和多数投票结论
- 已重新生成根目录启动器：`detection.exe`
- 已验证：
  - Python 依赖可导入
  - 5 种方法可识别
  - 5 种方法均可对样例图片完成预测
  - v0/SVM 在 `pythonw.exe`（GUI 同类环境）下可完成预测
  - TensorFlow Baseline CNN 可读取中文路径图片并预测
  - TensorFlow MobileNetV2 可在当前 TensorFlow 环境中加载权重并预测
  - `detection.exe` 可启动 GUI 进程

## 使用方式

直接双击：

```text
detection.exe
```

备用方式：

```bat
run_gui.bat
```

或：

```bat
python_env\python.exe detection_gui.py
```

## 便携式拷贝说明

请复制整个 `系统实践小程序` 文件夹到 U 盘或另一台电脑，不要只复制 `detection.exe`。`detection.exe` 是轻量启动器，需要同目录下的：

- `python_env/`
- `detection_gui.py`
- `predictor.py`
- 三个项目目录及其模型文件

保持目录结构不变即可在另一台 Windows 电脑双击运行。
