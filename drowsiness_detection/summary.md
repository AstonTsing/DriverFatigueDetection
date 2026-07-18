# Summary

**这是我和codex的交互文档，我在里面写的指令不要动，请你根据我的指示在这个文档的指定位置增改内容，保持语言简洁清晰**

## 梳理代码

帮我看一下整个repo的结构，然后将一下代码的逻辑和结构，主要就是jupyternotebook里的文件，看一下主要的功能，代码分区。

### 仓库结构

- `train.py`：终端训练入口，负责 YOLOv11 训练、按 `eval_freq` 抽样评估、输出 loss/成功率，并保存最佳 checkpoint。
- `run_train.sh`：训练启动脚本，集中放置可调参数。
- `Deep_Learning_for_Safer_Roads_YOLOv11_Driver_Drowsiness_Detection.ipynb`：notebook 版本流程，保留 YOLOv11 本地数据集训练、预测、备份与导出。
- `README.md`：项目说明，介绍疲劳驾驶检测任务和 YOLOv11 本地训练路线。
- `environment.yml`：conda 环境配置文件。
- `setup_env.sh`：一键创建/更新 conda 环境的脚本。
- `doc/screenshots/`：README 展示用截图。
- `LICENSE.txt`：许可证文件。

### Notebook 代码逻辑

当前主要训练入口改为 `train.py` + `run_train.sh`。Notebook 保留为交互式参考流程。

### 1. 项目介绍

开头的 Markdown 说明项目目标：检测驾驶员是否疲劳，类别是 `drowsy` 和 `notdrowsy`。当前 notebook 聚焦 YOLOv11 + 本地数据集单一路线。

### 2. YOLOv11 本地数据集流程

- 环境准备：安装 `ultralytics==8.3.40`、`tensorflow`、`matplotlib`、`pillow`、`opencv-python`。
- 加载数据：直接读取本地 `/home/hebu/dd_dataset` 数据集，并检查 `train`、`valid` 或 `val` 目录。
- 数据可视化：用 `glob` 和 `matplotlib` 抽样展示训练图片。
- 训练模型：优先检查 `runs/classify/train/weights/best.pt`，如果存在则直接加载；如果有 `runs_backup.zip` 则解压；否则使用 `YOLO("yolo11n-cls.pt")` 训练 20 个 epoch。
- 预测评估：自动读取验证集类别文件夹，并从 `drowsy` 和 `notdrowsy` 各抽样 100 张图片，用 YOLO 模型预测并计算准确率。
- 结果查看：显示混淆矩阵、归一化混淆矩阵和训练结果图。
- 备份模型：将 `runs` 目录压缩成 `runs_backup.zip`。
- 移动端导出：使用 `model.export(format="tflite")` 导出 TensorFlow Lite 模型。

### 3. 终端训练脚本结构

- `run_train.sh`：配置 `TASK_NAME`、`DATA_DIR`、`MODEL`、`EPOCHS`、`BATCH_SIZE`、`EVAL_FREQ`、`NUM_EVAL`、`DEVICE` 等参数，并调用 `train.py`。
- `train.py`：训练过程中实时输出 loss；每隔 `eval_freq` 个 epoch 从验证集抽样 `num_eval` 张图片评估成功率；如果成功率提升，则保存 `outputs/<task_name>/best.pt`。
- 内置完整验证被关闭，训练中只使用抽样评估，避免每次扫描整个验证集。

### 总体结构特点

- 代码是教学/实验型 notebook，按“数据准备 -> 模型训练 -> 评估 -> 保存/部署”的顺序组织。
- 当前只保留 YOLOv11 本地数据集路线，重点在终端训练、周期抽样评估、保存最佳权重和 TFLite 导出。
- 当前仓库没有独立 Python 模块，核心逻辑都在 notebook 单文件中，复现实验时需要准备 `/home/hebu/dd_dataset` 本地分类数据集。
