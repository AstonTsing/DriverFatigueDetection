# 📦 便携式部署 - 快速指南

## 🎯 目标

创建完全便携的Python环境，让整个项目可以拷贝到U盘，在任何Windows电脑上直接运行，无需安装Python。

## 📋 准备工作

1. 下载 Python 3.10 嵌入式版本 (约10MB)
   - 下载地址: https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip
   - 保存到项目根目录

## 🚀 一键配置

### Windows系统

```bash
# 双击运行配置脚本
setup_portable.bat

# 等待配置完成(约10-30分钟，取决于网速)
```

配置脚本会自动:
- ✅ 解压Python嵌入式包
- ✅ 配置Python环境
- ✅ 安装pip
- ✅ 安装所有依赖包

### 手动配置(如脚本失败)

```bash
# 1. 解压Python
解压 python-3.10.11-embed-amd64.zip 到 python_env/

# 2. 修改配置文件
编辑 python_env\python310._pth
在末尾添加: import site

# 3. 下载并安装pip
# 访问 https://bootstrap.pypa.io/get-pip.py
# 保存为 get-pip.py
python_env\python.exe get-pip.py

# 4. 安装依赖
python_env\python.exe -m pip install -r requirements_gui.txt
```

## 🎮 运行应用

### 首次运行前

⚠️ **重要**: 先修复Python文件缩进错误

1. 使用VSCode打开 `predictor.py` 和 `detection_gui.py`
2. 全选 (Ctrl+A)
3. 格式化文档 (Shift+Alt+F)
4. 保存

### 启动应用

```bash
# 双击运行
run_gui.bat
```

或者手动运行:
```bash
python_env\python.exe detection_gui.py
```

## 💾 便携式分发

配置完成后,整个目录结构:

```
系统实践小程序/              (总大小: 约 2-3 GB)
├── python_env/         (约 1.5-2.5 GB, Python + 所有依赖)
│   ├── python.exe
│   ├── Lib/
│   ├── Scripts/
│   └── ...
├── run_gui.bat             (一键启动)
├── setup_portable.bat      (配置脚本)
├── predictor.py            (⚠️需修复缩进)
├── detection_gui.py        (⚠️需修复缩进)
├── DriverFatigueDetection/ (包含模型)
├── drowsiness_detection/   (包含模型)
└── Introduction_to_AI-Project_chenbh/ (包含模型)
```

### 分发步骤

1. **配置完成后**，整个文件夹可以:
   - 压缩为 `.zip` 或 `.7z`
   - 拷贝到U盘
   - 上传到云盘
   - 分享给他人

2. **在其他电脑使用**:
   - 解压/拷贝整个文件夹
   - 双击 `run_gui.bat`
   - 无需安装任何软件！

## 📊 体积优化

### 完整版 (约 2-3 GB)
- 支持所有5种检测方法
- 包含 TensorFlow、PyTorch、YOLO等
- 使用 `requirements_gui.txt`

### 精简版 (约 500 MB - 1 GB)
- 仅支持传统ML方法(V0/V1)
- 不包含深度学习框架
- 使用 `requirements_minimal.txt`

选择精简版:
```bash
python_env\python.exe -m pip install -r requirements_minimal.txt
```

### 进一步优化

删除不需要的文件:
```bash
# 删除测试集(保留少量示例)
# 删除 DriverFatigueDetection/dataset_split/test/ 中的大部分图片

# 删除不用的模型
# 例如只保留最准确的 V1-HGB 模型
```

## ⚠️ 注意事项

### 路径问题
- ✅ 支持中文路径
- ✅ 支持空格路径
- ⚠️ 避免过长路径 (Windows 260字符限制)

### 权限问题
- ✅ 一般用户权限即可
- ⚠️ 某些企业电脑可能有安全策略限制

### 兼容性
- ✅ Windows 10/11
- ✅ 64位系统
- ⚠️ 不支持 Windows 7 (TensorFlow限制)
- ❌ 不支持 32位系统

## 🔧 故障排除

### 问题1: setup_portable.bat 运行失败

**原因**: 网络问题或下载失败

**解决**:
1. 手动下载 Python嵌入式zip
2. 手动下载 get-pip.py
3. 按"手动配置"步骤操作

### 问题2: 依赖安装超时

**原因**: 网络速度慢

**解决**:
```bash
# 使用国内镜像源
python_env\python.exe -m pip install -r requirements_gui.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题3: run_gui.bat 报错

**原因**: Python文件缩进未修复

**解决**: 使用VSCode格式化两个.py文件

### 问题4: 提示缺少模型

**原因**: 模型文件未包含

**解决**: 确保至少一个模型文件存在

## 📚 相关文档

- [SUMMARY.md](SUMMARY.md) - 项目完整说明
- [USAGE.md](USAGE.md) - 详细使用指南
- [PORTABLE.md](PORTABLE.md) - 便携式原理说明
- [README.md](README.md) - 项目总览

## ✅ 检查清单

部署前检查:
- [ ] Python嵌入式包已下载并解压
- [ ] pip已安装
- [ ] 所有依赖已安装
- [ ] predictor.py 和 detection_gui.py 缩进已修复
- [ ] run_gui.bat 可以正常启动
- [ ] 至少一个模型可用

分发前检查:
- [ ] 测试在本机运行正常
- [ ] 检查文件总大小
- [ ] 考虑是否需要精简
- [ ] 准备README说明

## 🎉 完成！

配置完成后，您将拥有一个**完全独立、即插即用**的驾驶员疲劳检测系统！

---

**提示**: 首次配置需要时间，但配置一次后可以无限次复制使用！💪
