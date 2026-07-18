@echo off
REM 便携式Python环境配置脚本
REM 自动下载、配置、安装依赖

echo ==================================
echo 驾驶员疲劳检测系统 - 便携式环境配置向导
echo ============================
echo.

REM 检查是否已有python_env
if exist "python_env\python.exe" (
    echo [检测] 已存在 python_env 目录
    set /p REINSTALL="是否重新配置? (y/n): "
    if /i not "%REINSTALL%"=="y" (
        echo 跳过配置，直接安装依赖...
        goto INSTALL_DEPS
    )
    echo 清理旧环境...
    rmdir /s /q python_env
)

echo.
echo =====================================
echo 第一步: 下载Python嵌入式版本
echo =======================
echo.
echo 需要下载 Python 3.10 嵌入式版本 (约 10 MB)
echo.
echo 下载地址: https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip
echo.
echo 请手动下载后放到当前目录，然后按任意键继续...
pause >nul

REM 检查zip文件
if not exist "python-3.10.*-embed-amd64.zip" (
    echo [错误] 未找到 Python 嵌入式zip文件！
    echo 请下载 python-3.10.x-embed-amd64.zip 并放到当前目录
    pause
    exit /b 1
)

echo [1/6] 解压Python嵌入式包...
powershell -command "Expand-Archive -Path 'python-3.10.*-embed-amd64.zip' -DestinationPath 'python_env' -Force"
if errorlevel 1 (
    echo [错误] 解压失败！
    pause
    exit /b 1
)
echo [成功] Python嵌入式包已解压

echo.
echo [2/6] 配置Python环境...
REM 查找._pth文件并修改
for %%f in (python_env\python*._pth) do (
    echo import site >> "%%f"
    echo [成功] 已启用 site-packages: %%f
)

echo.
echo [3/6] 下载 get-pip.py...
powershell -command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"
if errorlevel 1 (
    echo [警告] 自动下载失败，请手动下载:
    echo https://bootstrap.pypa.io/get-pip.py
    echo 保存为 get-pip.py 后按任意键继续...
    pause >nul
)

if not exist "get-pip.py" (
    echo [错误] 未找到 get-pip.py！
    pause
    exit /b 1
)

echo.
echo [4/6] 安装 pip...
python_env\python.exe get-pip.py
if errorlevel 1 (
    echo [错误] pip 安装失败！
    pause
    exit /b 1
)
echo [成功] pip 已安装

echo.
echo [5/6] 升级 pip...
python_env\python.exe -m pip install --upgrade pip

:INSTALL_DEPS
echo.
echo ======================================
echo 第二步: 安装项目依赖
echo =====================
echo.
echo 正在安装依赖包，这可能需要 10-30 分钟...
echo 请保持网络连接并耐心等待...
echo.

python_env\python.exe -m pip install -r requirements_gui.txt --no-warn-script-location
if errorlevel 1 (
    echo.
    echo [警告] 部分依赖安装可能失败
    echo 可以尝试分批安装:
    echo python_env\python.exe -m pip install PyQt5
    echo python_env\python.exe -m pip install tensorflow
    echo python_env\python.exe -m pip install torch torchvision
    echo python_env\python.exe -m pip install ultralytics
    pause
)

echo.
echo [6/6] 验证环境...
python_env\python.exe -c "import PyQt5; import tensorflow; import torch; import ultralytics; print('[成功] 核心依赖已安装')"
if errorlevel 1 (
    echo [警告] 环境验证未完全通过，可能缺少某些依赖
    echo 请检查上方错误信息
)

echo.
echo ==================================
echo 配置完成！
echo ==============================
echo.
echo 环境目录大小:
dir python_env /s | find "个文件"
echo.
echo 下一步:
echo 1. 修复 predictor.py 和 detection_gui.py 的缩进
echo    (使用 VSCode 打开后按 Shift+Alt+F)
echo 2. 运行应用: run_gui.bat
echo.
echo 便携式部署:
echo - 整个文件夹可以拷贝到U盘
echo - 在其他电脑上直接运行 run_gui.bat
echo.
pause
