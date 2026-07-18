@echo off
REM 驾驶员疲劳检测系统 - 便携式启动脚本
REM 使用本地 python_env 运行

echo =========================
echo 驾驶员疲劳检测系统
echo ================
echo.

REM 检查便携式Python环境
if not exist "python_env\python.exe" (
    echo [错误] 未找到便携式Python环境！
  echo.
    echo 请先运行配置脚本:
    echo   setup_portable.bat
    echo.
    pause
    exit /b 1
)

REM 检查必要文件
if not exist "predictor.py" (
    echo [错误] 未找到 predictor.py！
    pause
    exit /b 1
)

if not exist "detection_gui.py" (
    echo [错误] 未找到 detection_gui.py！
    pause
    exit /b 1
)

echo [启动] 正在启动图形界面...
echo.

REM 使用本地Python运行
python_env\python.exe detection_gui.py

REM 检查退出码
if errorlevel 1 (
    echo.
    echo =============================
    echo [错误] 程序运行失败！
    echo ===============
    echo.
    echo 可能的原因:
    echo 1. Python文件存在缩进错误
    echo    解决: 使用VSCode打开文件按 Shift+Alt+F 格式化
    echo.
    echo 2. 缺少依赖包
    echo    解决: 重新运行 setup_portable.bat
    echo.
    echo 3. 模型文件缺失
    echo    解决: 确保至少有一个模型文件存在
    echo.
    pause
    exit /b 1
)

echo.
echo 程序已正常退出
pause
