@echo off
REM 驾驶员疲劳检测系统 - 打包脚本 (Windows)

echo ===========================
echo 驾驶员疲劳检测系统 - PyInstaller 打包
echo =======================
echo.

REM 检查 PyInstaller
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [错误] 未安装 PyInstaller
    echo 请运行: pip install pyinstaller
    pause
    exit /b 1
)

echo [1/4] 清理旧的打包文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist detection.spec del /q detection.spec

echo [2/4] 准备数据文件...
REM 确保必要的目录存在
if not exist "DriverFatigueDetection\dataset_split\test" (
    echo [错误] 数据集目录不存在！
    pause
    exit /b 1
)

echo [3/4] 开始打包...
pyinstaller ^
    --name=detection ^
    --onefile ^
    --windowed ^
    --icon=NONE ^
    --add-data "DriverFatigueDetection;DriverFatigueDetection" ^
    --add-data "drowsiness_detection/outputs/test1/best.pt;drowsiness_detection/outputs/test1" ^
    --add-data "Introduction_to_AI-Project_chenbh/model_baseline.h5;Introduction_to_AI-Project_chenbh" ^
    --hidden-import=PyQt5 ^
    --hidden-import=tensorflow ^
    --hidden-import=torch ^
    --hidden-import=ultralytics ^
    --hidden-import=mediapipe ^
    --hidden-import=cv2 ^
   --hidden-import=sklearn ^
    --hidden-import=joblib ^
    --collect-all mediapipe ^
    --collect-all ultralytics ^
    detection_gui.py

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！
    pause
    exit /b 1
)

echo [4/4] 复制数据文件到输出目录...
REM 复制数据集（仅测试集）
xcopy /E /I /Y "DriverFatigueDetection\dataset_split\test" "dist\DriverFatigueDetection\dataset_split\test"

echo.
echo =====================
echo 打包完成！
echo =============================
echo.
echo 可执行文件位置: dist\detection.exe
echo.
echo 注意事项:
echo 1. 首次运行可能需要较长时间加载模型
echo 2. 确保数据集文件夹在同一目录下
echo 3. 如果缺少 MobileNetV2 模型,该选项将不可用
echo.
pause
