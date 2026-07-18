@echo off
REM 自动修复Python文件缩进问题

echo =============================
echo 驾驶员疲劳检测系统 - 代码修复工具
echo =========================
echo.

echo [1/3] 检查 autopep8...
python -c "import autopep8" 2>nul
if errorlevel 1 (
    echo autopep8 未安装，正在安装...
    pip install autopep8
    if errorlevel 1 (
        echo [错误] 安装失败！
        pause
        exit /b 1
    )
)

echo [2/3] 修复 predictor.py...
autopep8 --in-place --aggressive --aggressive predictor.py
if errorlevel 1 (
    echo [警告] predictor.py 修复可能不完整
) else (
    echo [成功] predictor.py 已修复
)

echo [3/3] 修复 detection_gui.py...
autopep8 --in-place --aggressive --aggressive detection_gui.py
if errorlevel 1 (
    echo [警告] detection_gui.py 修复可能不完整
) else (
    echo [成功] detection_gui.py 已修复
)

echo.
echo ===============================
echo 修复完成！
echo ====================
echo.
echo 下一步:
echo 1. 安装依赖: pip install -r requirements_gui.txt
echo 2. 运行应用: python detection_gui.py
echo.
echo 如果仍有问题，请使用IDE(VSCode/PyCharm)打开文件并格式化
echo.
pause
