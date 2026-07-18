from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    base_dir = Path(sys.executable).resolve().parent
    pythonw = base_dir / "python_env" / "pythonw.exe"
    python = base_dir / "python_env" / "python.exe"
    script = base_dir / "detection_gui.py"

    if not script.exists():
        script = Path(__file__).resolve().parent / "detection_gui.py"
    exe = pythonw if pythonw.exists() else python
    if not exe.exists():
        print(f"未找到便携式 Python: {exe}")
        return 1
    if not script.exists():
        print(f"未找到界面脚本: {script}")
        return 1

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    python_env = base_dir / "python_env"
    site_packages = python_env / "Lib" / "site-packages"
    qt_plugin_dir = site_packages / "PyQt5" / "Qt5" / "plugins" / "platforms"
    if qt_plugin_dir.exists():
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_plugin_dir)
        env["QT_PLUGIN_PATH"] = str(qt_plugin_dir.parent)
    dll_dirs = [
        python_env,
        site_packages,
        site_packages / "mediapipe" / "python",
        site_packages / "PyQt5" / "Qt5" / "bin",
        site_packages / "cv2",
        site_packages / "tensorflow",
        site_packages / "torch" / "lib",
        site_packages / "numpy.libs",
        site_packages / "scipy.libs",
        site_packages / "sklearn" / ".libs",
    ]
    existing = [str(path) for path in dll_dirs if path.exists()]
    env["PATH"] = os.pathsep.join(existing + [env.get("PATH", "")])
    subprocess.Popen([str(exe), str(script)], cwd=str(script.parent), env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
