from pathlib import Path
import os
import subprocess
from PyQt5.QtCore import Qt  # noqa: F401 - force Qt/PyQt DLLs to load first

base = Path(__file__).resolve().parent
img = next((base / 'DriverFatigueDetection' / 'dataset_split' / 'test' / 'notdrowsy').rglob('*.jpg'))
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
completed = subprocess.run(
    [str(base / 'python_env' / 'python.exe'), str(base / 'predict_cli.py'), '--method', 'DriverFatigueDetection-v0 (SVM)', '--image', str(img)],
    cwd=str(base),
    env=env,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',
    timeout=600,
)
print('returncode', completed.returncode)
print((completed.stdout or '').splitlines()[-1])
raise SystemExit(completed.returncode)
