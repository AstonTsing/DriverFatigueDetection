from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

base = Path(__file__).resolve().parent
python_exe = base / 'python_env' / 'python.exe'
server = base / 'prediction_server.py'
img = next((base / 'DriverFatigueDetection' / 'dataset_split' / 'test' / 'drowsy').rglob('*.jpg'))
methods = [
    'DriverFatigueDetection-v0 (SVM)',
    'DriverFatigueDetection-v1 (HistGradientBoosting)',
    'YOLOv11 Classification',
    'TensorFlow Baseline CNN',
    'TensorFlow MobileNetV2',
]

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
proc = subprocess.Popen(
    [str(python_exe), str(server)],
    cwd=str(base),
    env=env,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    encoding='utf-8',
    errors='replace',
    bufsize=1,
)

def request(payload: dict) -> dict:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError('no response')
    return json.loads(line)

try:
    preload = request({'cmd': 'preload'})
    print('preload', preload.get('success'), preload.get('status'))
    single = request({'cmd': 'predict', 'method': 'TensorFlow MobileNetV2', 'image': str(img)})
    print('single', single.get('success'), (single.get('prediction') or {}).get('label'), single.get('error'))
    compare = request({'cmd': 'compare', 'methods': methods, 'image': str(img)})
    print('compare', compare.get('success'), len(compare.get('results') or []))
    for result in compare.get('results') or []:
        print(result.get('method'), result.get('success'), (result.get('prediction') or {}).get('label'), result.get('error'))
    if not preload.get('success') or not single.get('success') or not all(r.get('success') for r in compare.get('results') or []):
        raise SystemExit(1)
finally:
    try:
        request({'cmd': 'exit'})
    except Exception:
        pass
    proc.terminate()
