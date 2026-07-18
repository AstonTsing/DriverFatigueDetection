from __future__ import annotations

from pathlib import Path
import json
from detection_gui import PredictionThread

base = Path(__file__).resolve().parent
img = next((base / 'DriverFatigueDetection' / 'dataset_split' / 'test' / 'drowsy').rglob('*.jpg'))
methods = [
    'DriverFatigueDetection-v0 (SVM)',
    'DriverFatigueDetection-v1 (HistGradientBoosting)',
    'YOLOv11 Classification',
    'TensorFlow Baseline CNN',
    'TensorFlow MobileNetV2',
]
thread = PredictionThread(methods, str(img), comparison=True)
results = [thread.run_one_method(method) for method in methods]
summary = {'comparison': True, 'results': results}
print(json.dumps([{'method': r.get('method'), 'success': r.get('success'), 'label': (r.get('prediction') or {}).get('label'), 'error': r.get('error')} for r in results], ensure_ascii=False, indent=2))
if not all(r.get('success') for r in results):
    raise SystemExit(1)
