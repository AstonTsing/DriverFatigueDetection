from __future__ import annotations

import json
from pathlib import Path

import predictor

base = Path(__file__).resolve().parent
image = next((base / 'DriverFatigueDetection' / 'dataset_split' / 'test' / 'drowsy').rglob('*.jpg'))
summary = []
for method in predictor.get_available_methods():
    result = predictor.predict_with_method(method, image)
    item = {
        'method': method,
        'success': bool(result.get('success')),
        'label': (result.get('prediction') or {}).get('label'),
        'error': result.get('error'),
    }
    summary.append(item)
    print(json.dumps(item, ensure_ascii=False))

if not all(item['success'] for item in summary):
    raise SystemExit(1)
