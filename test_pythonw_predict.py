from pathlib import Path
import json
import predictor

base = Path(__file__).resolve().parent
img = next((base / 'DriverFatigueDetection' / 'dataset_split' / 'test' / 'drowsy').rglob('*.jpg'))
result = predictor.predict_with_method('DriverFatigueDetection-v0 (SVM)', img)
(base / 'pythonw_predict_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
