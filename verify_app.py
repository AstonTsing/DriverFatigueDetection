from pathlib import Path
import predictor

base = Path(__file__).resolve().parent
print('detection.exe:', (base / 'detection.exe').exists())
methods = predictor.get_available_methods()
print('methods:', len(methods))
for method in methods:
    print('-', method)
assert (base / 'detection.exe').exists()
assert len(methods) >= 5
