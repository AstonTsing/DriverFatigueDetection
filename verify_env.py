import importlib.util
mods = ['PyQt5', 'cv2', 'numpy', 'sklearn', 'joblib', 'mediapipe', 'ultralytics', 'tensorflow', 'torch']
missing = []
for mod in mods:
    ok = importlib.util.find_spec(mod) is not None
    print(f'{mod}: {"OK" if ok else "MISSING"}')
    if not ok:
        missing.append(mod)
raise SystemExit(1 if missing else 0)
