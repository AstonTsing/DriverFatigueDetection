# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This workspace is a collection of related driver drowsiness/fatigue detection projects plus a finished Windows GUI wrapper. All projects are binary classifiers for `drowsy` vs `notdrowsy`, but they use different model families and runtime assumptions.

- Root GUI app: `detection_gui.py`, `prediction_server.py`, and `predictor.py` provide the current finished deliverable. `detection.exe` is a lightweight launcher that uses the bundled `python_env/`; keep the whole folder together for portable distribution.
- `DriverFatigueDetection/`: traditional ML pipeline using MediaPipe Face Mesh landmarks and scikit-learn classifiers. See `DriverFatigueDetection/CLAUDE.md` for detailed v0/v1 guidance.
- `drowsiness_detection/`: YOLOv11 image classification training project. It is classification, not bounding-box detection.
- `Introduction_to_AI-Project_chenbh/`: TensorFlow/Keras notebook and model artifacts for baseline CNN and transfer-learning models. The historical README describes VGG16, while the current GUI also uses a MobileNetV2 weights file.

## Common commands

Run these from the repository root unless a `cd` is shown.

### Root GUI / portable app

```bat
python_env\python.exe detection_gui.py
run_gui.bat
```

The GUI starts a persistent `prediction_server.py` subprocess, preloads available models, and sends JSON-line requests over stdin/stdout. To inspect available methods without opening the GUI:

```bat
python_env\python.exe predictor.py
```

Set up or rebuild the portable embedded Python environment:

```bat
setup_portable.bat
python_env\python.exe -m pip install -r requirements_gui.txt
```

For a smaller environment that only supports the traditional ML methods:

```bat
python_env\python.exe -m pip install -r requirements_minimal.txt
```

Package the launcher with PyInstaller:

```bat
build_exe.bat
```

### Root verification scripts

There is no formal pytest suite. Use the checked-in verification scripts as targeted smoke tests:

```bat
python_env\python.exe verify_env.py
python_env\python.exe verify_app.py
python_env\python.exe verify_all_methods.py
python_env\python.exe test_prediction_server.py
```

Run a single verification script by invoking just that file, for example:

```bat
python_env\python.exe test_prediction_server.py
```

For syntax checks after Python edits:

```bat
python_env\python.exe -m compileall detection_gui.py predictor.py prediction_server.py
```

### DriverFatigueDetection traditional ML

Create the documented environment and install the version-specific requirements:

```bat
conda create -n driver python=3.12
conda activate driver
cd DriverFatigueDetection\DriverFatigueDetection-v1
pip install -r requirements.txt
```

Dataset split commands run from `DriverFatigueDetection/`:

```bat
python split_dataset.py --source train --output dataset_split --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
python split_dataset.py --source train --output dataset_split --overwrite
```

v0 commands run from `DriverFatigueDetection/DriverFatigueDetection-v0/`:

```bat
python evaluate.py --test-dir ../dataset_split/test --output reports/rule_eval_metrics.json
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/svm_ear_mar.joblib --metrics reports/train_metrics.json
python evaluate.py --test-dir ../dataset_split/test --model models/svm_ear_mar.joblib --output reports/svm_eval_metrics.json
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg --model models/svm_ear_mar.joblib
```

v1 commands run from `DriverFatigueDetection/DriverFatigueDetection-v1/`:

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/hgb_fatigue.joblib --metrics reports/train_metrics.json
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/hgb_fatigue.joblib --metrics reports/train_metrics.json --rebuild-cache
python evaluate.py --test-dir ../dataset_split/test --model models/hgb_fatigue.joblib --output reports/test_metrics.json
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg --model models/hgb_fatigue.joblib
```

### YOLOv11 project

Commands run from `drowsiness_detection/`:

```bash
bash setup_env.sh
conda env create -f environment.yml
conda activate dd
bash run_train.sh
TASK_NAME=exp_50ep EPOCHS=50 BATCH_SIZE=16 EVAL_FREQ=2 NUM_EVAL=200 DEVICE=0 bash run_train.sh
```

The default documented dataset path is `/home/hebu/dd_dataset`, and the best checkpoint is saved to `outputs/<task_name>/best.pt`.

### TensorFlow/Keras project

Commands run from `Introduction_to_AI-Project_chenbh/`:

```bat
conda env create -f environment-gpu.yml
conda activate drowsiness-gpu
python -m ipykernel install --user --name drowsiness-gpu --display-name "Python (drowsiness-gpu GPU)"
python train_mobilenet.py
jupyter notebook Drowsiness_Detection.ipynb
```

See `GPU_SETUP_CN.md` for the documented Windows/WSL GPU constraints.

## Architecture notes

### Root GUI integration

`detection_gui.py` is the user-facing PyQt5 application. It sets `QT_QPA_PLATFORM_PLUGIN_PATH`, `QT_PLUGIN_PATH`, `PATH`, and DLL directories so the bundled `python_env/` can load PyQt5, OpenCV, MediaPipe, TensorFlow, Torch, NumPy, SciPy, and scikit-learn DLLs from Chinese or space-containing paths.

The GUI uses `PredictionServerClient` to keep one `prediction_server.py` process alive. `PreloadThread` sends `{"cmd": "preload"}` at startup; `PredictionThread` sends either `predict` for one method or `compare` for all selected methods. This avoids reloading large models for each button click.

`prediction_server.py` is intentionally small: it imports `predictor`, reads one JSON request per line, calls `preload_all()`, `predict_with_method()`, or a comparison loop, then writes one JSON response per line.

`predictor.py` is the integration layer. It owns model caching, portable-DLL setup, MediaPipe resource path workarounds, and the `PREDICTORS` mapping that controls which methods appear in the GUI. If adding or removing a model, update both the predictor function and `PREDICTORS`; `get_available_methods()` only exposes entries whose model artifact path exists.

Important predictor details:

- v0 and v1 both contain a package named `src`; `_use_project_src()` clears `src.*` modules and rewrites `sys.path` before importing one project. Do not import both projects' `src` modules globally.
- `_prepare_mediapipe_ascii_resources()` copies MediaPipe resources to `%TEMP%` so native MediaPipe can run from this repository's Chinese path.
- TensorFlow image loading uses `np.fromfile()` + `cv2.imdecode()` to handle Windows paths with Chinese characters.
- MobileNetV2 is reconstructed in code and loads weights from `model_mobilenetv2.h5`; the baseline CNN is loaded as a Keras model from `model_baseline.h5`.
- YOLO automatically selects CUDA device `0` when `torch.cuda.is_available()`.

### Traditional ML project

`DriverFatigueDetection/` expects this dataset layout:

```text
DriverFatigueDetection/dataset_split/
├── train/{drowsy,notdrowsy}/
├── val/{drowsy,notdrowsy}/
└── test/{drowsy,notdrowsy}/
```

Labels are fixed as `notdrowsy -> 0` and `drowsy -> 1`, and image discovery is recursive. `mediapipe==0.10.21` is pinned because the code uses the legacy `mp.solutions.face_mesh.FaceMesh` API.

v0 extracts EAR/MAR features and supports rule-threshold evaluation or an SVM over `[ear_left, ear_right, ear_mean, mar]`. v1 extracts a 35-feature handcrafted vector, caches extracted features, and trains a `SimpleImputer(strategy="median") + HistGradientBoostingClassifier` bundle with schema/version checks.

### Deep learning projects

`drowsiness_detection/train.py` trains YOLOv11 classification from local image folders and writes checkpoints/metrics under `outputs/<task_name>/`. The GUI expects the YOLO checkpoint at `drowsiness_detection/outputs/test1/best.pt`.

`Introduction_to_AI-Project_chenbh/` is notebook-centric. The GUI uses `model_baseline.h5` and `model_mobilenetv2.h5` from this directory; changing their expected input shape or class order requires updating `predictor.py`.

## Practical cautions

- The root README describes the portable GUI as the current finished deliverable; if moving or copying it, preserve `python_env/`, model artifacts, and the project directory structure.
- Avoid searching or editing inside `python_env/`, `.venv/`, generated `outputs/`, `reports/`, `dist/`, and `build/` unless the task is explicitly about environment/package artifacts.
- `DriverFatigueDetection/CLAUDE.md` contains more detailed baseline metrics and internal module notes for v0/v1; consult it before modifying those subprojects.
- No `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` were found at initialization time.
