# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project layout

The repository contains two sibling driver-fatigue detection projects plus a root dataset splitter:

- `DriverFatigueDetection-v0/` — classical baseline using MediaPipe Face Mesh + EAR/MAR features, with rule-threshold and SVM paths.
- `DriverFatigueDetection-v1/` — improved traditional ML project using MediaPipe Face Mesh + richer handcrafted geometry/image-quality features + `HistGradientBoostingClassifier`.
- `split_dataset.py` — root-level script for creating the sibling `dataset_split/` directory.

Both v0 and v1 are non-deep-learning training projects. MediaPipe is used only to extract face landmarks; the trained classifiers are traditional scikit-learn models.

Expected dataset layout is a sibling of both project directories:

```text
dataset_split/
├── train/{drowsy,notdrowsy}/
├── val/{drowsy,notdrowsy}/
└── test/{drowsy,notdrowsy}/
```

Labels are fixed in each project's `src/dataset.py`: `notdrowsy -> 0`, `drowsy -> 1`. The class directories may contain nested subdirectories; image discovery is recursive.

## Environment and dependencies

Use the Conda environment named `driver`:

```bat
conda create -n driver python=3.12
conda activate driver
```

Install dependencies from the project you are working in:

```bat
cd DriverFatigueDetection-v0
pip install -r requirements.txt
```

or:

```bat
cd DriverFatigueDetection-v1
pip install -r requirements.txt
```

`mediapipe` is pinned to `0.10.21` because both projects use the legacy `mp.solutions.face_mesh.FaceMesh` API. Newer MediaPipe versions may not expose `mp.solutions` in this environment.

Set `DFD_PROGRESS=0` to disable tqdm progress bars. Progress helpers use fixed-width ASCII output to avoid Windows terminal progress-bar spam.

## Common commands

Run project-specific commands from `DriverFatigueDetection-v0/` or `DriverFatigueDetection-v1/` unless noted otherwise.

### v0 baseline commands

Rule-threshold evaluation:

```bat
python evaluate.py --test-dir ../dataset_split/test --output reports/rule_eval_metrics.json
```

Train the v0 SVM classifier:

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/svm_ear_mar.joblib --metrics reports/train_metrics.json
```

Evaluate the v0 SVM model:

```bat
python evaluate.py --test-dir ../dataset_split/test --model models/svm_ear_mar.joblib --output reports/svm_eval_metrics.json
```

Predict one image with v0 rule thresholds:

```bat
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg
```

Predict one image with the v0 SVM model:

```bat
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg --model models/svm_ear_mar.joblib
```

### v1 improved traditional ML commands

Train the v1 HistGradientBoosting model:

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/hgb_fatigue.joblib --metrics reports/train_metrics.json
```

Force feature cache rebuild when feature code or dataset contents changed:

```bat
python train.py --train-dir ../dataset_split/train --val-dir ../dataset_split/val --output models/hgb_fatigue.joblib --metrics reports/train_metrics.json --rebuild-cache
```

Evaluate the v1 model:

```bat
python evaluate.py --test-dir ../dataset_split/test --model models/hgb_fatigue.joblib --output reports/test_metrics.json
```

Predict one image with v1:

```bat
python predict.py ../dataset_split/test/drowsy/sleepyCombination/001_glasses_sleepyCombination_1005_drowsy.jpg --model models/hgb_fatigue.joblib
```

Compile-check Python files:

```bat
python -m compileall .
```

There is currently no dedicated unit test suite, lint command, or build step.

### Dataset split commands

Run these from the repository root.

Create `dataset_split/` from an original labeled image directory:

```bat
python split_dataset.py --source train --output dataset_split --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

Recreate an existing split only when intended:

```bat
python split_dataset.py --source train --output dataset_split --overwrite
```

In Windows `cmd`, use `dir`, not `ls`, for path discovery. Example:

```bat
dir ..\dataset_split\test\drowsy /s /b
```

## Architecture notes

### v0 architecture

- `DriverFatigueDetection-v0/src/landmark_features.py` owns image loading and feature extraction. `read_image()` uses `cv2.imdecode(np.fromfile(...))` so Windows paths with Chinese characters work. `FaceLandmarkFeatureExtractor` returns `FatigueFeatures` with `face_found`, EAR values, and MAR.
- `DriverFatigueDetection-v0/src/rule_detector.py` defines `RuleConfig`, loads `config/rule_config.json`, and implements threshold scoring. Missing faces produce an `unknown` rule prediction; train/eval scripts skip no-face images before scoring.
- `DriverFatigueDetection-v0/evaluate.py` is shared by rule and SVM evaluation. Without `--model`, it uses `RuleFatigueDetector`; with `--model`, it loads the Joblib SVM and predicts with feature order `[ear_left, ear_right, ear_mean, mar]`.
- `DriverFatigueDetection-v0/train.py` extracts the same four features from train/val splits, skips no-face images, trains `StandardScaler + SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced", probability=True)`, and writes model/metrics outputs.
- `DriverFatigueDetection-v0/predict.py` uses the SVM only when `--model` is supplied and a face is found; otherwise it falls back to rule-based prediction.

### v1 architecture

- `DriverFatigueDetection-v1/src/feature_schema.py` defines the fixed v1 feature order and `FEATURE_SCHEMA_VERSION`. Keep train/evaluate/predict/cache aligned with this list.
- `DriverFatigueDetection-v1/src/image_io.py` keeps Windows-safe image loading and computes image-quality features: brightness, contrast, blur score.
- `DriverFatigueDetection-v1/src/geometry.py` contains distance, angle, midpoint, EAR/MAR, and safe division helpers.
- `DriverFatigueDetection-v1/src/feature_extractor.py` wraps MediaPipe Face Mesh and returns `FeatureResult`. It expands v0's EAR/MAR into 35 handcrafted features covering eye openness/asymmetry, mouth opening, face geometry, head-pose proxies, brow-eye distance, and image quality. It intentionally does not use filename/path-derived scenario labels as model features.
- `DriverFatigueDetection-v1/src/feature_cache.py` caches extracted split features as Joblib files under `cache/`. Cache compatibility checks schema version, feature names, split path, image paths, and labels. Use `--rebuild-cache` after feature schema changes or dataset changes.
- `DriverFatigueDetection-v1/src/model_factory.py` builds `SimpleImputer(strategy="median") + HistGradientBoostingClassifier`. Training passes balanced sample weights via `compute_sample_weight`; early stopping is disabled automatically for tiny smoke datasets with fewer than two samples per class.
- `DriverFatigueDetection-v1/train.py` saves a Joblib bundle, not just the sklearn pipeline. The bundle includes `model_type`, `pipeline`, `feature_names`, `classes`, `config`, and `schema_version` to prevent feature-order mismatches.
- `DriverFatigueDetection-v1/evaluate.py` verifies schema compatibility before evaluation and writes aggregate metrics plus per-image details.
- `DriverFatigueDetection-v1/predict.py` verifies schema compatibility before predicting one image and returns JSON with features, label, `prob_drowsy`, and class probabilities. Missing faces return an `unknown` prediction.

## Current measured baselines

On the current `dataset_split/test` with 9,980 images total and 9,888 images used after no-face skips:

- v0 rule threshold: accuracy about `0.5974`.
- v0 SVM: accuracy about `0.6872`.
- v1 HistGradientBoosting: accuracy about `0.9239`, ROC AUC about `0.9794`.

These results are from the existing reports in `DriverFatigueDetection-v0/reports/` and `DriverFatigueDetection-v1/reports/`. Interpret them with caution: if nearby frames or the same driver/video appear across train/val/test, accuracy may be optimistic.
