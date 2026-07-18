# Deep Learning for Safer Roads: YOLOv11 Driver Drowsiness Detection

This project uses a single YOLOv11 pipeline to detect driver drowsiness from a local classified image dataset.

## Goal

Classify driver state into two classes:

- `drowsy`
- `notdrowsy`

## Main File

- `train.py`: terminal training entry. It trains YOLOv11, evaluates every `eval_freq` epochs on a sampled validation subset, and saves only the best checkpoint to `outputs/<task_name>/best.pt`.
- `run_train.sh`: shell launcher with editable training parameters.
- `Deep_Learning_for_Safer_Roads_YOLOv11_Driver_Drowsiness_Detection.ipynb`: notebook containing the full YOLOv11 local training workflow.
- `environment.yml`: conda environment definition.
- `setup_env.sh`: one-command conda environment setup script.
- `doc/screenshots/`: images used by the documentation.
- `LICENSE.txt`: license file.

## Pipeline

1. Install dependencies:
   - `ultralytics==8.3.40`
   - `tensorflow==2.13.1`
   - `matplotlib`
   - `pillow`
   - `opencv-python`

2. Prepare the local dataset at `/home/hebu/dd_dataset`:
   - `train/drowsy`
   - `train/notdrowsy`
   - `val/drowsy`
   - `val/notdrowsy`

3. Preview training images with `glob`, `PIL`, and `matplotlib`.

4. Train or load YOLOv11 classification model:
   - Reuse `runs/classify/train/weights/best.pt` if it exists.
   - Restore `runs_backup.zip` if available.
   - Otherwise train from `YOLO("yolo11n-cls.pt")`.

5. Evaluate predictions on validation images from `drowsy` and `notdrowsy`.

6. View training artifacts:
   - `confusion_matrix.png`
   - `confusion_matrix_normalized.png`
   - `results.png`

7. Backup training results:
   - Compress `runs` to `runs_backup.zip`.

8. Export for mobile:
   - Use `model.export(format="tflite")`.

## Terminal Training

```bash
bash run_train.sh
```

Override parameters from the command line:

```bash
TASK_NAME=exp_50ep EPOCHS=50 BATCH_SIZE=16 EVAL_FREQ=2 NUM_EVAL=200 DEVICE=0 bash run_train.sh
```

The best checkpoint is saved to:

```text
outputs/<task_name>/best.pt
```

## Environment

Run `bash setup_env.sh`, then start Jupyter with the `Python (dd)` kernel.
