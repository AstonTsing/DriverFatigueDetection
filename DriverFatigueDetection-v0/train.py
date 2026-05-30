from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.dataset import LABEL_TO_CLASS, load_labeled_images
from src.landmark_features import FaceLandmarkFeatureExtractor, read_image
from src.progress import progress, progress_bar
from src.rule_detector import RuleConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a classical SVM classifier on EAR/MAR features.")
    parser.add_argument("--train-dir", default="../dataset_split/train", help="Training split directory.")
    parser.add_argument("--val-dir", default="../dataset_split/val", help="Validation split directory.")
    parser.add_argument("--config", default="config/rule_config.json", help="Rule/config JSON path.")
    parser.add_argument("--output", default="models/svm_ear_mar.joblib", help="Output model path.")
    parser.add_argument("--metrics", default="reports/train_metrics.json", help="Output metrics JSON path.")
    return parser


def extract_matrix(samples, extractor: FaceLandmarkFeatureExtractor, desc: str):
    x: List[List[float]] = []
    y: List[int] = []
    skipped: List[str] = []
    for image_path, label in progress(samples, desc=desc, total=len(samples), unit="image"):
        image = read_image(str(image_path))
        features = extractor.extract(image)
        if not features.face_found:
            skipped.append(str(image_path))
            continue
        x.append([features.ear_left, features.ear_right, features.ear_mean, features.mar])
        y.append(label)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), skipped


def evaluate(model: Pipeline, x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    pred = model.predict(x)
    labels = [0, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y,
            pred,
            labels=labels,
            target_names=[LABEL_TO_CLASS[i] for i in labels],
            zero_division=0,
            output_dict=True,
        ),
    }


def main() -> None:
    args = build_argparser().parse_args()
    config = RuleConfig.from_json(args.config)

    train_samples = load_labeled_images(args.train_dir)
    val_samples = load_labeled_images(args.val_dir)
    if not train_samples:
        raise RuntimeError(f"No training images found in {args.train_dir}")

    with FaceLandmarkFeatureExtractor(
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    ) as extractor:
        x_train, y_train, skipped_train = extract_matrix(train_samples, extractor, "Extracting train features")
        x_val, y_val, skipped_val = extract_matrix(val_samples, extractor, "Extracting val features")

    if len(x_train) == 0:
        raise RuntimeError("No faces were detected in training images; cannot train model.")

    scaler = StandardScaler()
    classifier = SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced", probability=True)
    with progress_bar(desc="Training SVM", total=3, unit="stage") as bar:
        x_train_scaled = scaler.fit_transform(x_train)
        bar.update(1)
        classifier.fit(x_train_scaled, y_train)
        bar.update(1)
        model = Pipeline(steps=[("scaler", scaler), ("classifier", classifier)])
        bar.update(1)

    metrics: Dict[str, object] = {
        "train_samples_total": len(train_samples),
        "train_samples_used": int(len(x_train)),
        "train_skipped_no_face": skipped_train,
        "val_samples_total": len(val_samples),
        "val_samples_used": int(len(x_val)),
        "val_skipped_no_face": skipped_val,
        "feature_order": ["ear_left", "ear_right", "ear_mean", "mar"],
        "classes": LABEL_TO_CLASS,
    }
    metrics["train"] = evaluate(model, x_train, y_train)
    if len(x_val) > 0:
        metrics["val"] = evaluate(model, x_val, y_val)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved model to {output_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
