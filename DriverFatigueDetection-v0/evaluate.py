from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset import LABEL_TO_CLASS, load_labeled_images
from src.landmark_features import FaceLandmarkFeatureExtractor, read_image
from src.progress import progress
from src.rule_detector import RuleConfig, RuleFatigueDetector


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate rule-based or SVM fatigue detector.")
    parser.add_argument("--test-dir", default="../dataset_split/test", help="Test split directory.")
    parser.add_argument("--config", default="config/rule_config.json", help="Rule/config JSON path.")
    parser.add_argument("--model", default=None, help="Optional SVM model path. If omitted, use rule detector.")
    parser.add_argument("--output", default="reports/eval_metrics.json", help="Output metrics JSON path.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = RuleConfig.from_json(args.config)
    samples = load_labeled_images(args.test_dir)
    if not samples:
        raise RuntimeError(f"No test images found in {args.test_dir}")

    model: Optional[object] = joblib.load(args.model) if args.model else None
    rule_detector = RuleFatigueDetector(config)

    y_true: List[int] = []
    y_pred: List[int] = []
    skipped: List[str] = []
    details: List[Dict[str, object]] = []

    with FaceLandmarkFeatureExtractor(
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    ) as extractor:
        for image_path, label in progress(samples, desc="Evaluating", total=len(samples), unit="image"):
            image = read_image(str(image_path))
            features = extractor.extract(image)
            if not features.face_found:
                skipped.append(str(image_path))
                continue

            if model is not None:
                pred = int(model.predict([[features.ear_left, features.ear_right, features.ear_mean, features.mar]])[0])
                prediction = {"label": LABEL_TO_CLASS[pred], "is_drowsy": bool(pred), "score": None, "reasons": {}}
            else:
                rule_pred = rule_detector.predict(features)
                pred = 1 if rule_pred.is_drowsy else 0
                prediction = rule_pred.to_dict()

            y_true.append(label)
            y_pred.append(pred)
            details.append({"image": str(image_path), "true": LABEL_TO_CLASS[label], "pred": LABEL_TO_CLASS[pred], "features": features.to_dict(), "prediction": prediction})

    labels = [0, 1]
    metrics: Dict[str, object] = {
        "mode": "svm" if model is not None else "rule",
        "samples_total": len(samples),
        "samples_used": len(y_true),
        "skipped_no_face": skipped,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if y_true else [[0, 0], [0, 0]],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[LABEL_TO_CLASS[i] for i in labels],
            zero_division=0,
            output_dict=True,
        ) if y_true else {},
        "details": details,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
