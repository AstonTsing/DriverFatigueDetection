from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib

from src.dataset import LABEL_TO_CLASS
from src.landmark_features import FaceLandmarkFeatureExtractor, read_image
from src.rule_detector import RuleConfig, RuleFatigueDetector


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict driver fatigue for a single image.")
    parser.add_argument("image", help="Image path.")
    parser.add_argument("--config", default="config/rule_config.json", help="Rule/config JSON path.")
    parser.add_argument("--model", default=None, help="Optional SVM model path. If omitted, use rule detector.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = RuleConfig.from_json(args.config)
    model: Optional[object] = joblib.load(args.model) if args.model else None

    image = read_image(args.image)
    with FaceLandmarkFeatureExtractor(
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    ) as extractor:
        features = extractor.extract(image)

    if model is not None and features.face_found:
        pred = int(model.predict([[features.ear_left, features.ear_right, features.ear_mean, features.mar]])[0])
        prediction = {"label": LABEL_TO_CLASS[pred], "is_drowsy": bool(pred), "score": None, "reasons": {"model": "svm"}}
    else:
        prediction = RuleFatigueDetector(config).predict(features).to_dict()

    result = {"image": str(Path(args.image)), "features": features.to_dict(), "prediction": prediction}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
