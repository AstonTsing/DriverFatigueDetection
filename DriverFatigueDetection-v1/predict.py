from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib

from src.dataset import LABEL_TO_CLASS
from src.feature_extractor import RichFeatureExtractor
from src.feature_schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from src.image_io import read_image
from src.metrics import predict_prob_drowsy


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict driver fatigue with v1 gradient-boosted model.")
    parser.add_argument("image", help="Image path.")
    parser.add_argument("--model", default="models/hgb_fatigue.joblib", help="Model bundle path.")
    parser.add_argument("--config", default="config/model_config.json", help="Config JSON path used for feature extraction.")
    return parser


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_bundle(path: str | Path) -> Dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(f"Model schema {bundle.get('schema_version')} does not match current {FEATURE_SCHEMA_VERSION}")
    if bundle.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("Model feature names do not match current FEATURE_NAMES")
    return bundle


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    image = read_image(args.image)

    with RichFeatureExtractor(
        min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
    ) as extractor:
        result = extractor.extract(image)

    if not result.face_found:
        print(json.dumps({"image": str(Path(args.image)), "face_found": False, "prediction": {"label": "unknown"}}, ensure_ascii=False, indent=2))
        return

    bundle = load_model_bundle(args.model)
    model = bundle["pipeline"]
    x = [result.feature_vector()]
    pred = int(model.predict(x)[0])
    prob_drowsy = predict_prob_drowsy(model, x)
    class_probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        class_probabilities = {LABEL_TO_CLASS[index]: float(value) for index, value in enumerate(proba)}

    output = {
        "image": str(Path(args.image)),
        "face_found": True,
        "features": result.features,
        "prediction": {
            "label": LABEL_TO_CLASS[pred],
            "is_drowsy": bool(pred),
            "prob_drowsy": float(prob_drowsy[0]) if prob_drowsy is not None else None,
            "class_probabilities": class_probabilities,
            "model_type": bundle.get("model_type", "unknown"),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
