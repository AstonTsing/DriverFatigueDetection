from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib

from src.dataset import LABEL_TO_CLASS
from src.feature_cache import extract_split_features
from src.feature_extractor import RichFeatureExtractor
from src.feature_schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from src.metrics import build_metrics, predict_prob_drowsy


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate v1 gradient-boosted fatigue detector.")
    parser.add_argument("--test-dir", default="../dataset_split/test", help="Test split directory.")
    parser.add_argument("--model", default="models/hgb_fatigue.joblib", help="Model bundle path.")
    parser.add_argument("--config", default="config/model_config.json", help="Config JSON path used for feature extraction.")
    parser.add_argument("--output", default="reports/test_metrics.json", help="Output metrics JSON path.")
    parser.add_argument("--cache-dir", default=None, help="Feature cache directory. Defaults to config cache.dir.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force feature re-extraction even if cache exists.")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature cache for this run.")
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
    bundle = load_model_bundle(args.model)
    model = bundle["pipeline"]
    cache_config = config.get("cache", {})
    cache_dir = args.cache_dir or cache_config.get("dir", "cache")
    use_cache = bool(cache_config.get("enabled", True)) and not args.no_cache

    with RichFeatureExtractor(
        min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
    ) as extractor:
        test_data = extract_split_features(
            args.test_dir,
            extractor,
            desc="Extracting test features",
            cache_dir=cache_dir,
            use_cache=use_cache,
            rebuild_cache=args.rebuild_cache,
        )

    x = test_data["x"]
    y = test_data["y"]
    if len(x) == 0:
        raise RuntimeError("No faces were detected in test images; cannot evaluate model.")

    pred = model.predict(x)
    prob_drowsy = predict_prob_drowsy(model, x)
    metrics = build_metrics(y, pred, prob_drowsy)

    details = []
    for index, item in enumerate(test_data["details"]):
        label = int(y[index])
        pred_label = int(pred[index])
        details.append(
            {
                "image": item["image"],
                "true": LABEL_TO_CLASS[label],
                "pred": LABEL_TO_CLASS[pred_label],
                "prob_drowsy": float(prob_drowsy[index]) if prob_drowsy is not None else None,
                "features": item["features"],
            }
        )

    report: Dict[str, object] = {
        "model_type": bundle.get("model_type", "unknown"),
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "samples_total": int(test_data["samples_total"]),
        "samples_used": int(test_data["samples_used"]),
        "skipped_no_face": test_data["skipped_no_face"],
        "metrics": metrics,
        "details": details,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
