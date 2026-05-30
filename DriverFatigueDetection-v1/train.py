from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
from sklearn.utils.class_weight import compute_sample_weight

from src.dataset import LABEL_TO_CLASS
from src.feature_cache import extract_split_features
from src.feature_extractor import RichFeatureExtractor
from src.feature_schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from src.metrics import build_metrics, predict_prob_drowsy
from src.model_factory import build_model
from src.progress import progress_bar


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train v1 gradient-boosted traditional ML fatigue detector.")
    parser.add_argument("--train-dir", default="../dataset_split/train", help="Training split directory.")
    parser.add_argument("--val-dir", default="../dataset_split/val", help="Validation split directory.")
    parser.add_argument("--config", default="config/model_config.json", help="Model/config JSON path.")
    parser.add_argument("--output", default="models/hgb_fatigue.joblib", help="Output model bundle path.")
    parser.add_argument("--metrics", default="reports/train_metrics.json", help="Output metrics JSON path.")
    parser.add_argument("--cache-dir", default=None, help="Feature cache directory. Defaults to config cache.dir.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force feature re-extraction even if cache exists.")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature cache for this run.")
    return parser


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_split(model, split_data: Dict[str, Any]) -> Dict[str, object]:
    x = split_data["x"]
    y = split_data["y"]
    pred = model.predict(x)
    prob_drowsy = predict_prob_drowsy(model, x)
    return build_metrics(y, pred, prob_drowsy)


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    cache_config = config.get("cache", {})
    cache_dir = args.cache_dir or cache_config.get("dir", "cache")
    use_cache = bool(cache_config.get("enabled", True)) and not args.no_cache

    with RichFeatureExtractor(
        min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
    ) as extractor:
        train_data = extract_split_features(
            args.train_dir,
            extractor,
            desc="Extracting train features",
            cache_dir=cache_dir,
            use_cache=use_cache,
            rebuild_cache=args.rebuild_cache,
        )
        val_data = extract_split_features(
            args.val_dir,
            extractor,
            desc="Extracting val features",
            cache_dir=cache_dir,
            use_cache=use_cache,
            rebuild_cache=args.rebuild_cache,
        )

    if len(train_data["x"]) == 0:
        raise RuntimeError("No faces were detected in training images; cannot train model.")

    model = build_model(config, train_data["y"])
    sample_weight = compute_sample_weight(class_weight="balanced", y=train_data["y"])
    with progress_bar(desc="Training HGB model", total=1, unit="stage") as bar:
        model.fit(train_data["x"], train_data["y"], classifier__sample_weight=sample_weight)
        bar.update(1)

    train_metrics = evaluate_split(model, train_data)
    val_metrics = evaluate_split(model, val_data) if len(val_data["x"]) else {}

    model_bundle = {
        "model_type": "hist_gradient_boosting",
        "pipeline": model,
        "feature_names": FEATURE_NAMES,
        "classes": LABEL_TO_CLASS,
        "config": config,
        "schema_version": FEATURE_SCHEMA_VERSION,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, output_path)

    metrics: Dict[str, object] = {
        "model_type": "hist_gradient_boosting",
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "classes": LABEL_TO_CLASS,
        "train_samples_total": int(train_data["samples_total"]),
        "train_samples_used": int(train_data["samples_used"]),
        "train_skipped_no_face": train_data["skipped_no_face"],
        "val_samples_total": int(val_data["samples_total"]),
        "val_samples_used": int(val_data["samples_used"]),
        "val_skipped_no_face": val_data["skipped_no_face"],
        "train": train_metrics,
        "val": val_metrics,
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved model bundle to {output_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
