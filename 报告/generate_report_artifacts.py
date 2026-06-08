from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "DriverFatigueDetection-v1"
REPORT_DIR = ROOT / "报告"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = V1 / "cache"
CONFIG_PATH = V1 / "config" / "model_config.json"
FEATURE_SCHEMA_PATH = V1 / "src" / "feature_schema.py"

CLASS_NAMES = ["notdrowsy", "drowsy"]


def load_cache(split: str) -> dict:
    path = CACHE_DIR / f"features_{split}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached features: {path}")
    return joblib.load(path)


def build_model(config: dict, y_train: np.ndarray) -> Pipeline:
    model_config = config["model"]
    early_stopping = bool(model_config.get("early_stopping", True))
    unique, counts = np.unique(y_train, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        early_stopping = False

    classifier = HistGradientBoostingClassifier(
        max_iter=int(model_config.get("max_iter", 250)),
        learning_rate=float(model_config.get("learning_rate", 0.05)),
        max_leaf_nodes=int(model_config.get("max_leaf_nodes", 15)),
        l2_regularization=float(model_config.get("l2_regularization", 0.01)),
        early_stopping=early_stopping,
        validation_fraction=float(model_config.get("validation_fraction", 0.15)),
        random_state=int(model_config.get("random_state", 42)),
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])


def evaluate(model: Pipeline, data: dict) -> dict:
    x = data["x"]
    y = data["y"]
    pred = model.predict(x)
    prob = model.predict_proba(x)[:, 1]
    return {
        "samples_total": int(data["samples_total"]),
        "samples_used": int(data["samples_used"]),
        "skipped_no_face_count": int(len(data["skipped_no_face"])),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y,
            pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }


def autolabel(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_metric_comparison(metrics: dict) -> str:
    fig_path = FIG_DIR / "metric_comparison.png"
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc"]
    labels = ["Accuracy", "Balanced Acc", "Macro F1", "Weighted F1", "ROC AUC"]
    splits = ["train", "val", "test"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for i, split in enumerate(splits):
        values = [metrics[split][key] for key in keys]
        bars = ax.bar(x + (i - 1) * width, values, width, label=split)
        autolabel(ax, bars)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Train / Validation / Test Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return str(fig_path)


def save_confusion_matrix(metrics: dict) -> str:
    fig_path = FIG_DIR / "test_confusion_matrix.png"
    cm = np.asarray(metrics["test"]["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return str(fig_path)


def save_class_report(metrics: dict) -> str:
    fig_path = FIG_DIR / "test_class_metrics.png"
    report = metrics["test"]["classification_report"]
    class_metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    for i, cls in enumerate(CLASS_NAMES):
        values = [report[cls][m] for m in class_metrics]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=cls)
        autolabel(ax, bars)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Test Precision / Recall / F1 by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-score"])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return str(fig_path)


def save_roc_curve(model: Pipeline, test_data: dict) -> str:
    fig_path = FIG_DIR / "test_roc_curve.png"
    y = test_data["y"]
    prob = model.predict_proba(test_data["x"])[:, 1]
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, label=f"HGB ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return str(fig_path)


def save_dataset_distribution(data_by_split: dict) -> str:
    fig_path = FIG_DIR / "dataset_distribution.png"
    splits = ["train", "val", "test"]
    counts = {split: np.bincount(data_by_split[split]["y"], minlength=2) for split in splits}
    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars0 = ax.bar(x - width / 2, [counts[s][0] for s in splits], width, label="notdrowsy")
    bars1 = ax.bar(x + width / 2, [counts[s][1] for s in splits], width, label="drowsy")
    autolabel(ax, bars0, fmt="{:.0f}")
    autolabel(ax, bars1, fmt="{:.0f}")
    ax.set_ylabel("Images with detected face")
    ax.set_title("Used Sample Distribution after Face Detection")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return str(fig_path)


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    train_data = load_cache("train")
    val_data = load_cache("val")
    test_data = load_cache("test")
    data_by_split = {"train": train_data, "val": val_data, "test": test_data}

    model = build_model(config, train_data["y"])
    sample_weight = compute_sample_weight(class_weight="balanced", y=train_data["y"])
    model.fit(train_data["x"], train_data["y"], classifier__sample_weight=sample_weight)

    metrics = {split: evaluate(model, data_by_split[split]) for split in ["train", "val", "test"]}

    classifier = model.named_steps["classifier"]
    result = {
        "model_type": "hist_gradient_boosting",
        "config": config,
        "feature_names": train_data["feature_names"],
        "feature_schema_version": train_data["schema_version"],
        "n_features": int(train_data["x"].shape[1]),
        "classes": {"0": "notdrowsy", "1": "drowsy"},
        "trained_from_cached_features": True,
        "cache_files": {
            "train": str(CACHE_DIR / "features_train.joblib"),
            "val": str(CACHE_DIR / "features_val.joblib"),
            "test": str(CACHE_DIR / "features_test.joblib"),
        },
        "classifier_actual": {
            "n_iter_": int(getattr(classifier, "n_iter_", -1)),
            "do_early_stopping_": bool(getattr(classifier, "do_early_stopping_", False)),
        },
        "metrics": metrics,
    }

    report_metrics_path = REPORT_DIR / "report_metrics.json"
    with open(report_metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    joblib.dump(model, REPORT_DIR / "hgb_fatigue_report.joblib")

    figures = {
        "dataset_distribution": save_dataset_distribution(data_by_split),
        "metric_comparison": save_metric_comparison(metrics),
        "test_confusion_matrix": save_confusion_matrix(metrics),
        "test_class_metrics": save_class_report(metrics),
        "test_roc_curve": save_roc_curve(model, test_data),
    }
    with open(REPORT_DIR / "report_figures.json", "w", encoding="utf-8") as f:
        json.dump(figures, f, ensure_ascii=False, indent=2)

    print(json.dumps({"metrics": result["metrics"], "figures": figures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
