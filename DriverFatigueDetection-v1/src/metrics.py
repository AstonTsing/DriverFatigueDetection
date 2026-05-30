from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .dataset import LABEL_TO_CLASS


def predict_prob_drowsy(model, x: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] > 1:
            return proba[:, 1]
    return None


def build_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob_drowsy: Optional[np.ndarray] = None) -> Dict[str, object]:
    labels = [0, 1]
    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if len(y_true) else [[0, 0], [0, 0]],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[LABEL_TO_CLASS[i] for i in labels],
            zero_division=0,
            output_dict=True,
        ) if len(y_true) else {},
    }
    if prob_drowsy is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob_drowsy))
    else:
        metrics["roc_auc"] = None
    return metrics
