from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_model(config: Dict[str, Any], y_train: np.ndarray | None = None) -> Pipeline:
    model_config = config.get("model", {})
    early_stopping = bool(model_config.get("early_stopping", True))
    if y_train is not None:
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
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )
