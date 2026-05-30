from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .landmark_features import FatigueFeatures


@dataclass
class RuleConfig:
    ear_threshold: float = 0.23
    mar_threshold: float = 0.62
    eye_closed_score_weight: float = 1.0
    mouth_open_score_weight: float = 0.55
    fatigue_score_threshold: float = 1.0
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    @classmethod
    def from_json(cls, path: str | Path) -> "RuleConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class RulePrediction:
    label: str
    is_drowsy: bool
    score: float
    reasons: Dict[str, bool]

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "is_drowsy": self.is_drowsy,
            "score": self.score,
            "reasons": self.reasons,
        }


class RuleFatigueDetector:
    """Binary classifier based on EAR and MAR thresholds."""

    def __init__(self, config: RuleConfig) -> None:
        self.config = config

    def predict(self, features: FatigueFeatures) -> RulePrediction:
        if not features.face_found:
            return RulePrediction(
                label="unknown",
                is_drowsy=False,
                score=0.0,
                reasons={"face_found": False, "eye_closed": False, "mouth_open": False},
            )

        eye_closed = features.ear_mean < self.config.ear_threshold
        mouth_open = features.mar > self.config.mar_threshold

        score = 0.0
        if eye_closed:
            score += self.config.eye_closed_score_weight
        if mouth_open:
            score += self.config.mouth_open_score_weight

        is_drowsy = score >= self.config.fatigue_score_threshold
        return RulePrediction(
            label="drowsy" if is_drowsy else "notdrowsy",
            is_drowsy=is_drowsy,
            score=score,
            reasons={"face_found": True, "eye_closed": eye_closed, "mouth_open": mouth_open},
        )
