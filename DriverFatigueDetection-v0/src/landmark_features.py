from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


Point = Tuple[float, float]


@dataclass
class FatigueFeatures:
    """Facial landmark features used by the rule-based detector."""

    ear_left: float
    ear_right: float
    ear_mean: float
    mar: float
    face_found: bool

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "ear_left": self.ear_left,
            "ear_right": self.ear_right,
            "ear_mean": self.ear_mean,
            "mar": self.mar,
            "face_found": self.face_found,
        }


# MediaPipe Face Mesh landmark indices.
# EAR uses six points around each eye: horizontal pair + two vertical pairs.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# MAR uses four lip landmarks: left/right mouth corners + upper/lower lips.
MOUTH = [61, 13, 291, 14]


def euclidean_distance(p1: Point, p2: Point) -> float:
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def eye_aspect_ratio(points: List[Point]) -> float:
    """Compute eye aspect ratio: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)."""
    horizontal = euclidean_distance(points[0], points[3])
    if horizontal <= 1e-6:
        return 0.0
    vertical_1 = euclidean_distance(points[1], points[5])
    vertical_2 = euclidean_distance(points[2], points[4])
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def mouth_aspect_ratio(points: List[Point]) -> float:
    """Compute mouth aspect ratio: ||upper-lower|| / ||left-right||."""
    horizontal = euclidean_distance(points[0], points[2])
    if horizontal <= 1e-6:
        return 0.0
    vertical = euclidean_distance(points[1], points[3])
    return float(vertical / horizontal)


class FaceLandmarkFeatureExtractor:
    """Extract EAR/MAR features from an image using MediaPipe Face Mesh."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> "FaceLandmarkFeatureExtractor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def extract(self, image_bgr: np.ndarray) -> FatigueFeatures:
        if image_bgr is None or image_bgr.size == 0:
            return FatigueFeatures(0.0, 0.0, 0.0, 0.0, False)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(image_rgb)
        if not result.multi_face_landmarks:
            return FatigueFeatures(0.0, 0.0, 0.0, 0.0, False)

        height, width = image_bgr.shape[:2]
        landmarks = result.multi_face_landmarks[0].landmark

        def collect(indices: Iterable[int]) -> List[Point]:
            return [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]

        left_eye_points = collect(LEFT_EYE)
        right_eye_points = collect(RIGHT_EYE)
        mouth_points = collect(MOUTH)

        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        ear_mean = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth_points)

        return FatigueFeatures(left_ear, right_ear, ear_mean, mar, True)


def read_image(image_path: str) -> Optional[np.ndarray]:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image
