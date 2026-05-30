from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np

from .feature_schema import (
    CHIN,
    FACE_LEFT,
    FACE_RIGHT,
    FEATURE_NAMES,
    FOREHEAD,
    LEFT_BROW,
    LEFT_EYE,
    LEFT_EYE_CORNERS,
    MOUTH,
    MOUTH_CORNERS,
    MOUTH_VERTICAL,
    NOSE_TIP,
    RIGHT_BROW,
    RIGHT_EYE,
    RIGHT_EYE_CORNERS,
)
from .geometry import (
    angle_degrees,
    collect_points,
    euclidean_distance,
    eye_aspect_ratio,
    midpoint,
    mouth_aspect_ratio,
    point_from_landmark,
    safe_divide,
)
from .image_io import image_quality_features


@dataclass
class FeatureResult:
    face_found: bool
    features: Dict[str, float]

    def feature_vector(self) -> list[float]:
        return [float(self.features[name]) for name in FEATURE_NAMES]


class RichFeatureExtractor:
    """Extract handcrafted facial geometry features for traditional ML models."""

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

    def __enter__(self) -> "RichFeatureExtractor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def extract(self, image_bgr: np.ndarray) -> FeatureResult:
        defaults = {name: 0.0 for name in FEATURE_NAMES}
        if image_bgr is None or image_bgr.size == 0:
            return FeatureResult(False, defaults)

        quality = image_quality_features(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(image_rgb)
        if not result.multi_face_landmarks:
            defaults.update(quality)
            return FeatureResult(False, defaults)

        height, width = image_bgr.shape[:2]
        landmarks = result.multi_face_landmarks[0].landmark

        left_eye = collect_points(landmarks, LEFT_EYE, width, height)
        right_eye = collect_points(landmarks, RIGHT_EYE, width, height)
        mouth = collect_points(landmarks, MOUTH, width, height)
        left_eye_corners = collect_points(landmarks, LEFT_EYE_CORNERS, width, height)
        right_eye_corners = collect_points(landmarks, RIGHT_EYE_CORNERS, width, height)
        mouth_corners = collect_points(landmarks, MOUTH_CORNERS, width, height)
        mouth_vertical = collect_points(landmarks, MOUTH_VERTICAL, width, height)
        left_brow = collect_points(landmarks, LEFT_BROW, width, height)
        right_brow = collect_points(landmarks, RIGHT_BROW, width, height)

        face_left = point_from_landmark(landmarks, FACE_LEFT, width, height)
        face_right = point_from_landmark(landmarks, FACE_RIGHT, width, height)
        forehead = point_from_landmark(landmarks, FOREHEAD, width, height)
        chin = point_from_landmark(landmarks, CHIN, width, height)
        nose_tip = point_from_landmark(landmarks, NOSE_TIP, width, height)

        face_width = euclidean_distance(face_left, face_right)
        face_height = euclidean_distance(forehead, chin)
        face_center = midpoint([face_left, face_right, forehead, chin])
        face_scale = max(face_width, face_height, 1.0)

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear_mean = (ear_left + ear_right) / 2.0
        mar = mouth_aspect_ratio(mouth)

        left_eye_width = euclidean_distance(left_eye_corners[0], left_eye_corners[1])
        right_eye_width = euclidean_distance(right_eye_corners[0], right_eye_corners[1])
        left_eye_height = (euclidean_distance(left_eye[1], left_eye[5]) + euclidean_distance(left_eye[2], left_eye[4])) / 2.0
        right_eye_height = (euclidean_distance(right_eye[1], right_eye[5]) + euclidean_distance(right_eye[2], right_eye[4])) / 2.0
        mouth_width = euclidean_distance(mouth_corners[0], mouth_corners[1])
        mouth_height = euclidean_distance(mouth_vertical[0], mouth_vertical[1])

        left_eye_center = midpoint(left_eye_corners)
        right_eye_center = midpoint(right_eye_corners)
        mouth_center = midpoint(mouth_corners)
        left_face_width = euclidean_distance(face_left, nose_tip)
        right_face_width = euclidean_distance(nose_tip, face_right)
        left_brow_center = midpoint(left_brow)
        right_brow_center = midpoint(right_brow)
        left_brow_eye = abs(left_brow_center[1] - left_eye_center[1])
        right_brow_eye = abs(right_brow_center[1] - right_eye_center[1])

        features = {
            "ear_left": ear_left,
            "ear_right": ear_right,
            "ear_mean": ear_mean,
            "mar": mar,
            "ear_min": min(ear_left, ear_right),
            "ear_max": max(ear_left, ear_right),
            "ear_diff_abs": abs(ear_left - ear_right),
            "ear_ratio_left_right": safe_divide(ear_left, ear_right),
            "left_eye_width_norm": safe_divide(left_eye_width, face_width),
            "right_eye_width_norm": safe_divide(right_eye_width, face_width),
            "left_eye_height_norm": safe_divide(left_eye_height, face_height),
            "right_eye_height_norm": safe_divide(right_eye_height, face_height),
            "mouth_width_norm": safe_divide(mouth_width, face_width),
            "mouth_height_norm": safe_divide(mouth_height, face_height),
            "mouth_open_area_proxy": safe_divide(mouth_width * mouth_height, face_scale * face_scale),
            "mouth_to_face_width": safe_divide(mouth_width, face_width),
            "mar_to_ear_mean": safe_divide(mar, ear_mean),
            "mar_minus_ear_mean": mar - ear_mean,
            "face_width": safe_divide(face_width, width),
            "face_height": safe_divide(face_height, height),
            "face_aspect_ratio": safe_divide(face_width, face_height),
            "nose_x_norm": safe_divide(nose_tip[0] - face_center[0], face_width),
            "nose_y_norm": safe_divide(nose_tip[1] - face_center[1], face_height),
            "nose_to_chin_norm": safe_divide(euclidean_distance(nose_tip, chin), face_height),
            "forehead_to_chin_norm": safe_divide(face_height, height),
            "left_right_face_width_ratio": safe_divide(left_face_width, right_face_width),
            "eye_line_angle": angle_degrees(left_eye_center, right_eye_center),
            "mouth_line_angle": angle_degrees(mouth_corners[0], mouth_corners[1]),
            "left_brow_eye_distance_norm": safe_divide(left_brow_eye, face_height),
            "right_brow_eye_distance_norm": safe_divide(right_brow_eye, face_height),
            "brow_eye_distance_mean": safe_divide((left_brow_eye + right_brow_eye) / 2.0, face_height),
            "brow_eye_distance_diff_abs": safe_divide(abs(left_brow_eye - right_brow_eye), face_height),
        }
        features.update(quality)

        for name in FEATURE_NAMES:
            value = features.get(name, 0.0)
            if not np.isfinite(value):
                value = 0.0
            defaults[name] = float(value)
        return FeatureResult(True, defaults)
