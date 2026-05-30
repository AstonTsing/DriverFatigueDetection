from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def read_image(image_path: str) -> Optional[np.ndarray]:
    return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def image_quality_features(image_bgr: np.ndarray) -> dict[str, float]:
    if image_bgr is None or image_bgr.size == 0:
        return {"brightness": 0.0, "contrast": 0.0, "blur_score": 0.0}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 255.0
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"brightness": brightness, "contrast": contrast, "blur_score": blur_score}
