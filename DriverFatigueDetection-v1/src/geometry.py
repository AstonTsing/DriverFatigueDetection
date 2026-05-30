from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) <= 1e-8:
        return default
    return float(numerator / denominator)


def euclidean_distance(p1: Point, p2: Point) -> float:
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def midpoint(points: Sequence[Point]) -> Point:
    if not points:
        return 0.0, 0.0
    arr = np.asarray(points, dtype=np.float32)
    mean = arr.mean(axis=0)
    return float(mean[0]), float(mean[1])


def collect_points(landmarks, indices: Iterable[int], width: int, height: int) -> list[Point]:
    return [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]


def point_from_landmark(landmarks, index: int, width: int, height: int) -> Point:
    return landmarks[index].x * width, landmarks[index].y * height


def eye_aspect_ratio(points: Sequence[Point]) -> float:
    horizontal = euclidean_distance(points[0], points[3])
    vertical_1 = euclidean_distance(points[1], points[5])
    vertical_2 = euclidean_distance(points[2], points[4])
    return safe_divide(vertical_1 + vertical_2, 2.0 * horizontal)


def mouth_aspect_ratio(points: Sequence[Point]) -> float:
    horizontal = euclidean_distance(points[0], points[2])
    vertical = euclidean_distance(points[1], points[3])
    return safe_divide(vertical, horizontal)


def angle_degrees(p1: Point, p2: Point) -> float:
    return float(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
