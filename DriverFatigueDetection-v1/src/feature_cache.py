from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

from .dataset import load_labeled_images
from .feature_extractor import RichFeatureExtractor
from .feature_schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .image_io import read_image
from .progress import progress


def cache_path_for_split(cache_dir: str | Path, split_dir: str | Path) -> Path:
    split_path = Path(split_dir)
    return Path(cache_dir) / f"features_{split_path.name}.joblib"


def _cache_is_compatible(cache: Dict[str, Any], split_dir: str | Path, samples: List[Tuple[Path, int]]) -> bool:
    return (
        cache.get("schema_version") == FEATURE_SCHEMA_VERSION
        and cache.get("feature_names") == FEATURE_NAMES
        and cache.get("split_dir") == str(Path(split_dir).resolve())
        and cache.get("image_paths") == [str(path) for path, _ in samples]
        and cache.get("labels") == [int(label) for _, label in samples]
    )


def extract_split_features(
    split_dir: str | Path,
    extractor: RichFeatureExtractor,
    *,
    desc: str,
    cache_dir: str | Path = "cache",
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> Dict[str, Any]:
    samples = load_labeled_images(split_dir)
    if not samples:
        raise RuntimeError(f"No images found in {split_dir}")

    cache_file = cache_path_for_split(cache_dir, split_dir)
    if use_cache and not rebuild_cache and cache_file.exists():
        cache = joblib.load(cache_file)
        if _cache_is_compatible(cache, split_dir, samples):
            return cache

    rows: list[list[float]] = []
    labels: list[int] = []
    used_images: list[str] = []
    skipped_no_face: list[str] = []
    details: list[dict[str, object]] = []

    for image_path, label in progress(samples, desc=desc, total=len(samples), unit="image"):
        image = read_image(str(image_path))
        result = extractor.extract(image)
        if not result.face_found:
            skipped_no_face.append(str(image_path))
            continue
        vector = result.feature_vector()
        rows.append(vector)
        labels.append(int(label))
        used_images.append(str(image_path))
        details.append({"image": str(image_path), "label": int(label), "features": result.features})

    data = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "split_dir": str(Path(split_dir).resolve()),
        "feature_names": FEATURE_NAMES,
        "image_paths": [str(path) for path, _ in samples],
        "labels": [int(label) for _, label in samples],
        "used_images": used_images,
        "x": np.asarray(rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "skipped_no_face": skipped_no_face,
        "details": details,
        "samples_total": len(samples),
        "samples_used": len(rows),
    }

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, cache_file)
    return data
