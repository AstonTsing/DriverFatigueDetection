from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_LABEL = {"notdrowsy": 0, "drowsy": 1}
LABEL_TO_CLASS = {0: "notdrowsy", 1: "drowsy"}


def iter_image_files(root: str | Path) -> Iterable[Path]:
    root_path = Path(root)
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_labeled_images(split_dir: str | Path) -> List[Tuple[Path, int]]:
    split_path = Path(split_dir)
    samples: List[Tuple[Path, int]] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = split_path / class_name
        if not class_dir.exists():
            continue
        for image_path in iter_image_files(class_dir):
            samples.append((image_path, label))
    return sorted(samples, key=lambda item: str(item[0]))
