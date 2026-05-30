from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
T = TypeVar("T")
_PROGRESS_DISABLED_VALUES = {"0", "false", "no", "off"}


def progress_disabled() -> bool:
    return os.environ.get("DFD_PROGRESS", "").strip().lower() in _PROGRESS_DISABLED_VALUES


def tqdm_kwargs(desc: str, total: int | None, unit: str) -> dict[str, object]:
    return {
        "desc": desc,
        "total": total,
        "unit": unit,
        "file": sys.stdout,
        "ascii": True,
        "dynamic_ncols": False,
        "ncols": 80,
        "mininterval": 1.0,
        "maxinterval": 5.0,
        "leave": False,
        "disable": progress_disabled(),
    }


def progress(iterable: Iterable[T], *, desc: str, total: int | None = None, unit: str = "it") -> Iterator[T]:
    return tqdm(iterable, **tqdm_kwargs(desc, total, unit))


def progress_bar(*, desc: str, total: int, unit: str = "it") -> tqdm:
    return tqdm(**tqdm_kwargs(desc, total, unit))


def collect_images_by_class(source_dir: Path) -> dict[str, list[Path]]:
    images_by_class: dict[str, list[Path]] = defaultdict(list)
    class_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())

    for class_dir in progress(class_dirs, desc="Scanning classes", total=len(class_dirs), unit="class"):
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images_by_class[class_dir.name].append(image_path)

    return {label: sorted(paths) for label, paths in images_by_class.items()}


def split_paths(paths: list[Path], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = paths[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_paths = shuffled[:train_count]
    val_paths = shuffled[train_count:train_count + val_count]
    test_paths = shuffled[train_count + val_count:]

    return train_paths, val_paths, test_paths


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"目标文件已存在，避免覆盖：{dst}")
    shutil.copy2(src, dst)


def remove_tree_with_progress(path: Path) -> None:
    entries = sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    with progress_bar(desc="Removing old output", total=len(entries) + 1, unit="path") as bar:
        for entry in entries:
            if entry.is_dir():
                entry.rmdir()
            else:
                entry.unlink()
            bar.update(1)
        path.rmdir()
        bar.update(1)


def copy_class_splits(split_paths: dict[str, list[Path]], source_dir: Path, output_dir: Path, class_name: str) -> None:
    total = sum(len(paths) for paths in split_paths.values())
    with progress_bar(desc=f"Copying {class_name}", total=total, unit="file") as bar:
        for split_name, paths in split_paths.items():
            for src in paths:
                relative_to_class = src.relative_to(source_dir / class_name)
                dst = output_dir / split_name / class_name / relative_to_class
                safe_copy(src, dst)
                bar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="按类别分层划分疲劳检测图像数据集。")
    parser.add_argument("--source", type=Path, default=Path("train"), help="原始数据目录，默认：train")
    parser.add_argument("--output", type=Path, default=Path("dataset_split"), help="输出目录，默认：dataset_split")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例，默认：0.7")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例，默认：0.15")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例，默认：0.15")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认：42")
    parser.add_argument("--overwrite", action="store_true", help="如果输出目录存在，先删除后重新生成")
    args = parser.parse_args()

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"源目录不存在：{source_dir}")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"比例之和必须为 1，目前为：{ratio_sum}")

    if output_dir.exists():
        if args.overwrite:
            remove_tree_with_progress(output_dir)
        else:
            raise FileExistsError(f"输出目录已存在：{output_dir}；如需重建请添加 --overwrite")

    images_by_class = collect_images_by_class(source_dir)
    if not images_by_class:
        raise RuntimeError(f"未在源目录中找到图片：{source_dir}")

    summary: dict[str, dict[str, int]] = {}

    for class_name, paths in images_by_class.items():
        train_paths, val_paths, test_paths = split_paths(paths, args.train_ratio, args.val_ratio, args.seed)

        copy_class_splits(
            {"train": train_paths, "val": val_paths, "test": test_paths},
            source_dir,
            output_dir,
            class_name,
        )

        summary[class_name] = {
            "total": len(paths),
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
        }

    print(f"数据集划分完成，输出目录：{output_dir}")
    print("采用比例：train/val/test = "
          f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}")
    print("类别统计：")
    for class_name, counts in summary.items():
        print(
            f"  {class_name}: total={counts['total']}, "
            f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
        )


if __name__ == "__main__":
    main()
