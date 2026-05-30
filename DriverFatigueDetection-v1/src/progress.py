from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")
_PROGRESS_DISABLED_VALUES = {"0", "false", "no", "off"}


def _progress_disabled() -> bool:
    return os.environ.get("DFD_PROGRESS", "").strip().lower() in _PROGRESS_DISABLED_VALUES


def _tqdm_kwargs(desc: str, total: int | None, unit: str) -> dict[str, object]:
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
        "disable": _progress_disabled(),
    }


def progress(iterable: Iterable[T], *, desc: str, total: int | None = None, unit: str = "it") -> Iterator[T]:
    return tqdm(iterable, **_tqdm_kwargs(desc, total, unit))


def progress_bar(*, desc: str, total: int, unit: str = "it") -> tqdm:
    return tqdm(**_tqdm_kwargs(desc, total, unit))
