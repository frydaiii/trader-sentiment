from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


def slugify(value: str, limit: int = 80) -> str:
    original = value
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    if not value:
        value = hashlib.sha1(original.encode("utf-8")).hexdigest()[:16]
    return value[:limit]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def ensure_extension(value: str, ext: str) -> str:
    return value if value.endswith(ext) else f"{value}{ext}"
