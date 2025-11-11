from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import orjson

from .config import (
    COLLECT_STATE_PATH,
    CRAWL_PROGRESS_PATH,
    STATE_PATH,
    default_since_timedelta,
    DEFAULT_SINCE_YEARS,
)


@dataclass
class CrawlState:
    last_processed_date: date


class StateManager:
    def __init__(self, path: Path = STATE_PATH) -> None:
        self._path = path

    def load(self) -> Optional[CrawlState]:
        if not self._path.exists():
            return None
        data = orjson.loads(self._path.read_bytes())
        if "last_processed_date" not in data:
            return None
        return CrawlState(last_processed_date=date.fromisoformat(data["last_processed_date"]))

    def save(self, processed_date: date) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"last_processed_date": processed_date.isoformat()}
        self._path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    def reset(self) -> None:
        if self._path.exists():
            self._path.unlink()


class CollectionStateManager:
    def __init__(self, path: Path = COLLECT_STATE_PATH) -> None:
        self._path = path

    def _load_raw(self) -> Dict[str, Dict[str, str]]:
        if not self._path.exists():
            return {}
        return orjson.loads(self._path.read_bytes())

    def get_until(self, source: str) -> Optional[date]:
        data = self._load_raw()
        if source not in data:
            return None
        value = data[source].get("until_date")
        return date.fromisoformat(value) if value else None

    def save_until(self, source: str, until_date: date) -> None:
        data = self._load_raw()
        data.setdefault(source, {})
        data[source]["until_date"] = until_date.isoformat()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def reset(self, sources: Optional[list[str]] = None) -> None:
        if not self._path.exists():
            return
        if not sources:
            self._path.unlink()
            return
        data = self._load_raw()
        for src in sources:
            data.pop(src, None)
        self._path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


class CrawlProgressManager:
    def __init__(self, path: Path = CRAWL_PROGRESS_PATH) -> None:
        self._path = path

    def _load_raw(self) -> Dict[str, int]:
        if not self._path.exists():
            return {}
        return orjson.loads(self._path.read_bytes())

    def get_offset(self, file_path: Path) -> int:
        data = self._load_raw()
        return data.get(str(file_path), 0)

    def save_offset(self, file_path: Path, offset: int) -> None:
        data = self._load_raw()
        data[str(file_path)] = offset
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def reset(self, file_path: Optional[Path] = None) -> None:
        if not self._path.exists():
            return
        if file_path is None:
            self._path.unlink()
            return
        data = self._load_raw()
        data.pop(str(file_path), None)
        self._path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def determine_since_date(
    since_date: Optional[date],
    since_years: int = DEFAULT_SINCE_YEARS,
    resume: bool = True,
    state_manager: Optional[StateManager] = None,
) -> date:
    if since_date:
        return since_date
    if resume and state_manager:
        state = state_manager.load()
        if state:
            return state.last_processed_date + timedelta(days=1)
    baseline = date.today() - default_since_timedelta(since_years)
    return baseline
