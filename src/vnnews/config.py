from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List

DEFAULT_USER_AGENT = "vnnews-collector/0.1 (+https://github.com/trader-sentiment)"
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_SINCE_YEARS = 5
DATA_DIR = Path("data")
URL_LIST_DIR = DATA_DIR / "url_list"
ARTICLES_DIR = DATA_DIR / "articles"
STATE_DIR = DATA_DIR / "state"
STATE_PATH = STATE_DIR / "last_run.json"
COLLECT_STATE_PATH = STATE_DIR / "collect_state.json"
CRAWL_PROGRESS_PATH = STATE_DIR / "crawl_progress.json"
DEFAULT_CONCURRENCY = 6
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 3


@dataclass(frozen=True)
class SourceConfig:
    name: str
    base_url: str


SOURCES: List[SourceConfig] = [
    SourceConfig(name="cafef", base_url="https://cafef.vn"),
    SourceConfig(name="cafebiz", base_url="https://cafebiz.vn"),
    SourceConfig(name="vietstock", base_url="https://vietstock.vn"),
    SourceConfig(name="vneconomy", base_url="https://vneconomy.vn"),
    SourceConfig(name="thesaigontimes", base_url="https://thesaigontimes.vn"),
    SourceConfig(name="diendandoanhnghiep", base_url="https://diendandoanhnghiep.vn"),
    SourceConfig(name="baodautu", base_url="https://baodautu.vn"),
]


def default_since_timedelta(years: int = DEFAULT_SINCE_YEARS) -> timedelta:
    return timedelta(days=years * 365)
