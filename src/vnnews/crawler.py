from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx
import orjson
from newsplease import NewsPlease
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed

from .config import (
    ARTICLES_DIR,
    DEFAULT_CONCURRENCY,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    MAX_RETRIES,
    RETRY_WAIT_SECONDS,
    SourceConfig,
)
from .robots import RobotsManager
from .utils import ensure_parent, slugify


logger = logging.getLogger(__name__)


class ArticleCrawler:
    def __init__(
        self,
        sources: Iterable[SourceConfig],
        output_dir: Path = ARTICLES_DIR,
        user_agent: str = DEFAULT_USER_AGENT,
        max_workers: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self._output_dir = output_dir
        self._user_agent = user_agent
        self._max_workers = max_workers
        self._sources = list(sources)
        self._domain_lookup: Dict[str, str] = {}
        for src in self._sources:
            netloc = urlparse(src.base_url).netloc
            self._domain_lookup[netloc] = src.name
            self._domain_lookup[f"www.{netloc}"] = src.name
        self._http_client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
        )
        self._robots = RobotsManager(self._http_client, user_agent=user_agent)

    def close(self) -> None:
        self._http_client.close()

    def __enter__(self) -> "ArticleCrawler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def crawl(
        self,
        urls: Iterable[str],
        since: date,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[date]:
        url_mapping = self._group_urls_by_source(urls)
        processed_dates: List[date] = []
        workers = max_workers or self._max_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for source, source_urls in url_mapping.items():
                for url in source_urls:
                    future = executor.submit(self._process_single, source, url, since)
                    futures[future] = url
            for future in as_completed(futures):
                url = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - logging path
                    logger.error("Error while crawling article: %s", exc, exc_info=True)
                    result = None
                else:
                    if result:
                        processed_dates.append(result)
                finally:
                    if progress_callback:
                        progress_callback(url)
        return processed_dates

    def _group_urls_by_source(self, urls: Iterable[str]) -> Dict[str, List[str]]:
        bucket: Dict[str, List[str]] = {}
        for url in urls:
            url = url.strip()
            if not url:
                continue
            parsed = urlparse(url)
            if not parsed.scheme:
                url = f"https://{url}"
                parsed = urlparse(url)
            source_name = self._resolve_source_name(parsed.netloc)
            bucket.setdefault(source_name, []).append(url)
        return bucket

    def _resolve_source_name(self, netloc: str) -> str:
        if netloc in self._domain_lookup:
            return self._domain_lookup[netloc]
        stripped = netloc.lstrip("www.")
        for known, name in self._domain_lookup.items():
            if stripped.endswith(known):
                return name
        return stripped.replace(".", "-")

    def _process_single(self, source: str, url: str, since: date) -> Optional[date]:
        if not self._robots.allowed(url):
            logger.debug("Skipping %s due to robots disallow", url)
            return None
        try:
            article = self._fetch_with_retry(url)
        except RetryError as exc:
            logger.warning("Retries exhausted for %s: %s", url, exc)
            return None
        if not article:
            logger.debug("No article content for %s", url)
            return None
        publish_dt = self._extract_date(article)
        if publish_dt and publish_dt < since:
            logger.debug("Article %s older than %s; skipping", url, since.isoformat())
            return None
        text = getattr(article, "maintext", None) or getattr(article, "text", None)
        if not text:
            logger.debug("Article %s missing main text; skipping", url)
            return None

        payload = {
            "source": source,
            "url": url,
            "title": getattr(article, "title", None),
            "authors": getattr(article, "authors", None),
            "date_publish": publish_dt.isoformat() if publish_dt else None,
            "text": text,
            "language": getattr(article, "language", None),
            "top_image": getattr(article, "top_image", None),
            "meta_data": getattr(article, "meta_data", None),
            "retrieved_at": datetime.utcnow().isoformat(),
        }
        path = self._build_article_path(source, publish_dt or date.today(), url, payload["title"])
        ensure_parent(path)
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        return publish_dt or date.today()

    def _build_article_path(self, source: str, publish_date: date, url: str, title: Optional[str]) -> Path:
        day_dir = self._output_dir / source / f"{publish_date.year}" / publish_date.isoformat()
        slug_base = title or url
        filename = f"{slugify(slug_base)}.json"
        full_path = day_dir / filename
        if full_path.exists():
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
            alt = slugify(f"{slug_base}-{digest}")
            full_path = day_dir / f"{alt}.json"
        return full_path

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_fixed(RETRY_WAIT_SECONDS), reraise=True)
    def _fetch_with_retry(self, url: str):
        return NewsPlease.from_url(url)

    @staticmethod
    def _extract_date(article) -> Optional[date]:
        publish_dt = getattr(article, "date_publish", None)
        if publish_dt is None:
            return None
        if isinstance(publish_dt, datetime):
            return publish_dt.date()
        if isinstance(publish_dt, date):
            return publish_dt
        return None
