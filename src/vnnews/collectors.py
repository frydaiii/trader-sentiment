from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Sequence

from .config import URL_LIST_DIR
from .config import SourceConfig
from .sitemap_client import SitemapClient


logger = logging.getLogger(__name__)


@dataclass
class CollectionResult:
    source: SourceConfig
    urls: List[str]
    url_entries: List[tuple[str, date | None]]
    earliest_date: date | None
    latest_date: date | None
    sitemaps_fetched: int


class ArticleURLCollector:
    def __init__(self, sitemap_client: SitemapClient) -> None:
        self._client = sitemap_client

    def collect_for_source(
        self,
        source: SourceConfig,
        since: date,
        until: date | None = None,
        max_sitemaps: int | None = None,
    ) -> CollectionResult:
        until_label = until.isoformat() if until else "today"
        logger.info(
            "Collecting URLs for %s since %s (through %s)",
            source.name,
            since.isoformat(),
            until_label,
        )
        dedup: Dict[str, datetime | None] = {}
        count_total = 0
        earliest_date = None
        latest_date = None
        for entry in self._client.iter_urls(source.base_url, since=since, max_documents=max_sitemaps):
            count_total += 1
            if not entry.loc:
                continue
            lastmod = entry.lastmod
            if lastmod:
                entry_date = lastmod.date()
                if entry_date < since:
                    continue
                if until and entry_date > until:
                    continue
                earliest_date = entry_date if not earliest_date or entry_date < earliest_date else earliest_date
                latest_date = entry_date if not latest_date or entry_date > latest_date else latest_date
            dedup[entry.loc] = lastmod

        sorted_entries = list(sorted(dedup.items(), key=self._sort_key, reverse=True))
        sorted_urls = [url for url, _ in sorted_entries]
        url_entries = [(url, lastmod.date() if lastmod else None) for url, lastmod in sorted_entries]
        logger.info(
            "Source %s -> %d urls collected (from %d sitemap entries, %d sitemaps fetched)",
            source.name,
            len(sorted_urls),
            count_total,
            self._client.last_stats().documents_fetched,
        )
        return CollectionResult(
            source=source,
            urls=sorted_urls,
            url_entries=url_entries,
            earliest_date=earliest_date,
            latest_date=latest_date,
            sitemaps_fetched=self._client.last_stats().documents_fetched,
        )

    @staticmethod
    def _sort_key(item: tuple[str, datetime | None]) -> datetime:
        _, lastmod = item
        return lastmod or datetime.min


def write_url_lists(results: Sequence[CollectionResult], timestamp_slug: str) -> Dict[str, Path]:
    URL_LIST_DIR.mkdir(parents=True, exist_ok=True)
    per_source: Dict[tuple[str, str], List[str]] = defaultdict(list)
    combined: Dict[str, List[str]] = defaultdict(list)
    for result in results:
        if not result.url_entries:
            continue
        for url, entry_date in result.url_entries:
            slug = entry_date.strftime("%Y%m%d") if entry_date else timestamp_slug
            key = (result.source.name, slug)
            per_source[key].append(url)
            combined[slug].append(url)

    if not combined:
        logger.warning("No URLs collected for any source; combined lists not created")
        return {}

    combined_paths: Dict[str, Path] = {}
    for (source_name, slug), urls in per_source.items():
        day_dir = URL_LIST_DIR / slug
        day_dir.mkdir(parents=True, exist_ok=True)
        deduped_urls = list(dict.fromkeys(urls))
        source_path = day_dir / f"{source_name}.txt"
        source_path.write_text("\n".join(deduped_urls), encoding="utf-8")
        logger.info("Saved %d urls to %s", len(deduped_urls), source_path)

    for slug, urls in combined.items():
        day_dir = URL_LIST_DIR / slug
        deduped_combined = list(dict.fromkeys(urls))
        combined_path = day_dir / "all.txt"
        combined_path.write_text("\n".join(deduped_combined), encoding="utf-8")
        logger.info("Saved %d combined urls to %s", len(deduped_combined), combined_path)
        combined_paths[slug] = combined_path

    return combined_paths
