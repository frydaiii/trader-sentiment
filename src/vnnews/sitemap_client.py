from __future__ import annotations

import gzip
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Generator, Iterable, List, Optional, Set
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import httpx

from .config import DEFAULT_TIMEOUT, DEFAULT_USER_AGENT
from .robots import RobotsManager


logger = logging.getLogger(__name__)


@dataclass
class UrlEntry:
    loc: str
    lastmod: datetime | None


@dataclass
class TraversalStats:
    documents_fetched: int = 0


@dataclass
class _TraversalState:
    max_documents: Optional[int]
    processed_documents: int = 0
    stop: bool = False


class SitemapClient:
    """
    Handles sitemap discovery (via robots.txt) and traversal for nested sitemap indexes.
    """

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
        )
        self._robots = RobotsManager(self._client, user_agent)
        self._user_agent = user_agent
        self._last_stats = TraversalStats()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SitemapClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def iter_urls(
        self,
        base_url: str,
        since: date | None = None,
        max_documents: Optional[int] = None,
    ) -> Generator[UrlEntry, None, None]:
        sitemap_urls = self._robots.get_sitemaps(base_url)
        if not sitemap_urls:
            sitemap_urls = [urljoin(base_url.rstrip("/") + "/", "sitemap.xml")]
            logger.info("No sitemap entries in robots.txt for %s, falling back to %s", base_url, sitemap_urls[0])

        visited: Set[str] = set()
        state = _TraversalState(max_documents=max_documents)
        for sitemap in sitemap_urls:
            if state.stop:
                break
            yield from self._walk_sitemap(sitemap, visited, since, state)
        self._last_stats = TraversalStats(documents_fetched=state.processed_documents)

    def last_stats(self) -> TraversalStats:
        return self._last_stats

    def _walk_sitemap(
        self,
        sitemap_url: str,
        visited: Set[str],
        since: date | None,
        state: _TraversalState,
    ) -> Generator[UrlEntry, None, None]:
        if state.stop:
            return
        if sitemap_url in visited:
            return
        visited.add(sitemap_url)

        if not self._robots.allowed(sitemap_url):
            logger.debug("Skipping sitemap disallowed by robots %s", sitemap_url)
            return

        logger.debug("Fetching sitemap %s", sitemap_url)

        try:
            response = self._client.get(sitemap_url)
            response.raise_for_status()
            content = self._maybe_decompress(response)
            state.processed_documents += 1
            if state.max_documents and state.processed_documents >= state.max_documents:
                state.stop = True
        except httpx.HTTPError as exc:
            logger.warning("Failed to download sitemap %s: %s", sitemap_url, exc)
            return

        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            logger.warning("Unable to parse sitemap XML from %s: %s", sitemap_url, exc)
            return

        tag = self._strip_ns(root.tag)
        if tag == "sitemapindex":
            for loc, lastmod in self._parse_sitemapindex(root):
                if state.stop:
                    break
                if since and lastmod and lastmod.date() < since:
                    logger.debug("Skipping sitemap %s (lastmod %s < %s)", loc, lastmod.date(), since)
                    continue
                if loc:
                    yield from self._walk_sitemap(loc, visited, since, state)
        elif tag == "urlset":
            for entry in self._parse_urlset(root):
                yield entry
        else:
            logger.debug("Unknown sitemap root tag %s in %s", tag, sitemap_url)

    @staticmethod
    def _strip_ns(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def _parse_sitemapindex(self, root: ET.Element) -> Iterable[tuple[str | None, datetime | None]]:
        for child in root:
            if self._strip_ns(child.tag) != "sitemap":
                continue
            loc = None
            lastmod = None
            for sub in child:
                tag = self._strip_ns(sub.tag)
                if tag == "loc":
                    loc = sub.text.strip() if sub.text else None
                elif tag == "lastmod" and sub.text:
                    lastmod = self._parse_lastmod(sub.text.strip())
            yield loc, lastmod

    def _parse_urlset(self, root: ET.Element) -> Generator[UrlEntry, None, None]:
        for url_elem in root:
            if self._strip_ns(url_elem.tag) != "url":
                continue
            loc = None
            lastmod = None
            for child in url_elem:
                tag = self._strip_ns(child.tag)
                if tag == "loc":
                    loc = child.text.strip() if child.text else None
                elif tag == "lastmod" and child.text:
                    lastmod = self._parse_lastmod(child.text.strip())
            if loc:
                yield UrlEntry(loc=loc, lastmod=lastmod)

    @staticmethod
    def _parse_lastmod(value: str) -> datetime | None:
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(value, fmt)
                return SitemapClient._normalize_datetime(parsed)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(value)
            return SitemapClient._normalize_datetime(parsed)
        except ValueError:
            logger.debug("Unsupported lastmod format: %s", value)
            return None

    @staticmethod
    def _normalize_datetime(value: datetime) -> datetime:
        if value.tzinfo:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    @staticmethod
    def _maybe_decompress(response: httpx.Response) -> bytes:
        content = response.content
        if response.headers.get("content-encoding") == "gzip" or response.headers.get("Content-Type", "").endswith("gzip") or response.url.path.endswith(".gz"):
            try:
                return gzip.decompress(content)
            except OSError:
                with io.BytesIO(content) as buffer:
                    with gzip.GzipFile(fileobj=buffer) as gz:
                        return gz.read()
        return content
