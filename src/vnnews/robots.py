from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib import robotparser
from urllib.parse import urlparse, urlunparse

import httpx
import logging


logger = logging.getLogger(__name__)


@dataclass
class RobotsInfo:
    parser: robotparser.RobotFileParser
    sitemaps: List[str]


class RobotsManager:
    """
    Keeps a cache of robots.txt data per domain to determine fetch permissions and sitemap URLs.
    """

    def __init__(self, client: httpx.Client, user_agent: str) -> None:
        self._client = client
        self._user_agent = user_agent
        self._cache: Dict[str, RobotsInfo] = {}

    def get_info(self, base_url: str) -> RobotsInfo:
        parsed = urlparse(base_url)
        netloc = parsed.netloc or parsed.path
        scheme = parsed.scheme or "https"
        cache_key = f"{scheme}://{netloc}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        robots_url = urlunparse((scheme, netloc, "/robots.txt", "", "", ""))
        parser_obj = robotparser.RobotFileParser()
        parser_obj.set_url(robots_url)

        sitemaps: List[str] = []

        try:
            response = self._client.get(robots_url)
            text = response.text if response.is_success else ""
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch robots.txt for %s: %s", cache_key, exc)
            text = ""

        lines = text.splitlines()
        parser_obj.parse(lines)

        for line in lines:
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                if sitemap_url:
                    sitemaps.append(sitemap_url)

        info = RobotsInfo(parser=parser_obj, sitemaps=sitemaps)
        self._cache[cache_key] = info
        return info

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.netloc:
            return True
        base = urlunparse((parsed.scheme or "https", parsed.netloc, "/", "", "", ""))
        info = self.get_info(base)
        allowed = info.parser.can_fetch(self._user_agent, url)
        if not allowed:
            logger.debug("Blocked by robots.txt: %s", url)
        return allowed

    def get_sitemaps(self, base_url: str) -> List[str]:
        return self.get_info(base_url).sitemaps
