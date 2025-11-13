from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ArticleInput:
    """
    Representation of the text that will be sent to the LLM for sentiment scoring.
    """

    title: str
    text: str
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ArticleInput":
        text = payload.get("text") or payload.get("body") or ""
        title = payload.get("title") or payload.get("headline") or "Untitled"
        url = payload.get("url") or payload.get("link")
        metadata = {
            key: payload[key]
            for key in ("authors", "date_publish", "language", "source")
            if key in payload
        }
        return cls(title=title.strip(), text=text.strip(), url=url, metadata=metadata)


@dataclass
class SentimentResult:
    label: str
    score: float
    confidence: float
    reasoning: str
    keywords: list[str]
    model: str
    article_title: str
    article_url: Optional[str]
    metadata: Dict[str, Any]
    raw_response_id: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
            "model": self.model,
            "article_title": self.article_title,
            "article_url": self.article_url,
            "metadata": self.metadata,
            "raw_response_id": self.raw_response_id,
        }
