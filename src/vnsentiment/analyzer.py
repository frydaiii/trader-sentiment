from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI

from .config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_INPUT_TOKENS
from .models import ArticleInput, SentimentResult


logger = logging.getLogger(__name__)


SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "score": {
            "type": "number",
            "description": "Sentiment score between -1 (very negative) and 1 (very positive).",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "reasoning": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key words or phrases influencing the sentiment classification.",
        },
    },
    "required": ["label", "score", "confidence", "reasoning", "keywords"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a financial news sentiment analyst. "
    "Given an article, classify the sentiment about Vietnam's economic outlook "
    "or market tone as positive, neutral, or negative. "
    "Provide a numeric score between -1 and 1 and explain the reasoning briefly."
)


@dataclass
class SentimentAnalyzer:
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    client: Optional[OpenAI] = None

    def __post_init__(self) -> None:
        self._client = self.client or OpenAI()

    def score_article(self, article: ArticleInput) -> SentimentResult:
        prompt = self._build_prompt(article)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_schema", "json_schema": {"name": "sentiment", "schema": SENTIMENT_SCHEMA}},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
        )
        payload = self._parse_response(response)
        logger.debug("Sentiment payload: %s", payload)
        return SentimentResult(
            label=payload["label"],
            score=float(payload["score"]),
            confidence=float(payload["confidence"]),
            reasoning=payload["reasoning"],
            keywords=payload.get("keywords", []),
            model=self.model,
            article_title=article.title,
            article_url=article.url,
            metadata=article.metadata,
            raw_response_id=getattr(response, "id", None),
        )

    def score_articles(self, articles: Iterable[ArticleInput]) -> List[SentimentResult]:
        results = []
        for article in articles:
            try:
                results.append(self.score_article(article))
            except Exception as exc:  # pragma: no cover - logging path
                logger.error("Failed to score article '%s': %s", article.title, exc, exc_info=True)
        return results

    def _build_prompt(self, article: ArticleInput) -> str:
        text = article.text.strip()
        max_chars = MAX_INPUT_TOKENS * 4
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]..."
        meta_lines = []
        if article.url:
            meta_lines.append(f"URL: {article.url}")
        for key, value in article.metadata.items():
            meta_lines.append(f"{key}: {value}")
        meta_block = "\n".join(meta_lines)
        return f"Title: {article.title}\n{meta_block}\n\nArticle:\n{text}"

    @staticmethod
    def _parse_response(response) -> dict:
        # Extract the response content from chat completion
        if hasattr(response, 'choices') and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, 'content') and message.content:
                return json.loads(message.content)
        raise ValueError("No textual output returned by OpenAI response.")
