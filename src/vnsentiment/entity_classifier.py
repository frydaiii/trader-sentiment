from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import OpenAI

from .config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_INPUT_TOKENS
from .models import ArticleInput


logger = logging.getLogger(__name__)


@dataclass
class ICBEntry:
    code: str
    name: str
    level: Optional[int] = None


@dataclass
class EntityDefinition:
    symbol: str
    organ_short_name: str
    organ_name: str
    icb_code: str
    icb_name: str
    icb_level: Optional[int] = None


@dataclass
class EntityMatch:
    symbol: str
    confidence: float
    rationale: Optional[str] = None


@dataclass
class EntityClassification:
    matches: List[EntityMatch]
    macro: bool = False


ENTITY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol of the entity that appears in the article."},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence that the article materially refers to the entity.",
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["symbol", "confidence"],
                "additionalProperties": False,
            },
        },
        "macro": {
            "type": "boolean",
            "description": "True if the article is primarily about macroeconomics or broad market-wide topics rather than specific companies.",
        },
    },
    "required": ["matches", "macro"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a financial news assistant tagging Vietnamese articles to HOSE-listed companies. "
    "Given a list of entities (ticker, company name, ICB industry) and an article, return only the tickers that are clearly "
    "discussed or impacted (explicit name, ticker, or unmistakable reference). "
    "Ignore generic sector mentions that do not identify a specific company. "
    "If no entity is relevant, return an empty list. "
    "Also set macro=true when the article is primarily about macroeconomics or whole stock market conditions rather than specific companies."
)


def load_icb_lookup(path: Path) -> Dict[str, ICBEntry]:
    """
    Build a mapping of ICB code to ICBEntry with name and level.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        lookup: Dict[str, ICBEntry] = {}
        for row in reader:
            code = row.get("icb_code")
            if not code:
                continue
            name = row.get("icb_name", "")
            level = row.get("level")
            lookup[code] = ICBEntry(code=code, name=name, level=int(level) if level and level.isdigit() else None)
    if not lookup:
        raise ValueError(f"No ICB entries loaded from {path}")
    return lookup


def load_hose_symbols(path: Path, icb_lookup: Dict[str, ICBEntry]) -> List[EntityDefinition]:
    """
    Load HOSE symbols from CSV, enriching with ICB names.
    """
    entities: List[EntityDefinition] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            icb_code = (row.get("icb_code") or "").strip()
            icb_entry = icb_lookup.get(icb_code)
            entities.append(
                EntityDefinition(
                    symbol=symbol,
                    organ_short_name=(row.get("organ_short_name") or "").strip(),
                    organ_name=(row.get("organ_name") or "").strip(),
                    icb_code=icb_code,
                    icb_name=icb_entry.name if icb_entry else "",
                    icb_level=icb_entry.level if icb_entry else None,
                )
            )
    if not entities:
        raise ValueError(f"No HOSE symbols loaded from {path}")
    return entities


class EntityClassifier:
    model: str
    temperature: float

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, client: Optional[OpenAI] = None) -> None:
        self.model = model
        self.temperature = temperature
        self._client = client or OpenAI()

    @staticmethod
    def _supports_temperature(model: str) -> bool:
        # Some OpenAI models only allow the default temperature (1) and reject overrides.
        fixed_temp_prefixes = ("gpt-5-nano",)
        return not any(model.startswith(prefix) for prefix in fixed_temp_prefixes)

    def classify_article(self, article: ArticleInput, entities: Iterable[EntityDefinition]) -> EntityClassification:
        request_body = self.build_request_body(article, entities)
        response = self._client.chat.completions.create(**request_body)
        payload = self._parse_response(response)
        logger.debug("Entity payload: %s", payload)
        matches: List[EntityMatch] = []
        for match in payload.get("matches", []):
            symbol = str(match.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            confidence = float(match.get("confidence", 0))
            rationale = match.get("reasoning") or match.get("rationale")
            matches.append(EntityMatch(symbol=symbol, confidence=confidence, rationale=rationale))
        macro_flag = bool(payload.get("macro", False))
        return EntityClassification(matches=matches, macro=macro_flag)

    def build_request_body(self, article: ArticleInput, entities: Iterable[EntityDefinition]) -> dict:
        """
        Build the OpenAI chat.completions request payload used for both realtime and batch calls.
        """
        entity_list = list(entities)
        prompt = self._build_prompt(article, entity_list)
        request: dict = {
            "model": self.model,
            "response_format": {"type": "json_schema", "json_schema": {"name": "entity_matches", "schema": ENTITY_RESPONSE_SCHEMA}},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        if self._supports_temperature(self.model):
            request["temperature"] = self.temperature
        elif self.temperature not in (None, 1):
            logger.warning("Model %s enforces default temperature; ignoring temperature override %s", self.model, self.temperature)
        return request

    def _build_prompt(self, article: ArticleInput, entities: List[EntityDefinition]) -> str:
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
        entity_block = self._entities_block(entities)
        return (
            "Choose zero or more tickers from the list below that the article clearly references. "
            "Do not add tickers that are only implied by general market commentary.\n\n"
            f"Entities:\n{entity_block}\n\n"
            f"Title: {article.title}\n{meta_block}\n\n"
            f"Article:\n{text}"
        )

    @staticmethod
    def _entities_block(entities: List[EntityDefinition]) -> str:
        company_lines = []
        industry_names = set()
        for entity in entities:
            short_name = entity.organ_short_name or entity.organ_name or "unknown company"
            company_lines.append(f"{entity.symbol}: {short_name}")
            if entity.icb_name:
                industry_names.add(entity.icb_name)
        industry_lines = sorted(industry_names) or ["industry unknown"]
        company_section = "Company short names:\n" + "\n".join(company_lines)
        industry_section = "Industry names:\n" + "\n".join(industry_lines)
        return f"{company_section}\n\n{industry_section}"

    @staticmethod
    def _parse_response(response) -> dict:
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "content") and message.content:
                return json.loads(message.content)
        raise ValueError("No textual output returned by OpenAI response.")
