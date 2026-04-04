"""Search dispatch helpers leveraging HelloAgents SearchTool."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from hello_agents.tools import SearchTool

from contracts import SearchBundleContract, SearchDocumentContract
from config import Configuration
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000
_GLOBAL_SEARCH_TOOL: SearchTool | None = None


def _get_search_tool() -> SearchTool:
    global _GLOBAL_SEARCH_TOOL

    if _GLOBAL_SEARCH_TOOL is not None:
        return _GLOBAL_SEARCH_TOOL

    # Some Windows terminals default to GBK and crash when upstream prints emoji.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:  # pragma: no cover - best effort only
        pass

    _GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")
    return _GLOBAL_SEARCH_TOOL


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[SearchBundleContract, str]:
    """Execute configured search backend and normalise response payload."""

    search_api = get_config_value(config.search_api)

    try:
        raw_response = _get_search_tool().run(
            {
                "input": query,
                "backend": search_api,
                "mode": "structured",
                "fetch_full_page": config.fetch_full_page,
                "max_results": config.resolved_max_sources(),
                "max_tokens_per_source": MAX_TOKENS_PER_SOURCE,
                "loop_count": loop_count,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Search backend %s failed: %s", search_api, exc)
        raise

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_api, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    documents: list[SearchDocumentContract] = []
    for item in results:
        if not isinstance(item, dict):
            continue

        source_type = "web"
        if isinstance(item.get("source_type"), str) and item.get("source_type") in {
            "web",
            "news",
            "academic",
            "other",
        }:
            source_type = str(item.get("source_type"))

        documents.append(
            SearchDocumentContract(
                title=str(item.get("title") or item.get("url") or ""),
                summary=str(item.get("content") or item.get("snippet") or ""),
                url=item.get("url"),
                confidence=float(item.get("score") or 0.5),
                source_type=source_type,  # type: ignore[arg-type]
                fetched_at=str(item.get("fetched_at") or datetime.now(timezone.utc).isoformat()),
                raw_content=str(item.get("raw_content") or ""),
            )
        )

    bundle = SearchBundleContract(
        backend=backend_label,
        answer=str(answer_text) if answer_text else None,
        notices=notices,
        documents=documents,
    )

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    return bundle, backend_label


def prepare_research_context(
    search_result: SearchBundleContract,
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    source_payload = {
        "results": [
            {
                "title": item.title,
                "url": str(item.url) if item.url else "",
                "content": item.summary,
                "raw_content": item.raw_content,
                "confidence": item.confidence,
                "source_type": item.source_type,
                "fetched_at": item.fetched_at,
            }
            for item in search_result.documents
        ]
    }

    sources_summary = format_sources(source_payload)
    context = deduplicate_and_format_sources(
        source_payload,
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if search_result.answer:
        context = f"AI直接答案：\n{search_result.answer}\n\n{context}"

    return sources_summary, context
