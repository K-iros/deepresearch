"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import Configuration, SearchAPI
from agent import DeepResearchAgent
from services.run_store import RunStore

logger.remove()

# 添加控制台日志处理程序
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


# 添加错误日志文件处理程序
logger.add(
    sink=sys.stderr,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )
    run_id: str | None = Field(
        default=None,
        description="Optional idempotent run identifier for resume/retry",
    )
    trace_id: str | None = Field(
        default=None,
        description="Optional distributed trace identifier",
    )
    resume_from_sequence: int = Field(
        default=0,
        ge=0,
        description="Replay events whose sequence is greater than this value",
    )
    mode: str | None = Field(
        default=None,
        description="Runtime mode: quick, standard, deep",
    )
    max_sources: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override max sources per task",
    )
    concurrency: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description="Override ready-task concurrency",
    )


class ResumeRunRequest(BaseModel):
    """Payload for resuming a historical run."""

    resume_from_sequence: int = Field(default=0, ge=0)
    search_api: SearchAPI | None = Field(default=None)
    trace_id: str | None = Field(default=None)
    mode: str | None = Field(default=None)
    max_sources: int | None = Field(default=None, ge=1, le=20)
    concurrency: int | None = Field(default=None, ge=1, le=8)


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )
    run_id: str = Field(..., description="Stable run identifier")
    trace_id: str = Field(..., description="Trace identifier")


class RunListItem(BaseModel):
    run_id: str
    topic: str
    friendly_name: str = ""
    status: str
    progress: int
    created_at: str
    updated_at: str
    completed: bool


class RunListResponse(BaseModel):
    items: list[RunListItem]
    total: int
    page: int
    page_size: int


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api
    if payload.mode is not None:
        overrides["runtime_mode"] = payload.mode
    if payload.max_sources is not None:
        overrides["max_sources"] = payload.max_sources
    if payload.concurrency is not None:
        overrides["task_concurrency"] = payload.concurrency

    return Configuration.from_env(overrides=overrides)


def _build_resume_config(payload: ResumeRunRequest) -> Configuration:
    overrides: Dict[str, Any] = {}
    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api
    if payload.mode is not None:
        overrides["runtime_mode"] = payload.mode
    if payload.max_sources is not None:
        overrides["max_sources"] = payload.max_sources
    if payload.concurrency is not None:
        overrides["task_concurrency"] = payload.concurrency
    return Configuration.from_env(overrides=overrides)


def _build_run_store(config: Configuration | None = None) -> RunStore:
    cfg = config or Configuration.from_env()
    run_workspace = cfg.notes_workspace or "./notes"
    return RunStore(run_workspace)


def _log_startup_configuration() -> None:
    """Log effective runtime configuration once on app startup."""

    config = Configuration.from_env()

    if config.llm_provider == "ollama":
        base_url = config.sanitized_ollama_url()
    elif config.llm_provider == "lmstudio":
        base_url = config.lmstudio_base_url
    else:
        base_url = config.llm_base_url or "unset"

    logger.info(
        "DeepResearch configuration loaded: provider={} model={} base_url={} search_api={} "
        "max_loops={} fetch_full_page={} tool_calling={} strip_thinking={} api_key={}",
        config.llm_provider,
        config.resolved_model() or "unset",
        base_url,
        (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
        config.max_web_research_loops,
        config.fetch_full_page,
        config.use_tool_calling,
        config.strip_thinking_tokens,
        _mask_secret(config.llm_api_key),
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        _log_startup_configuration()
        yield

    app = FastAPI(title="HelloAgents Deep Researcher", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest) -> ResearchResponse:
        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
            result = agent.run(
                payload.topic,
                run_id=payload.run_id,
                trace_id=payload.trace_id,
            )
        except ValueError as exc:  # Likely due to unsupported configuration
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Research failed") from exc

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "note_id": item.note_id,
                "note_path": item.note_path,
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
            run_id=result.run_id,
            trace_id=result.trace_id,
        )

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest) -> StreamingResponse:
        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream(
                    payload.topic,
                    run_id=payload.run_id,
                    trace_id=payload.trace_id,
                    resume_from_sequence=payload.resume_from_sequence,
                ):
                    sequence = event.get("sequence")
                    if isinstance(sequence, int):
                        yield f"id: {sequence}\n"
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Streaming research failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.get("/research/runs", response_model=RunListResponse)
    def list_research_runs(
        status: str | None = Query(default=None),
        keyword: str | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
    ) -> RunListResponse:
        store = _build_run_store()
        payload = store.list_runs(
            status=status,
            keyword=keyword,
            page=page,
            page_size=page_size,
        )
        return RunListResponse(**payload)

    @app.get("/research/runs/{run_id}")
    def get_research_run_detail(
        run_id: str,
        latest_events_limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        store = _build_run_store()
        detail = store.get_run_detail(run_id, latest_events_limit=latest_events_limit)
        if not detail:
            raise HTTPException(status_code=404, detail="Run not found")
        return detail

    @app.delete("/research/runs/{run_id}")
    def delete_research_run(run_id: str) -> dict[str, Any]:
        store = _build_run_store()
        deleted = store.delete_run(run_id, remove_artifacts=True)
        if not deleted:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"ok": True, "run_id": run_id}

    @app.post("/research/runs/{run_id}/resume")
    def resume_research_run(run_id: str, payload: ResumeRunRequest) -> StreamingResponse:
        config = _build_resume_config(payload)
        store = _build_run_store(config)
        doc = store.get(run_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Run not found")

        topic = str(doc.get("topic") or "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Run topic is missing")

        try:
            agent = DeepResearchAgent(config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        trace_id = payload.trace_id or str(doc.get("trace_id") or run_id)

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream(
                    topic,
                    run_id=run_id,
                    trace_id=trace_id,
                    resume_from_sequence=payload.resume_from_sequence,
                ):
                    sequence = event.get("sequence")
                    if isinstance(sequence, int):
                        yield f"id: {sequence}\n"
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Resuming streaming research failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.get("/research/runtime/options")
    def get_runtime_options() -> dict[str, Any]:
        config = Configuration.from_env()
        store = _build_run_store(config)
        return {
            "mode": config.normalized_runtime_mode(),
            "max_sources": config.resolved_max_sources(),
            "concurrency": config.resolved_task_concurrency(),
            "stage_duration_stats": store.aggregate_stage_duration_stats(),
            "model_profile": config.model_profile(),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
