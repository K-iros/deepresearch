"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator
from uuid import uuid4

from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool

from config import Configuration
from contracts import EventEnvelopeContract
from prompts import (
    report_writer_instructions,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)
from models import SummaryState, SummaryStateOutput, TodoItem
from services.planner import PlanningService
from services.reporter import ReportingService
from services.run_store import RunStore
from services.search import dispatch_search, prepare_research_context
from services.summarizer import SummarizationService
from services.tool_events import ToolCallTracker

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """Coordinator orchestrating TODO-based research workflow using HelloAgents."""

    def __init__(self, config: Configuration | None = None) -> None:
        self._ensure_utf8_stdout()
        self.config = config or Configuration.from_env()
        self.llm = self._init_llm()

        self.note_tool = (
            NoteTool(workspace=self.config.notes_workspace)
            if self.config.enable_notes
            else None
        )

        self.tools_registry: ToolRegistry | None = None
        if self.note_tool:
            registry = ToolRegistry()
            registry.register_tool(self.note_tool)
            self.tools_registry = registry

        self._tool_tracker = ToolCallTracker(
            self.config.notes_workspace if self.config.enable_notes else None
        )
        run_workspace = self.config.notes_workspace or "./notes"
        self.run_store = RunStore(run_workspace)
        self._state_lock = Lock()

        self.todo_agent = self._create_tool_aware_agent(
            name="研究规划专家",
            system_prompt=todo_planner_system_prompt.strip(),
        )
        self.report_agent = self._create_tool_aware_agent(
            name="报告撰写专家",
            system_prompt=report_writer_instructions.strip(),
        )

        self._summarizer_factory: Callable[[], ToolAwareSimpleAgent] = lambda: self._create_tool_aware_agent(
            name="任务总结专家",
            system_prompt=task_summarizer_instructions.strip(),
        )

        self.planner = PlanningService(self.todo_agent, self.config)
        self.summarizer = SummarizationService(self._summarizer_factory, self.config)
        self.reporting = ReportingService(self.report_agent, self.config)

    @staticmethod
    def _ensure_utf8_stdout() -> None:
        """Avoid UnicodeEncodeError in Windows GBK terminals from third-party prints."""

        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        except Exception:  # pragma: no cover - best effort only
            pass

    def _init_llm(self) -> HelloAgentsLLM:
        llm_kwargs: dict[str, Any] = {"temperature": 0.0}

        model_id = self.config.llm_model_id or self.config.local_llm
        if model_id:
            llm_kwargs["model"] = model_id

        provider = (self.config.llm_provider or "").strip()
        if provider:
            llm_kwargs["provider"] = provider

        if provider == "ollama":
            llm_kwargs["base_url"] = self.config.sanitized_ollama_url()
            llm_kwargs["api_key"] = self.config.llm_api_key or "ollama"
        elif provider == "lmstudio":
            llm_kwargs["base_url"] = self.config.lmstudio_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
        else:
            if self.config.llm_base_url:
                llm_kwargs["base_url"] = self.config.llm_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key

        return HelloAgentsLLM(**llm_kwargs)

    def _create_tool_aware_agent(self, *, name: str, system_prompt: str) -> ToolAwareSimpleAgent:
        return ToolAwareSimpleAgent(
            name=name,
            llm=self.llm,
            system_prompt=system_prompt,
            enable_tool_calling=self.tools_registry is not None,
            tool_registry=self.tools_registry,
            tool_call_listener=self._tool_tracker.record,
        )

    def run(
        self,
        topic: str,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> SummaryStateOutput:
        state = self._bootstrap_state(topic, run_id=run_id, trace_id=trace_id)

        self._ensure_tasks(state)
        for task in self._ordered_ready_tasks(state.todo_items):
            if task.status in {"completed", "skipped", "failed"}:
                continue
            for _ in self._execute_task_with_retry(state, task, emit_stream=False, step=None):
                pass
            self._snapshot_state(state)

        report = self.reporting.generate_report(state)
        self._sync_tool_events(state, emit_stream=False, step=None)
        state.structured_report = report
        state.running_summary = report
        self._persist_final_report(state, report)
        state.completed = True
        self._snapshot_state(state, completed=True)

        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=state.todo_items,
            run_id=state.run_id,
            trace_id=state.trace_id,
        )

    def run_stream(
        self,
        topic: str,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        resume_from_sequence: int = 0,
    ) -> Iterator[dict[str, Any]]:
        state = self._bootstrap_state(topic, run_id=run_id, trace_id=trace_id)
        persisted = self.run_store.get(state.run_id) or {}

        if state.completed or bool(persisted.get("completed")):
            for event in self.run_store.events_after(state.run_id, resume_from_sequence):
                yield event
            return

        if resume_from_sequence > 0:
            for event in self.run_store.events_after(state.run_id, resume_from_sequence):
                yield event

            persisted = self.run_store.get(state.run_id) or {}
            if persisted.get("completed"):
                return

        yield self._emit_event(
            state,
            event_type="status",
            payload={"message": "初始化研究流程"},
            step=0,
        )

        tasks_preexisting = bool(state.todo_items)
        self._ensure_tasks(state)
        if not tasks_preexisting:
            yield self._emit_event(
                state,
                event_type="todo_list",
                payload={"tasks": [self._serialize_task(t) for t in state.todo_items]},
                step=0,
            )

        step_map = {
            task.id: index
            for index, task in enumerate(self._ordered_ready_tasks(state.todo_items), start=1)
        }

        completed_ids = {task.id for task in state.todo_items if task.status == "completed"}

        while True:
            remaining = [
                task for task in state.todo_items if task.status in {"pending", "in_progress", "retrying"}
            ]
            if not remaining:
                break

            ready = [
                task
                for task in remaining
                if all(dep in completed_ids for dep in task.depends_on)
            ]

            if not ready:
                for task in remaining:
                    task.status = "skipped"
                    task.last_error = "依赖未满足或存在循环依赖"
                    yield self._emit_event(
                        state,
                        event_type="task_status",
                        payload=self._task_status_payload(task, detail=task.last_error),
                        task_id=task.id,
                        step=step_map.get(task.id),
                    )
                break

            ready.sort(key=lambda item: (-item.priority, item.id))
            task = ready[0]
            step = step_map.get(task.id)

            yield self._emit_event(
                state,
                event_type="task_status",
                payload=self._task_status_payload(task, status="in_progress"),
                task_id=task.id,
                step=step,
            )

            for event in self._execute_task_with_retry(state, task, emit_stream=True, step=step):
                yield event

            if task.status == "completed":
                completed_ids.add(task.id)

            self._snapshot_state(state)

        report = self.reporting.generate_report(state)
        for event in self._sync_tool_events(state, emit_stream=True, step=len(state.todo_items) + 1):
            yield event

        state.structured_report = report
        state.running_summary = report
        note_event = self._persist_final_report(state, report)
        if note_event:
            yield self._emit_event(
                state,
                event_type="report_note",
                payload=note_event,
                step=len(state.todo_items) + 1,
            )

        yield self._emit_event(
            state,
            event_type="final_report",
            payload={
                "report": report,
                "note_id": state.report_note_id,
                "note_path": state.report_note_path,
            },
            step=len(state.todo_items) + 1,
        )

        state.completed = True
        self._snapshot_state(state, completed=True)
        yield self._emit_event(state, event_type="done", payload={"ok": True}, step=len(state.todo_items) + 1)

    def _ensure_tasks(self, state: SummaryState) -> None:
        if state.todo_items:
            return

        state.todo_items = self.planner.plan_todo_list(state)
        if not state.todo_items:
            state.todo_items = [self.planner.create_fallback_task(state)]

    def _ordered_ready_tasks(self, tasks: list[TodoItem]) -> list[TodoItem]:
        task_map = {task.id: task for task in tasks}
        in_degree = {task.id: 0 for task in tasks}
        graph: dict[int, list[int]] = {task.id: [] for task in tasks}

        for task in tasks:
            for dep in task.depends_on:
                if dep not in task_map:
                    continue
                in_degree[task.id] += 1
                graph[dep].append(task.id)

        ordered: list[TodoItem] = []
        available = [task for task in tasks if in_degree[task.id] == 0]
        available.sort(key=lambda item: (-item.priority, item.id))

        while available:
            current = available.pop(0)
            ordered.append(current)
            for nxt in graph[current.id]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    available.append(task_map[nxt])
            available.sort(key=lambda item: (-item.priority, item.id))

        if len(ordered) != len(tasks):
            unresolved = [task for task in tasks if task not in ordered]
            unresolved.sort(key=lambda item: (-item.priority, item.id))
            ordered.extend(unresolved)

        return ordered

    def _execute_task_with_retry(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        emit_stream: bool,
        step: int | None,
    ) -> Iterator[dict[str, Any]]:
        max_attempts = max(1, task.retry_max_attempts)

        for attempt in range(1, max_attempts + 1):
            task.attempt_count = attempt

            if emit_stream and attempt > 1:
                yield self._emit_event(
                    state,
                    event_type="status",
                    payload={
                        "message": f"任务 {task.id} 第 {attempt} 次重试",
                        "retry_attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                    task_id=task.id,
                    step=step,
                )

            try:
                for event in self._execute_task(state, task, emit_stream=emit_stream, step=step):
                    yield event
                if task.status in {"completed", "skipped"}:
                    return
            except Exception as exc:  # pragma: no cover
                logger.exception("Task %s execution failed", task.id, exc_info=exc)
                task.last_error = str(exc)
                task.status = "retrying" if attempt < max_attempts else "failed"

                if attempt >= max_attempts:
                    if emit_stream:
                        yield self._emit_event(
                            state,
                            event_type="task_status",
                            payload=self._task_status_payload(task, status="failed", detail=str(exc)),
                            task_id=task.id,
                            step=step,
                        )
                    return

    def _execute_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        emit_stream: bool,
        step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        task.status = "in_progress"

        search_bundle, backend = dispatch_search(
            task.query,
            self.config,
            state.research_loop_count,
        )
        task.notices = search_bundle.notices

        for event in self._sync_tool_events(state, emit_stream=emit_stream, step=step):
            yield event

        if search_bundle.notices and emit_stream:
            for notice in search_bundle.notices:
                if notice:
                    yield self._emit_event(
                        state,
                        event_type="status",
                        payload={"message": notice},
                        task_id=task.id,
                        step=step,
                    )

        if not search_bundle.documents:
            task.status = "skipped"
            if emit_stream:
                yield self._emit_event(
                    state,
                    event_type="task_status",
                    payload=self._task_status_payload(task, status="skipped"),
                    task_id=task.id,
                    step=step,
                )
            return

        sources_summary, context = prepare_research_context(search_bundle, self.config)
        task.sources_summary = sources_summary

        with self._state_lock:
            state.web_research_results.append(context)
            state.sources_gathered.append(sources_summary)
            state.research_loop_count += 1

        if emit_stream:
            yield self._emit_event(
                state,
                event_type="sources",
                payload={
                    "latest_sources": sources_summary,
                    "raw_context": context,
                    "backend": backend,
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                },
                task_id=task.id,
                step=step,
            )

            summary_stream, summary_getter = self.summarizer.stream_task_summary(state, task, context)
            try:
                for chunk in summary_stream:
                    if not chunk:
                        continue
                    yield self._emit_event(
                        state,
                        event_type="task_summary_chunk",
                        payload={
                            "content": chunk,
                            "note_id": task.note_id,
                        },
                        task_id=task.id,
                        step=step,
                    )
                    for event in self._sync_tool_events(state, emit_stream=True, step=step):
                        yield event
            finally:
                summary_text = summary_getter()
        else:
            summary_text = self.summarizer.summarize_task(state, task, context)
            self._sync_tool_events(state, emit_stream=False, step=step)

        task.summary = summary_text.strip() if summary_text else "暂无可用信息"
        task.status = "completed"

        if emit_stream:
            for event in self._sync_tool_events(state, emit_stream=True, step=step):
                yield event
            yield self._emit_event(
                state,
                event_type="task_status",
                payload=self._task_status_payload(
                    task,
                    status="completed",
                    summary=task.summary,
                    sources_summary=task.sources_summary,
                ),
                task_id=task.id,
                step=step,
            )

    def _sync_tool_events(
        self,
        state: SummaryState,
        *,
        emit_stream: bool,
        step: int | None,
    ) -> list[dict[str, Any]]:
        drained = self._tool_tracker.drain(state, step=step)
        if not emit_stream:
            return []

        wrapped: list[dict[str, Any]] = []
        for event in drained:
            raw_type = str(event.get("type") or "tool_call")
            payload = dict(event)
            payload.pop("type", None)
            task_id = payload.get("task_id") if isinstance(payload.get("task_id"), int) else None
            wrapped.append(
                self._emit_event(
                    state,
                    event_type=raw_type,
                    payload=payload,
                    task_id=task_id,
                    step=step,
                )
            )
        return wrapped

    def _emit_event(
        self,
        state: SummaryState,
        *,
        event_type: str,
        payload: dict[str, Any],
        task_id: int | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        envelope = EventEnvelopeContract(
            event_type=event_type,
            run_id=state.run_id,
            trace_id=state.trace_id,
            sequence=state.next_sequence,
            task_id=task_id,
            payload=payload,
        )
        state.next_sequence += 1

        event_dict = envelope.model_dump()
        event_dict["type"] = event_type
        if step is not None:
            event_dict["step"] = step

        for key, value in payload.items():
            if key in {"event_type", "type", "run_id", "trace_id", "sequence", "timestamp", "task_id", "payload"}:
                continue
            event_dict[key] = value

        if task_id is not None:
            event_dict["task_id"] = task_id

        self.run_store.append_event(state.run_id, event_dict)
        return event_dict

    def _task_status_payload(
        self,
        task: TodoItem,
        *,
        status: str | None = None,
        detail: str | None = None,
        summary: str | None = None,
        sources_summary: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status or task.status,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "attempt_count": task.attempt_count,
            "priority": task.priority,
            "depends_on": task.depends_on,
            "retry_max_attempts": task.retry_max_attempts,
            "retry_backoff_seconds": task.retry_backoff_seconds,
        }
        if detail:
            payload["detail"] = detail
        if summary:
            payload["summary"] = summary
        if sources_summary:
            payload["sources_summary"] = sources_summary
        return payload

    def _bootstrap_state(
        self,
        topic: str,
        *,
        run_id: str | None,
        trace_id: str | None,
    ) -> SummaryState:
        resolved_run_id = (run_id or "").strip() or uuid4().hex
        resolved_trace_id = (trace_id or "").strip() or resolved_run_id

        run_doc = self.run_store.load_or_create(
            run_id=resolved_run_id,
            trace_id=resolved_trace_id,
            topic=topic,
        )

        state_payload = run_doc.get("state")
        if isinstance(state_payload, dict) and state_payload:
            restored = self._restore_state_from_payload(
                state_payload,
                topic=topic,
                run_id=resolved_run_id,
                trace_id=resolved_trace_id,
                persisted_sequence=int(run_doc.get("sequence") or 0),
            )
            return restored

        return SummaryState(
            research_topic=topic,
            run_id=resolved_run_id,
            trace_id=resolved_trace_id,
            next_sequence=max(1, int(run_doc.get("sequence") or 0) + 1),
        )

    def _restore_state_from_payload(
        self,
        payload: dict[str, Any],
        *,
        topic: str,
        run_id: str,
        trace_id: str,
        persisted_sequence: int,
    ) -> SummaryState:
        todos_raw = payload.get("todo_items") if isinstance(payload.get("todo_items"), list) else []
        todo_items: list[TodoItem] = []
        for item in todos_raw:
            if not isinstance(item, dict):
                continue
            try:
                todo_items.append(TodoItem(**item))
            except TypeError:
                continue

        return SummaryState(
            research_topic=str(payload.get("research_topic") or topic),
            run_id=run_id,
            trace_id=trace_id,
            next_sequence=max(int(payload.get("next_sequence") or 1), persisted_sequence + 1),
            web_research_results=list(payload.get("web_research_results") or []),
            sources_gathered=list(payload.get("sources_gathered") or []),
            research_loop_count=int(payload.get("research_loop_count") or 0),
            running_summary=payload.get("running_summary"),
            todo_items=todo_items,
            structured_report=payload.get("structured_report"),
            report_note_id=payload.get("report_note_id"),
            report_note_path=payload.get("report_note_path"),
            completed=bool(payload.get("completed") or False),
        )

    def _snapshot_state(self, state: SummaryState, *, completed: bool | None = None) -> None:
        payload = {
            "research_topic": state.research_topic,
            "run_id": state.run_id,
            "trace_id": state.trace_id,
            "next_sequence": state.next_sequence,
            "web_research_results": list(state.web_research_results),
            "sources_gathered": list(state.sources_gathered),
            "research_loop_count": state.research_loop_count,
            "running_summary": state.running_summary,
            "todo_items": [asdict(item) for item in state.todo_items],
            "structured_report": state.structured_report,
            "report_note_id": state.report_note_id,
            "report_note_path": state.report_note_path,
            "completed": state.completed if completed is None else completed,
        }
        self.run_store.save_state(state.run_id, payload, completed=completed)

    @property
    def _tool_call_events(self) -> list[dict[str, Any]]:
        return self._tool_tracker.as_dicts()

    def _serialize_task(self, task: TodoItem) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "depends_on": task.depends_on,
            "priority": task.priority,
            "retry_max_attempts": task.retry_max_attempts,
            "retry_backoff_seconds": task.retry_backoff_seconds,
            "status": task.status,
            "attempt_count": task.attempt_count,
            "summary": task.summary,
            "sources_summary": task.sources_summary,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "stream_token": task.stream_token,
        }

    def _persist_final_report(self, state: SummaryState, report: str) -> dict[str, Any] | None:
        if not self.note_tool or not report or not report.strip():
            return None

        note_title = f"研究报告：{state.research_topic}".strip() or "研究报告"
        tags = ["deep_research", "report"]
        content = report.strip()

        note_id = self._find_existing_report_note_id(state)
        response = ""

        if note_id:
            response = self.note_tool.run(
                {
                    "action": "update",
                    "note_id": note_id,
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            if response.startswith("❌"):
                note_id = None

        if not note_id:
            response = self.note_tool.run(
                {
                    "action": "create",
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            note_id = self._extract_note_id_from_text(response)

        if not note_id:
            return None

        state.report_note_id = note_id
        if self.config.notes_workspace:
            note_path = Path(self.config.notes_workspace) / f"{note_id}.md"
            state.report_note_path = str(note_path)
        else:
            note_path = None

        payload = {
            "note_id": note_id,
            "title": note_title,
            "content": content,
        }
        if note_path:
            payload["note_path"] = str(note_path)

        return payload

    def _find_existing_report_note_id(self, state: SummaryState) -> str | None:
        if state.report_note_id:
            return state.report_note_id

        for event in reversed(self._tool_tracker.as_dicts()):
            if event.get("tool") != "note":
                continue

            parameters = event.get("parsed_parameters") or {}
            if not isinstance(parameters, dict):
                continue

            action = parameters.get("action")
            if action not in {"create", "update"}:
                continue

            note_type = parameters.get("note_type")
            if note_type != "conclusion":
                title = parameters.get("title")
                if not (isinstance(title, str) and title.startswith("研究报告")):
                    continue

            note_id = parameters.get("note_id")
            if not note_id:
                note_id = self._tool_tracker._extract_note_id(event.get("result", ""))  # type: ignore[attr-defined]

            if note_id:
                return str(note_id)

        return None

    @staticmethod
    def _extract_note_id_from_text(response: str) -> str | None:
        if not response:
            return None

        match = re.search(r"ID:\s*([^\n]+)", response)
        if not match:
            return None

        return match.group(1).strip()


def run_deep_research(
    topic: str,
    config: Configuration | None = None,
    *,
    run_id: str | None = None,
) -> SummaryStateOutput:
    agent = DeepResearchAgent(config=config)
    return agent.run(topic, run_id=run_id)
