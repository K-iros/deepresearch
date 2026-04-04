"""Persistent run state and event history store for idempotency and resume."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


class RunStore:
    """Simple JSON file based store keyed by run_id."""

    def __init__(self, workspace: str) -> None:
        root = Path(workspace)
        self._dir = root / ".runs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._readable_dir = self._dir / "readable"
        self._readable_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._cache: dict[str, dict[str, Any]] = {}

    def _path(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.json"

    def _events_path(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.events.jsonl"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _slugify_topic(topic: str, *, fallback: str = "run", max_length: int = 32) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(topic or "").strip())
        cleaned = cleaned.strip("_")
        if not cleaned:
            cleaned = fallback
        return cleaned[:max_length]

    def _friendly_name(self, *, topic: str, run_id: str, created_at: str) -> str:
        prefix = str(created_at or self._utc_now())[:19].replace("-", "").replace(":", "").replace("T", "-")
        topic_slug = self._slugify_topic(topic)
        return f"{prefix}-{topic_slug}-{run_id[:8]}"

    def _readable_path(self, friendly_name: str) -> Path:
        safe_name = self._slugify_topic(friendly_name, fallback="run_meta", max_length=96)
        return self._readable_dir / f"{safe_name}.json"

    def _default_doc(self, *, run_id: str, trace_id: str, topic: str) -> dict[str, Any]:
        now = self._utc_now()
        return {
            "run_id": run_id,
            "trace_id": trace_id,
            "topic": topic,
            "friendly_name": self._friendly_name(topic=topic, run_id=run_id, created_at=now),
            "sequence": 0,
            "completed": False,
            "created_at": now,
            "updated_at": now,
            "state": {},
            "events": [],
            "artifact_paths": [],
        }

    def _normalize_doc(self, doc: dict[str, Any], *, run_id: str, trace_id: str, topic: str) -> dict[str, Any]:
        if not isinstance(doc.get("run_id"), str) or not doc.get("run_id"):
            doc["run_id"] = run_id
        if not isinstance(doc.get("trace_id"), str) or not doc.get("trace_id"):
            doc["trace_id"] = trace_id
        if not isinstance(doc.get("topic"), str) or not doc.get("topic"):
            doc["topic"] = topic

        if not isinstance(doc.get("state"), dict):
            doc["state"] = {}
        if not isinstance(doc.get("events"), list):
            doc["events"] = []
        if not isinstance(doc.get("artifact_paths"), list):
            doc["artifact_paths"] = []

        if not isinstance(doc.get("sequence"), int):
            doc["sequence"] = int(doc.get("sequence") or 0)
        if not isinstance(doc.get("completed"), bool):
            doc["completed"] = bool(doc.get("completed"))

        now = self._utc_now()
        if not isinstance(doc.get("created_at"), str) or not doc.get("created_at"):
            doc["created_at"] = now
        if not isinstance(doc.get("updated_at"), str) or not doc.get("updated_at"):
            doc["updated_at"] = now

        if not isinstance(doc.get("friendly_name"), str) or not doc.get("friendly_name"):
            doc["friendly_name"] = self._friendly_name(
                topic=str(doc.get("topic") or topic),
                run_id=str(doc.get("run_id") or run_id),
                created_at=str(doc.get("created_at") or now),
            )

        return doc

    def _write_readable_summary_unlocked(self, doc: dict[str, Any]) -> None:
        friendly_name = str(doc.get("friendly_name") or "")
        if not friendly_name:
            return

        payload = {
            "run_id": str(doc.get("run_id") or ""),
            "trace_id": str(doc.get("trace_id") or ""),
            "topic": str(doc.get("topic") or ""),
            "friendly_name": friendly_name,
            "created_at": str(doc.get("created_at") or ""),
            "updated_at": str(doc.get("updated_at") or ""),
            "completed": bool(doc.get("completed") or False),
        }

        path = self._readable_path(friendly_name)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_doc_unlocked(self, run_id: str, *, trace_id: str = "", topic: str = "") -> dict[str, Any]:
        if run_id in self._cache:
            return self._cache[run_id]

        path = self._path(run_id)
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            doc = loaded if isinstance(loaded, dict) else self._default_doc(run_id=run_id, trace_id=trace_id, topic=topic)
        else:
            doc = self._default_doc(run_id=run_id, trace_id=trace_id, topic=topic)

        doc = self._normalize_doc(doc, run_id=run_id, trace_id=trace_id, topic=topic)
        self._cache[run_id] = doc
        return doc

    def load_or_create(self, *, run_id: str, trace_id: str, topic: str) -> dict[str, Any]:
        with self._lock:
            doc = self._load_doc_unlocked(run_id, trace_id=trace_id, topic=topic)
            if not doc.get("trace_id"):
                doc["trace_id"] = trace_id
            if not doc.get("topic"):
                doc["topic"] = topic
            doc["updated_at"] = self._utc_now()
            self._write(self._path(run_id), doc)
            self._write_readable_summary_unlocked(doc)
            return doc

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            path = self._path(run_id)
            if not path.exists() and run_id not in self._cache:
                return None
            doc = self._load_doc_unlocked(run_id)
            return dict(doc)

    def append_event(self, run_id: str, event: dict[str, Any], *, max_events: int = 5000) -> dict[str, Any]:
        with self._lock:
            doc = self._load_doc_unlocked(run_id)
            events = doc.setdefault("events", [])
            if not isinstance(events, list):
                events = []
                doc["events"] = events

            events.append(event)
            if len(events) > max_events:
                doc["events"] = events[-max_events:]

            sequence = event.get("sequence")
            if isinstance(sequence, int):
                doc["sequence"] = max(int(doc.get("sequence") or 0), sequence)

            doc["updated_at"] = self._utc_now()
            self._append_event_line(run_id, event)
            return dict(doc)

    def save_state(self, run_id: str, state: dict[str, Any], *, completed: bool | None = None) -> dict[str, Any]:
        with self._lock:
            doc = self._load_doc_unlocked(run_id)
            doc["state"] = state
            if completed is not None:
                doc["completed"] = bool(completed)

            artifact_paths = state.get("artifact_paths") if isinstance(state, dict) else None
            if isinstance(artifact_paths, list):
                doc["artifact_paths"] = artifact_paths

            doc["updated_at"] = self._utc_now()
            self._write(self._path(run_id), doc)
            self._write_readable_summary_unlocked(doc)
            return dict(doc)

    def events_after(self, run_id: str, sequence: int) -> list[dict[str, Any]]:
        with self._lock:
            events = self._events_from_jsonl_unlocked(run_id)
            if not events:
                doc = self._load_doc_unlocked(run_id)
                legacy = doc.get("events")
                if isinstance(legacy, list):
                    events = [item for item in legacy if isinstance(item, dict)]

            filtered: list[dict[str, Any]] = []
            for item in events:
                seq = item.get("sequence")
                if isinstance(seq, int) and seq > sequence:
                    filtered.append(item)
            return filtered

    def list_runs(
        self,
        *,
        status: str | None = None,
        keyword: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        with self._lock:
            page = max(1, int(page or 1))
            page_size = max(1, min(100, int(page_size or 20)))
            normalized_status = (status or "").strip().lower()
            normalized_keyword = (keyword or "").strip().lower()

            summaries: list[dict[str, Any]] = []
            for path in self._dir.glob("*.json"):
                if path.name.endswith(".events.jsonl"):
                    continue
                if path.name == "runs_index.json":
                    continue
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    continue

                run_id = str(loaded.get("run_id") or path.stem)
                doc = self._normalize_doc(
                    loaded,
                    run_id=run_id,
                    trace_id=str(loaded.get("trace_id") or ""),
                    topic=str(loaded.get("topic") or ""),
                )
                self._cache[run_id] = doc

                summary = self._summary_from_doc_unlocked(doc)
                summary_status = str(summary.get("status") or "").lower()
                target = f"{summary.get('run_id', '')} {summary.get('topic', '')}".lower()

                if normalized_status and summary_status != normalized_status:
                    continue
                if normalized_keyword and normalized_keyword not in target:
                    continue
                summaries.append(summary)

            summaries.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
            total = len(summaries)
            start = (page - 1) * page_size
            end = start + page_size
            items = summaries[start:end]

            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
            }

    def delete_run(self, run_id: str, *, remove_artifacts: bool = True) -> bool:
        with self._lock:
            path = self._path(run_id)
            if not path.exists() and run_id not in self._cache:
                return False

            doc = self._load_doc_unlocked(run_id)

            if remove_artifacts:
                state = doc.get("state") if isinstance(doc.get("state"), dict) else {}
                artifact_root = state.get("artifact_root") if isinstance(state, dict) else None
                if isinstance(artifact_root, str) and artifact_root.strip():
                    artifact_path = Path(artifact_root)
                    if artifact_path.exists() and artifact_path.is_dir():
                        shutil.rmtree(artifact_path, ignore_errors=True)

            events_path = self._events_path(run_id)
            if events_path.exists():
                events_path.unlink(missing_ok=True)

            if path.exists():
                path.unlink(missing_ok=True)

            friendly_name = str(doc.get("friendly_name") or "")
            if friendly_name:
                readable_path = self._readable_path(friendly_name)
                if readable_path.exists():
                    readable_path.unlink(missing_ok=True)

            if run_id in self._cache:
                self._cache.pop(run_id, None)

            return True

    def get_run_detail(self, run_id: str, *, latest_events_limit: int = 50) -> dict[str, Any] | None:
        with self._lock:
            path = self._path(run_id)
            if not path.exists() and run_id not in self._cache:
                return None
            doc = self._load_doc_unlocked(run_id)
            if not doc:
                return None

            state = doc.get("state") if isinstance(doc.get("state"), dict) else {}
            todo_items = state.get("todo_items") if isinstance(state.get("todo_items"), list) else []
            artifact_paths = state.get("artifact_paths") if isinstance(state.get("artifact_paths"), list) else []
            if not artifact_paths:
                raw_paths = doc.get("artifact_paths")
                if isinstance(raw_paths, list):
                    artifact_paths = raw_paths

            latest_events = self._events_from_jsonl_unlocked(run_id)
            if not latest_events:
                raw_events = doc.get("events")
                latest_events = [item for item in raw_events if isinstance(item, dict)] if isinstance(raw_events, list) else []
            latest_events = latest_events[-max(1, int(latest_events_limit)) :]

            summary = self._summary_from_doc_unlocked(doc)

            return {
                **summary,
                "trace_id": doc.get("trace_id") or "",
                "todo_items": todo_items,
                "report_markdown": state.get("structured_report") or state.get("running_summary") or "",
                "artifact_paths": artifact_paths,
                "latest_events": latest_events,
                "stage_durations": state.get("stage_durations") if isinstance(state.get("stage_durations"), dict) else {},
                "runtime_config": {
                    "mode": state.get("runtime_mode") or "standard",
                    "max_sources": int(state.get("max_sources") or 5),
                    "concurrency": int(state.get("task_concurrency") or 2),
                },
            }

    def aggregate_stage_duration_stats(self, *, recent_limit: int = 200) -> dict[str, float]:
        with self._lock:
            rows = self.list_runs(page=1, page_size=max(1, recent_limit)).get("items", [])
            totals: dict[str, float] = {}
            counts: dict[str, int] = {}

            for row in rows:
                run_id = row.get("run_id")
                if not isinstance(run_id, str) or not run_id:
                    continue
                detail = self.get_run_detail(run_id, latest_events_limit=1)
                if not detail:
                    continue
                stage_durations = detail.get("stage_durations")
                if not isinstance(stage_durations, dict):
                    continue

                for stage, value in stage_durations.items():
                    if not isinstance(value, (int, float)):
                        continue
                    totals[stage] = totals.get(stage, 0.0) + float(value)
                    counts[stage] = counts.get(stage, 0) + 1

            averaged: dict[str, float] = {}
            for stage, total in totals.items():
                count = max(1, counts.get(stage, 1))
                averaged[stage] = round(total / count, 3)
            return averaged

    def _summary_from_doc_unlocked(self, doc: dict[str, Any]) -> dict[str, Any]:
        status, progress = self._status_progress_from_doc(doc)
        return {
            "run_id": str(doc.get("run_id") or ""),
            "topic": str(doc.get("topic") or ""),
            "friendly_name": str(doc.get("friendly_name") or ""),
            "status": status,
            "progress": progress,
            "created_at": str(doc.get("created_at") or ""),
            "updated_at": str(doc.get("updated_at") or ""),
            "completed": bool(doc.get("completed") or False),
        }

    def _status_progress_from_doc(self, doc: dict[str, Any]) -> tuple[str, int]:
        state = doc.get("state") if isinstance(doc.get("state"), dict) else {}
        todo_items = state.get("todo_items") if isinstance(state.get("todo_items"), list) else []

        if bool(doc.get("completed") or state.get("completed")):
            return "completed", 100

        if not todo_items:
            return "pending", 0

        total = len(todo_items)
        status_list: list[str] = []
        done_count = 0
        for item in todo_items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "pending").strip().lower()
            status_list.append(status)
            if status in {"completed", "skipped", "failed"}:
                done_count += 1

        progress = int((max(0, done_count) / max(1, total)) * 100)

        if any(status in {"in_progress", "retrying"} for status in status_list):
            return "running", progress
        if any(status == "failed" for status in status_list):
            return "failed", progress
        if any(status == "pending" for status in status_list):
            return "pending", progress
        if all(status in {"completed", "skipped"} for status in status_list):
            return "completed", 100
        return "running", progress

    def _events_from_jsonl_unlocked(self, run_id: str) -> list[dict[str, Any]]:
        events_path = self._events_path(run_id)
        if not events_path.exists():
            return []

        events: list[dict[str, Any]] = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
        return events

    def _append_event_line(self, run_id: str, event: dict[str, Any]) -> None:
        path = self._events_path(run_id)
        line = json.dumps(event, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(line)
            fp.write("\n")

    def _write(self, path: Path, doc: dict[str, Any]) -> None:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
