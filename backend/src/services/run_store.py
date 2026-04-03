"""Persistent run state and event history store for idempotency and resume."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class RunStore:
    """Simple JSON file based store keyed by run_id."""

    def __init__(self, workspace: str) -> None:
        root = Path(workspace)
        self._dir = root / ".runs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _path(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.json"

    def _default_doc(self, *, run_id: str, trace_id: str, topic: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "trace_id": trace_id,
            "topic": topic,
            "sequence": 0,
            "completed": False,
            "state": {},
            "events": [],
        }

    def load_or_create(self, *, run_id: str, trace_id: str, topic: str) -> dict[str, Any]:
        with self._lock:
            path = self._path(run_id)
            if not path.exists():
                doc = self._default_doc(run_id=run_id, trace_id=trace_id, topic=topic)
                path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
                return doc

            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                doc = self._default_doc(run_id=run_id, trace_id=trace_id, topic=topic)
            if not doc.get("trace_id"):
                doc["trace_id"] = trace_id
            if not doc.get("topic"):
                doc["topic"] = topic
            self._write(path, doc)
            return doc

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            path = self._path(run_id)
            if not path.exists():
                return None
            doc = json.loads(path.read_text(encoding="utf-8"))
            return doc if isinstance(doc, dict) else None

    def append_event(self, run_id: str, event: dict[str, Any], *, max_events: int = 5000) -> dict[str, Any]:
        with self._lock:
            path = self._path(run_id)
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                doc = loaded if isinstance(loaded, dict) else self._default_doc(run_id=run_id, trace_id="", topic="")
            else:
                doc = self._default_doc(run_id=run_id, trace_id="", topic="")
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

            self._write(path, doc)
            return doc

    def save_state(self, run_id: str, state: dict[str, Any], *, completed: bool | None = None) -> dict[str, Any]:
        with self._lock:
            path = self._path(run_id)
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                doc = loaded if isinstance(loaded, dict) else self._default_doc(run_id=run_id, trace_id="", topic="")
            else:
                doc = self._default_doc(run_id=run_id, trace_id="", topic="")
            doc["state"] = state
            if completed is not None:
                doc["completed"] = bool(completed)
            self._write(path, doc)
            return doc

    def events_after(self, run_id: str, sequence: int) -> list[dict[str, Any]]:
        doc = self.get(run_id)
        if not doc:
            return []
        events = doc.get("events")
        if not isinstance(events, list):
            return []

        filtered: list[dict[str, Any]] = []
        for item in events:
            if not isinstance(item, dict):
                continue
            seq = item.get("sequence")
            if isinstance(seq, int) and seq > sequence:
                filtered.append(item)
        return filtered

    def _write(self, path: Path, doc: dict[str, Any]) -> None:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
