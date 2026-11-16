"""One trace per question, with a span for every LLM and tool call.

Traces are returned to the client and appended to `traces/YYYY-MM-DD.jsonl`.
"""

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Span:
    kind: str  # "llm" or "tool"
    name: str
    start_ms: float  # offset from the start of the trace
    input: Any = None
    output: Any = None
    duration_ms: float = 0.0
    usage: dict[str, int] | None = None
    ttft_ms: float | None = None  # LLM spans: time to first streamed token
    error: str | None = None


@dataclass
class Trace:
    question: str
    model: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="milliseconds"))
    spans: list[Span] = field(default_factory=list)
    answer: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @contextmanager
    def span(self, kind: str, name: str, input: Any = None) -> Iterator[Span]:
        span = Span(kind=kind, name=name, start_ms=round(self._elapsed_ms(), 1), input=input)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - start) * 1000, 1)
            self.spans.append(span)

    def finish(self, answer: str | None, error: str | None = None) -> dict[str, Any]:
        self.answer = answer
        self.error = error
        self.latency_ms = round(self._elapsed_ms(), 1)
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        llm = [s for s in self.spans if s.kind == "llm"]
        tools = [s for s in self.spans if s.kind == "tool"]

        def total(key: str) -> int:
            return sum((s.usage or {}).get(key, 0) for s in llm)

        # TTFT: mean time-to-first-token over LLM calls.
        ttfts = [s.ttft_ms for s in llm if s.ttft_ms is not None]
        # TPOT: decode time divided by tokens generated after the first one.
        decode = [
            (s.duration_ms - s.ttft_ms, s.usage["completion_tokens"] - 1)
            for s in llm
            if s.ttft_ms is not None and s.usage and s.usage.get("completion_tokens", 0) > 1
        ]
        decode_tokens = sum(n for _, n in decode)

        return {
            "id": self.id,
            "started_at": self.started_at,
            "model": self.model,
            "question": self.question,
            "answer": self.answer,
            "error": self.error,
            "latency_ms": self.latency_ms if self.latency_ms is not None else round(self._elapsed_ms(), 1),
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "span_errors": sum(1 for s in self.spans if s.error),
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "total_tokens": total("total_tokens"),
            "ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
            "tpot_ms": round(sum(d for d, _ in decode) / decode_tokens, 2) if decode_tokens else None,
            "spans": [asdict(s) for s in self.spans],
        }


class TraceStore:
    """Append-only JSONL storage, one file per day."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def save(self, trace: dict[str, Any]) -> Path:
        path = self.directory / f"{trace['started_at'][:10]}.jsonl"
        line = json.dumps(trace, ensure_ascii=False, default=str)
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return path

    def _iter_traces(self) -> Iterator[dict[str, Any]]:
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.jsonl"), reverse=True):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip a corrupt line rather than failing the whole listing

    def list(self) -> list[dict[str, Any]]:
        """Every saved trace without its spans, newest first."""
        summaries = [{k: v for k, v in t.items() if k != "spans"} for t in self._iter_traces()]
        return sorted(summaries, key=_started_ts, reverse=True)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        return next((t for t in self._iter_traces() if t.get("id") == trace_id), None)


def _started_ts(trace: dict[str, Any]) -> float:
    try:
        return datetime.fromisoformat(trace["started_at"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return 0.0
