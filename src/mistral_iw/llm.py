"""Mistral chat completions over plain HTTP (no SDK).

Responses are streamed so we can measure time-to-first-token; the stream is
reassembled into a single assistant message before returning.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings


class LLMError(Exception):
    pass


@dataclass
class LLMResult:
    message: dict[str, Any]  # assistant message, ready to append to the conversation
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    ttft_ms: float | None = None

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self.message.get("tool_calls") or []

    @property
    def content(self) -> str:
        return self.message.get("content") or ""


async def chat(
    client: httpx.AsyncClient,
    settings: Settings,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> LLMResult:
    payload: dict[str, Any] = {
        "model": settings.mistral_model,
        "messages": messages,
        "temperature": 0,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Accept": "text/event-stream",
    }
    url = f"{settings.mistral_base_url}/chat/completions"
    start = time.perf_counter()  # TTFT includes any 429 back-off: it's what the user waits for

    for attempt in range(settings.max_retries + 1):
        async with client.stream("POST", url, json=payload, headers=headers, timeout=60) as resp:
            if resp.status_code == 429 and attempt < settings.max_retries:
                await resp.aread()
                delay = _retry_after(resp) or settings.retry_base_delay * 2**attempt
            elif resp.status_code >= 400:
                body = (await resp.aread()).decode(errors="replace")[:500]
                raise LLMError(f"Mistral API returned HTTP {resp.status_code}: {body}")
            else:
                return await _read_stream(resp, start)
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return min(float(resp.headers["retry-after"]), 30.0)
    except (KeyError, ValueError):
        return None


def _text(content: Any) -> str:
    """Mistral content is a string, or a list of typed chunks for some models."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


async def _read_stream(resp: httpx.Response, start: float) -> LLMResult:
    content: list[str] = []
    calls: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    usage: dict[str, int] = {}
    finish_reason = None
    ttft_ms = None

    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = _text(delta.get("content"))
            if ttft_ms is None and (text or delta.get("tool_calls")):
                ttft_ms = round((time.perf_counter() - start) * 1000, 1)
            content.append(text)
            for tc in delta.get("tool_calls") or []:
                slot = _tool_call_slot(tc, calls, by_id, by_index)
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                args = fn.get("arguments")
                if args:
                    slot["function"]["arguments"] += args if isinstance(args, str) else json.dumps(args)
            finish_reason = choice.get("finish_reason") or finish_reason

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if calls:
        message["tool_calls"] = calls
    return LLMResult(message=message, finish_reason=finish_reason, usage=usage, ttft_ms=ttft_ms)


def _tool_call_slot(tc: dict, calls: list, by_id: dict, by_index: dict) -> dict[str, Any]:
    """Merge streamed tool-call fragments: by id when present, else by index."""
    call_id, index = tc.get("id"), tc.get("index")
    slot = by_id.get(call_id) if call_id else by_index.get(index) if index is not None else None
    if slot is None and not call_id and index is None and calls:
        slot = calls[-1]
    if slot is None:
        slot = {"id": call_id or f"call_{len(calls)}", "type": "function", "function": {"name": "", "arguments": ""}}
        calls.append(slot)
        if call_id:
            by_id[call_id] = slot
    if index is not None:
        by_index[index] = slot
    return slot
