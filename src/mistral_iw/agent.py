"""The agent: a hand-rolled function-calling loop."""

import json
from dataclasses import dataclass
from typing import Any

import httpx

from . import llm, prompts
from .config import Settings
from .tools import execute_tool, tool_schemas
from .tracing import Trace

STEP_CAP_ANSWER = "Sorry, I couldn't finish looking that up. Please try rephrasing your question."
LLM_ERROR_ANSWER = "Sorry, the language model is unavailable right now. Please try again in a moment."


@dataclass
class AgentResult:
    answer: str
    trace: dict[str, Any]


async def run_agent(
    client: httpx.AsyncClient,
    settings: Settings,
    message: str,
    history: list[dict[str, str]] | None = None,
) -> AgentResult:
    trace = Trace(question=message, model=settings.mistral_model)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.weather_agent_prompt()},
        *(history or []),
        {"role": "user", "content": message},
    ]
    tools = tool_schemas()

    for step in range(1, settings.max_llm_calls + 1):
        try:
            with trace.span("llm", f"chat #{step}", input={"messages": list(messages)}) as span:
                result = await llm.chat(client, settings, messages, tools)
                span.output = {"message": result.message, "finish_reason": result.finish_reason}
                span.usage = result.usage
                span.ttft_ms = result.ttft_ms
        except (llm.LLMError, httpx.HTTPError) as exc:
            return AgentResult(LLM_ERROR_ANSWER, trace.finish(LLM_ERROR_ANSWER, error=f"{type(exc).__name__}: {exc}"))

        if not result.tool_calls:
            answer = result.content.strip()
            return AgentResult(answer, trace.finish(answer))

        messages.append(result.message)
        for call in result.tool_calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            with trace.span("tool", name, input=_parse_args(arguments)) as span:
                output = await execute_tool(client, name, arguments)
                span.output = output
                if "error" in output:
                    span.error = str(output["error"])
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": json.dumps(output, ensure_ascii=False),
            })

    error = f"Step cap reached: {settings.max_llm_calls} LLM calls without a final answer"
    return AgentResult(STEP_CAP_ANSWER, trace.finish(STEP_CAP_ANSWER, error=error))


def _parse_args(arguments: str) -> Any:
    try:
        return json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
