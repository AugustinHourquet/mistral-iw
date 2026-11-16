# Weather agent

A minimal AI weather agent with a chat UI. The model is Mistral, called over plain HTTP with function calling (no SDK, no agent framework). Weather data comes from Nominatim (geocoding) and Open-Meteo (forecast).

## Run

```bash
cp .env.example .env        # then set MISTRAL_API_KEY
uv run mistral-iw           # http://127.0.0.1:8000
```

`MISTRAL_MODEL` defaults to `mistral-small-latest`. `HOST`, `PORT` and `TRACES_DIR` are optional.

## Test

```bash
uv run pytest
```

The tests run offline: Mistral, Nominatim and Open-Meteo are faked with `httpx.MockTransport`.

## Layout

| File | Concern |
| --- | --- |
| `src/mistral_iw/config.py` | Settings from `.env` and the environment |
| `src/mistral_iw/prompts.py` | System prompt |
| `src/mistral_iw/tools.py` | `geocode`, `get_forecast`, WMO codes, `@tool` registry |
| `src/mistral_iw/llm.py` | Streaming chat completions, 429 retries |
| `src/mistral_iw/agent.py` | Tool-calling loop, capped at 6 LLM calls |
| `src/mistral_iw/tracing.py` | Traces and spans, JSONL storage in `traces/YYYY-MM-DD.jsonl` |
| `src/mistral_iw/server.py` | FastAPI routes and static UI |
| `src/mistral_iw/static/index.html` | Chat UI and trace explorer |

## API

- `POST /api/chat {message, history}` → `{answer, trace}`
- `GET /api/health` → model name and whether the API key is set
- `GET /api/traces` → every saved trace without spans, newest first
- `GET /api/traces/{id}` → one full trace, or 404

## Trace metrics

Responses are streamed from Mistral, so time to first token is measured, not estimated.

- **TTFT**: time from sending an LLM request to its first streamed token, including any 429 back-off. A trace's TTFT is the mean over its LLM calls.
- **TPOT**: time per output token after the first one, summed over a trace's LLM calls.
- **Latency**: total time to answer one question.
- **Tokens / answer**: prompt and completion tokens over all LLM calls of a successful trace.
