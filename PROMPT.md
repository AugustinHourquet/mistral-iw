# The single prompt

The prompt below rebuilds this app from an empty `uv` project.

---

Build a minimal Python 3.14 web app (uv, FastAPI, httpx) exposing an AI weather agent through a chat UI.

**Agent**
- Use the Mistral chat completions API over plain HTTP (no SDK, no agent framework), with function calling and `temperature=0`.
- Read `MISTRAL_API_KEY` and `MISTRAL_MODEL` (default `mistral-small-latest`) from `.env`.
- Write a hand-rolled loop: call the model, execute any `tool_calls`, append the results as `role: tool` messages with their `tool_call_id`, and repeat until a plain answer comes back.
- Cap the loop at 6 LLM calls. Retry on HTTP 429.

**Tools**
- `geocode(location)` calls Nominatim with a User-Agent header.
- `get_forecast(latitude, longitude, days=7, temperature_unit)` calls Open-Meteo with `timezone=auto` and returns:
  - current temperature, feels-like temperature, humidity, precipitation, wind and conditions
  - a compact daily list: date, conditions, min/max temperature, max precipitation probability, precipitation sum and max wind
- Map WMO weather codes to text.
- Tool errors are returned to the model as `{"error": ...}` and never raised.

**System prompt**
- Include today's date.
- Never answer from memory; always geocode and then fetch the forecast.
- Relative days are local to the location: `daily[0]` is today there.
- Answer the question first, then give one or two figures with units.
- Decline anything that isn't about weather.
- Reply in the user's language.

**Structure: one file per concern**
- `prompts.py` holds every system prompt.
- `tools.py` holds every tool, each with its JSON schema registered by an `@tool` decorator.
- `tracing.py` creates one trace per question, with spans for every LLM and tool call recording input, output, duration and token usage. Traces are returned to the client and appended to `traces/YYYY-MM-DD.jsonl`, and can be read back: a summary list (newest first, no spans) and one full trace by id.
- `llm.py`, `agent.py`, `config.py` and `server.py` hold the rest.

**API**
- `POST /api/chat {message, history}` returns `{answer, trace}`.
- `GET /api/health` reports the model and whether the API key is set.
- `GET /api/traces` lists every saved trace's metadata, newest first. `GET /api/traces/{id}` returns one full trace, or 404.
- The UI is served as static files.

**UI**
- One `index.html`: vanilla JS, no build step, light and dark mode, responsive.
- A chat with example chips ("Hey, how cold is it in Paris?", "Will it rain in New York tomorrow?").
- A typing indicator and minimal markdown rendering.
- Under each answer: the number of tool calls, latency, tokens and a "view trace" link.
- A side panel that opens on a menu of every saved trace, grouped by day, with a filter box. Each entry shows time, id, question, latency, LLM calls, tool calls, tokens and an error badge.
- Clicking a trace (or "view trace") shows it as a timeline of LLM and tool spans, with duration bars and expandable JSON input and output. A "← All traces" button and Esc return to the menu at the same scroll position.
- Build a menu where all passed trace can be explored. In this page display key KPIs like TTFT, TPOT, p95 total response time (latency) and average token per answer. 

**Tests**
- Offline pytest tests using `httpx.MockTransport`. They cover the full geocode → forecast → answer loop, tool errors fed back to the model, the step cap, and the trace history endpoints.
