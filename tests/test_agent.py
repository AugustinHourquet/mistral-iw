import asyncio
import json

import httpx

from conftest import FakeWeb, text_reply, tool_call_reply
from mistral_iw.agent import STEP_CAP_ANSWER, run_agent
from mistral_iw.tools import describe_weather_code, execute_tool

PARIS = {"latitude": 48.8589, "longitude": 2.32}


def run(web: FakeWeb, settings, message: str, history=None):
    async def go():
        async with web.client() as client:
            return await run_agent(client, settings, message, history)

    return asyncio.run(go())


def test_full_geocode_forecast_answer_loop(settings):
    web = FakeWeb([
        tool_call_reply("geocode", {"location": "Paris, France"}, "call_geo"),
        tool_call_reply("get_forecast", PARIS, "call_fc"),
        text_reply("It's chilly in Paris: 6.2 °C, feels like 3.1 °C."),
    ])
    result = run(web, settings, "Hey, how cold is it in Paris?")

    assert result.answer == "It's chilly in Paris: 6.2 °C, feels like 3.1 °C."

    # Tools hit the real APIs' endpoints with the required parameters.
    (geo,) = web.to("nominatim.openstreetmap.org")
    assert geo.url.params["q"] == "Paris, France"
    assert "mistral-iw" in geo.headers["user-agent"]
    (fc,) = web.to("api.open-meteo.com")
    assert fc.url.params["timezone"] == "auto"
    assert fc.url.params["forecast_days"] == "7"
    assert fc.url.params["temperature_unit"] == "celsius"

    # Every LLM call: temperature 0, tools offered, system prompt carries today's date.
    payloads = web.mistral_payloads()
    assert len(payloads) == 3
    assert all(p["temperature"] == 0 and p["model"] == "mistral-small-latest" for p in payloads)
    assert {t["function"]["name"] for t in payloads[0]["tools"]} == {"geocode", "get_forecast"}
    from datetime import date
    assert date.today().isoformat() in payloads[0]["messages"][0]["content"]

    # Tool results go back as role=tool messages with the matching tool_call_id.
    final_msgs = payloads[2]["messages"]
    tool_msgs = [m for m in final_msgs if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_geo", "call_fc"]
    geo_out = json.loads(tool_msgs[0]["content"])
    assert geo_out["latitude"] == 48.8589 and geo_out["country_code"] == "FR"
    forecast = json.loads(tool_msgs[1]["content"])
    assert forecast["current"]["conditions"] == "overcast"
    assert forecast["current"]["feels_like"] == 3.1
    assert forecast["daily"][1] == {
        "date": "2026-11-17", "conditions": "slight rain", "temp_min": 4.0, "temp_max": 10.5,
        "precip_probability_max": 85, "precip_sum": 6.4, "wind_max": 27.5,
    }

    # Trace: 3 LLM spans + 2 tool spans, tokens summed, TTFT/TPOT measured.
    trace = result.trace
    assert [s["kind"] for s in trace["spans"]] == ["llm", "tool", "llm", "tool", "llm"]
    assert (trace["llm_calls"], trace["tool_calls"]) == (3, 2)
    assert trace["total_tokens"] == 120 + 120 + 215
    assert trace["error"] is None and trace["span_errors"] == 0
    assert trace["ttft_ms"] is not None and trace["tpot_ms"] is not None
    assert all(s["duration_ms"] >= 0 for s in trace["spans"])
    assert trace["spans"][1]["input"] == {"location": "Paris, France"}


def test_history_is_forwarded(settings):
    web = FakeWeb([text_reply("Which city do you mean?")])
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]
    run(web, settings, "weather?", history)
    roles = [m["role"] for m in web.mistral_payloads()[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_tool_errors_are_fed_back_to_the_model(settings):
    web = FakeWeb([
        tool_call_reply("geocode", {"location": "Atlantis"}, "c1"),
        tool_call_reply("get_forecast", PARIS, "c2"),
        tool_call_reply("teleport", {}, "c3"),
        text_reply("Sorry, I couldn't find that place."),
    ])
    web.nominatim = lambda r: httpx.Response(200, json=[])
    web.open_meteo = lambda r: httpx.Response(500, json={"reason": "boom"})

    result = run(web, settings, "Weather in Atlantis?")

    assert result.answer == "Sorry, I couldn't find that place."
    tool_msgs = [m for m in web.mistral_payloads()[-1]["messages"] if m["role"] == "tool"]
    errors = [json.loads(m["content"]) for m in tool_msgs]
    assert [list(e) for e in errors] == [["error"], ["error"], ["error"]]
    assert "Atlantis" in errors[0]["error"]
    assert "HTTP 500" in errors[1]["error"]
    assert "Unknown tool" in errors[2]["error"]
    assert result.trace["span_errors"] == 3
    assert result.trace["error"] is None  # the run itself succeeded


def test_execute_tool_never_raises_on_bad_arguments():
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))) as c:
            return [
                await execute_tool(c, "geocode", "{not json"),
                await execute_tool(c, "geocode", "{}"),
                await execute_tool(c, "get_forecast", '{"latitude": 999, "longitude": 0}'),
            ]

    for out in asyncio.run(go()):
        assert set(out) == {"error"}


def test_step_cap_stops_after_six_llm_calls(settings):
    web = FakeWeb([tool_call_reply("geocode", {"location": "Paris"}, f"c{i}") for i in range(10)])
    result = run(web, settings, "Weather in Paris?")

    assert len(web.to("api.mistral.ai")) == 6
    assert result.answer == STEP_CAP_ANSWER
    assert result.trace["llm_calls"] == 6 and result.trace["tool_calls"] == 6
    assert "Step cap" in result.trace["error"]


def test_retries_on_429(settings):
    web = FakeWeb([
        httpx.Response(429, json={"message": "rate limited"}),
        httpx.Response(429, json={"message": "rate limited"}, headers={"retry-after": "0"}),
        text_reply("Which city?"),
    ])
    result = run(web, settings, "weather?")
    assert result.answer == "Which city?"
    assert len(web.to("api.mistral.ai")) == 3


def test_llm_http_error_is_reported_in_trace(settings):
    web = FakeWeb([httpx.Response(401, json={"message": "Unauthorized"})])
    result = run(web, settings, "weather?")
    assert "HTTP 401" in result.trace["error"]
    assert result.trace["spans"][0]["error"]


def test_streamed_tool_call_fragments_are_merged(settings):
    from conftest import sse

    body = sse(
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "geocode", "arguments": '{"loca'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'tion": "Paris"}'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "b", "function": {"name": "geocode", "arguments": '{"location": "Rome"}'}}]},
            "finish_reason": "tool_calls"}]},
    )
    web = FakeWeb([httpx.Response(200, content=body), text_reply("done")])
    run(web, settings, "Paris and Rome?")
    assistant = web.mistral_payloads()[1]["messages"][2]
    assert [(c["id"], json.loads(c["function"]["arguments"])) for c in assistant["tool_calls"]] == [
        ("a", {"location": "Paris"}),
        ("b", {"location": "Rome"}),
    ]


def test_wmo_codes():
    assert describe_weather_code(0) == "clear sky"
    assert describe_weather_code(95) == "thunderstorm"
    assert "unknown" in describe_weather_code(42)
    assert describe_weather_code(None) == "unknown"
