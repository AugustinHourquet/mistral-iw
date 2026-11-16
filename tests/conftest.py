"""Offline fakes for Mistral, Nominatim and Open-Meteo behind one httpx.MockTransport."""

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from mistral_iw.config import Settings

NOMINATIM_PARIS = [{
    "display_name": "Paris, Île-de-France, France",
    "lat": "48.8588897",
    "lon": "2.3200410",
    "address": {"country": "France", "country_code": "fr"},
}]

OPEN_METEO_PARIS = {
    "timezone": "Europe/Paris",
    "current": {
        "time": "2026-11-16T22:45",
        "temperature_2m": 6.2,
        "apparent_temperature": 3.1,
        "relative_humidity_2m": 81,
        "precipitation": 0.0,
        "weather_code": 3,
        "wind_speed_10m": 12.4,
        "wind_gusts_10m": 25.0,
    },
    "daily": {
        "time": ["2026-11-16", "2026-11-17"],
        "weather_code": [3, 61],
        "temperature_2m_min": [2.1, 4.0],
        "temperature_2m_max": [9.3, 10.5],
        "precipitation_probability_max": [10, 85],
        "precipitation_sum": [0.0, 6.4],
        "wind_speed_10m_max": [18.0, 27.5],
    },
}


def sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def tool_call_reply(name: str, args: dict, call_id: str = "call_1", tokens: int = 20) -> httpx.Response:
    return httpx.Response(200, content=sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"id": call_id, "index": 0, "function": {"name": name, "arguments": json.dumps(args)}}
        ]}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": tokens, "total_tokens": 100 + tokens}},
    ), headers={"content-type": "text/event-stream"})


def text_reply(text: str, tokens: int = 15) -> httpx.Response:
    half = len(text) // 2
    return httpx.Response(200, content=sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text[:half]}}]},
        {"choices": [{"index": 0, "delta": {"content": text[half:]}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 200, "completion_tokens": tokens, "total_tokens": 200 + tokens}},
    ), headers={"content-type": "text/event-stream"})


class FakeWeb:
    """Routes requests by host. Mistral replies are scripted, in order."""

    def __init__(self, mistral_replies: list[httpx.Response | Callable[[httpx.Request], httpx.Response]]):
        self.mistral_replies = list(mistral_replies)
        self.requests: list[httpx.Request] = []
        self.nominatim: Callable[[httpx.Request], httpx.Response] = lambda r: httpx.Response(200, json=NOMINATIM_PARIS)
        self.open_meteo: Callable[[httpx.Request], httpx.Response] = lambda r: httpx.Response(200, json=OPEN_METEO_PARIS)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        match request.url.host:
            case "api.mistral.ai":
                reply = self.mistral_replies.pop(0)
                return reply(request) if callable(reply) else reply
            case "nominatim.openstreetmap.org":
                return self.nominatim(request)
            case "api.open-meteo.com":
                return self.open_meteo(request)
        return httpx.Response(404)

    def to(self, host: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == host]

    def mistral_payloads(self) -> list[dict]:
        return [json.loads(r.content) for r in self.to("api.mistral.ai")]

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(mistral_api_key="test-key", traces_dir=tmp_path / "traces", retry_base_delay=0)
