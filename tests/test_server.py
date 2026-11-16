import json
from dataclasses import replace

from fastapi.testclient import TestClient

from conftest import FakeWeb, text_reply, tool_call_reply
from mistral_iw.server import create_app


def make_client(settings, web: FakeWeb) -> TestClient:
    return TestClient(create_app(settings, client=web.client()))


def test_health(settings):
    with make_client(settings, FakeWeb([])) as api:
        assert api.get("/api/health").json() == {
            "status": "ok", "model": "mistral-small-latest", "api_key_set": True,
        }


def test_chat_requires_api_key(settings):
    with make_client(replace(settings, mistral_api_key=""), FakeWeb([])) as api:
        assert api.get("/api/health").json()["api_key_set"] is False
        assert api.post("/api/chat", json={"message": "hi"}).status_code == 503


def test_chat_returns_answer_and_trace(settings):
    web = FakeWeb([
        tool_call_reply("geocode", {"location": "New York"}, "g"),
        tool_call_reply("get_forecast", {"latitude": 40.7, "longitude": -74.0}, "f"),
        text_reply("Yes, rain is likely tomorrow in New York (85%)."),
    ])
    with make_client(settings, web) as api:
        resp = api.post("/api/chat", json={"message": "Will it rain in New York tomorrow?", "history": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("Yes, rain")
    assert body["trace"]["question"] == "Will it rain in New York tomorrow?"
    assert len(body["trace"]["spans"]) == 5


def test_trace_history_endpoints(settings):
    web = FakeWeb([text_reply("First answer"), text_reply("Second answer")])
    with make_client(settings, web) as api:
        first = api.post("/api/chat", json={"message": "first?"}).json()["trace"]
        second = api.post("/api/chat", json={"message": "second?"}).json()["trace"]

        # Persisted to traces/YYYY-MM-DD.jsonl
        (path,) = settings.traces_dir.glob("*.jsonl")
        assert path.name == f"{first['started_at'][:10]}.jsonl"
        assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == [first["id"], second["id"]]

        # Summary list: newest first, no spans.
        listing = api.get("/api/traces").json()
        assert [t["id"] for t in listing] == [second["id"], first["id"]]
        assert all("spans" not in t for t in listing)
        assert listing[0]["question"] == "second?"
        assert {"latency_ms", "llm_calls", "tool_calls", "total_tokens", "ttft_ms", "tpot_ms", "error"} <= set(listing[0])

        # One full trace by id.
        full = api.get(f"/api/traces/{first['id']}").json()
        assert full["id"] == first["id"] and full["spans"] == first["spans"]

        assert api.get("/api/traces/does-not-exist").status_code == 404


def test_trace_list_spans_multiple_days_and_skips_corrupt_lines(settings):
    settings.traces_dir.mkdir(parents=True)
    old = {"id": "old", "started_at": "2026-01-01T10:00:00.000+01:00", "question": "q", "spans": []}
    new = {"id": "new", "started_at": "2026-01-02T09:00:00.000+01:00", "question": "q", "spans": []}
    (settings.traces_dir / "2026-01-01.jsonl").write_text(json.dumps(old) + "\n{corrupt\n")
    (settings.traces_dir / "2026-01-02.jsonl").write_text(json.dumps(new) + "\n")
    with make_client(settings, FakeWeb([])) as api:
        assert [t["id"] for t in api.get("/api/traces").json()] == ["new", "old"]
        assert api.get("/api/traces/old").json()["id"] == "old"


def test_ui_is_served(settings):
    with make_client(settings, FakeWeb([])) as api:
        resp = api.get("/")
    assert resp.status_code == 200 and "<html" in resp.text.lower()
