"""FastAPI app: chat API, trace history API and the static UI."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import run_agent
from .config import Settings
from .tracing import TraceStore

STATIC_DIR = Path(__file__).parent / "static"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=40)


def create_app(settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> FastAPI:
    settings = settings or Settings.load()
    store = TraceStore(settings.traces_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if client is not None:  # injected (tests): the caller owns it
            app.state.client = client
            yield
            return
        async with httpx.AsyncClient() as owned:
            app.state.client = owned
            yield

    app = FastAPI(title="Weather agent", lifespan=lifespan)

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> dict:
        if not settings.mistral_api_key:
            raise HTTPException(503, "MISTRAL_API_KEY is not set. Add it to .env and restart.")
        history = [t.model_dump() for t in req.history]
        result = await run_agent(app.state.client, settings, req.message.strip(), history)
        store.save(result.trace)
        return {"answer": result.answer, "trace": result.trace}

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "model": settings.mistral_model, "api_key_set": bool(settings.mistral_api_key)}

    @app.get("/api/traces")
    async def list_traces() -> list[dict]:
        return store.list()

    @app.get("/api/traces/{trace_id}")
    async def get_trace(trace_id: str) -> dict:
        trace = store.get(trace_id)
        if trace is None:
            raise HTTPException(404, f"Trace '{trace_id}' not found")
        return trace

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
