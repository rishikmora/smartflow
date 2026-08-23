"""Read-only LLM analytics service API for SmartFlow."""

from __future__ import annotations

from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI(title="SmartFlow LLM Service")


class QueryRequest(BaseModel):
    """Natural-language query request."""

    question: str


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok", "service": "llm_service"}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, str]:
    """Return a read-only analytics response placeholder."""
    return {"answer": f"SmartFlow analytics is read-only. Received: {request.question}"}
