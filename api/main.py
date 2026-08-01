"""FastAPI application exposing the Ghana Parliament Hansard RAG agent."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import run_agent
from pipeline.ingest import get_latest_ingested_date
from pipeline.retriever import count_documents
from pipeline.sync import sync_hansards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ghana Parliament Hansard RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# job_id -> status dict, for tracking background /ingest runs.
_INGEST_JOBS: dict[str, dict] = {}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


class QueryRequest(BaseModel):
    question: str
    date_filter: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    latest_sitting_date: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    job_id: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        documents_indexed = count_documents()
        latest_sitting_date = get_latest_ingested_date()
    except Exception:
        logger.exception("Failed to read index status")
        documents_indexed = 0
        latest_sitting_date = None
    return HealthResponse(
        status="ok", documents_indexed=documents_indexed, latest_sitting_date=latest_sitting_date
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        result = run_agent(request.question, date_filter=request.date_filter)
    except Exception as exc:
        logger.exception("Query failed for question=%r", request.question)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=str(uuid.uuid4()),
    )


def _run_ingest_job(job_id: str) -> None:
    _INGEST_JOBS[job_id]["status"] = "running"
    try:
        result = sync_hansards()
        _INGEST_JOBS[job_id].update(status="completed", **result)
    except Exception as exc:
        logger.exception("Ingest job %s failed", job_id)
        _INGEST_JOBS[job_id].update(status="failed", error=str(exc))


@app.post("/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    """Manually trigger the same incremental sync the daily job runs.

    No parameters: this always catches the index up to today, from
    wherever it last left off (or does a full backfill on an empty
    index). End users querying the app never need to call this — it's
    for ops/admin use when you don't want to wait for the next scheduled
    sync (see the `sync` service in docker-compose.yml).
    """
    job_id = str(uuid.uuid4())
    _INGEST_JOBS[job_id] = {"status": "started"}

    asyncio.get_event_loop().run_in_executor(None, _run_ingest_job, job_id)

    return IngestResponse(status="started", job_id=job_id)


@app.get("/ingest/{job_id}")
async def ingest_status(job_id: str) -> dict:
    return _INGEST_JOBS.get(job_id, {"status": "unknown"})


# Serves web/ (the static JS frontend) at "/", after every API route above
# so it never shadows them. Same files can be deployed as-is to Netlify —
# see web/app.js for the API_BASE setting when hosting it separately.
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
