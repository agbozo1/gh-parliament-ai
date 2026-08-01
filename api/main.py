"""FastAPI application exposing the Ghana Parliament Hansard RAG agent."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import run_agent
from pipeline.retriever import count_documents
from scraper.pdf_downloader import download_range

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


class IngestResponse(BaseModel):
    status: str
    job_id: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        documents_indexed = count_documents()
    except Exception:
        logger.exception("Failed to count indexed documents")
        documents_indexed = 0
    return HealthResponse(status="ok", documents_indexed=documents_indexed)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    result = run_agent(request.question, date_filter=request.date_filter)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=str(uuid.uuid4()),
    )


def _run_ingest_job(job_id: str, days_back: int) -> None:
    from pipeline.ingest import run_ingest

    _INGEST_JOBS[job_id]["status"] = "running"
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        scrape_result = download_range(start_date, end_date)
        chunks_written = run_ingest()
        _INGEST_JOBS[job_id].update(
            status="completed",
            documents_downloaded=len(scrape_result.documents_downloaded),
            chunks_written=chunks_written,
        )
    except Exception as exc:
        logger.exception("Ingest job %s failed", job_id)
        _INGEST_JOBS[job_id].update(status="failed", error=str(exc))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request) -> IngestResponse:
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    days_back = int(body.get("days_back", 30)) if isinstance(body, dict) else 30

    job_id = str(uuid.uuid4())
    _INGEST_JOBS[job_id] = {"status": "started"}

    import asyncio

    asyncio.get_event_loop().run_in_executor(None, _run_ingest_job, job_id, days_back)

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
