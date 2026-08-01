"""FastAPI application exposing the Ghana Parliament Hansard RAG agent."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
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


class IngestRequest(BaseModel):
    # Preferred: an explicit period, ISO format (YYYY-MM-DD). If both are
    # set, every sitting date in [start_date, end_date] is scraped and
    # ingested. Falls back to the last `days_back` days when omitted.
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days_back: int = 30


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


def _parse_ingest_date(value: str, field_name: str) -> datetime:
    try:
        return datetime.combine(date.fromisoformat(value), datetime.min.time())
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"{field_name} must be an ISO date (YYYY-MM-DD), got {value!r}"
        ) from None


def _resolve_ingest_range(payload: IngestRequest) -> tuple[datetime, datetime]:
    """Resolve the (start_date, end_date) period to scrape for a request.

    Either both start_date and end_date must be set (an explicit period),
    or neither (falls back to the last `days_back` days).
    """
    if payload.start_date or payload.end_date:
        if not (payload.start_date and payload.end_date):
            raise HTTPException(
                status_code=400, detail="Provide both start_date and end_date, or neither."
            )
        start_date = _parse_ingest_date(payload.start_date, "start_date")
        end_date = _parse_ingest_date(payload.end_date, "end_date")
        if end_date < start_date:
            raise HTTPException(status_code=400, detail="end_date must be on or after start_date.")
        return start_date, end_date

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=payload.days_back)
    return start_date, end_date


def _run_ingest_job(job_id: str, start_date: datetime, end_date: datetime) -> None:
    from pipeline.ingest import run_ingest

    _INGEST_JOBS[job_id]["status"] = "running"
    try:
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
async def ingest(payload: IngestRequest) -> IngestResponse:
    start_date, end_date = _resolve_ingest_range(payload)

    job_id = str(uuid.uuid4())
    _INGEST_JOBS[job_id] = {"status": "started"}

    asyncio.get_event_loop().run_in_executor(None, _run_ingest_job, job_id, start_date, end_date)

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
