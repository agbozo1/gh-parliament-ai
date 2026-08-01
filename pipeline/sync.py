"""Keep the vector store in sync with the Parliament's published Hansards.

One function, two uses:
- Empty vector store (nothing indexed yet) -> full backfill, from
  HANSARD_ARCHIVE_START_DATE through today.
- Non-empty vector store -> incremental sync, from the day after the most
  recently indexed sitting date through today.

This is what both the daily cron job (scripts/sync_hansards.py) and the
manual "sync now" API trigger (POST /ingest) call — end users never pick a
date range themselves; the store is expected to already be up to date by
the time anyone queries it.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

from pipeline.ingest import get_latest_ingested_date, run_ingest
from scraper.pdf_downloader import download_range

load_dotenv()  # must run before the os.environ.get() call below

logger = logging.getLogger(__name__)

# How far back to backfill when the vector store is completely empty.
ARCHIVE_START_DATE = os.environ.get("HANSARD_ARCHIVE_START_DATE", "2017-01-01")


def resolve_sync_start_date() -> datetime:
    """The first date to (re-)check on this sync run."""
    latest = get_latest_ingested_date()
    if latest:
        return datetime.fromisoformat(latest) + timedelta(days=1)
    return datetime.fromisoformat(ARCHIVE_START_DATE)


def sync_hansards() -> dict:
    """Download and index every sitting brief published since the last
    successful sync (or the full configured archive, on an empty store).

    Safe to call repeatedly/on a schedule: if nothing new has been
    published, this is a cheap no-op.
    """
    start_date = resolve_sync_start_date()
    end_date = datetime.utcnow()

    if start_date.date() > end_date.date():
        last_indexed = (start_date - timedelta(days=1)).date()
        logger.info("Vector store already up to date (through %s)", last_indexed)
        return {"documents_downloaded": 0, "chunks_written": 0, "up_to_date": True}

    logger.info("Syncing Hansards from %s to %s", start_date.date(), end_date.date())
    scrape_result = download_range(start_date, end_date)
    chunks_written = run_ingest()

    logger.info(
        "Sync complete: %d new document(s) downloaded, %d chunk(s) written",
        len(scrape_result.documents_downloaded),
        chunks_written,
    )
    return {
        "documents_downloaded": len(scrape_result.documents_downloaded),
        "chunks_written": chunks_written,
        "up_to_date": False,
    }
