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

Fetching prefers the listing-page discovery in scraper.page_scraper
(finds only dates that actually have a published brief) over blindly
requesting every calendar day in the range — the latter is the fallback,
used only if discovery is unavailable (e.g. no Chrome/Selenium locally)
or turns up nothing, so a wrong/changed listing path degrades to "slower"
rather than "silently skips the backfill."
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

from pipeline.ingest import get_latest_ingested_date, run_ingest
from scraper.pdf_downloader import ScrapeResult, download_pdf, download_range

load_dotenv()  # must run before the os.environ.get() call below

logger = logging.getLogger(__name__)

# How far back to backfill when the vector store is completely empty.
ARCHIVE_START_DATE = os.environ.get("HANSARD_ARCHIVE_START_DATE", "2017-01-01")

# discover_hansard_documents() only fetches the listing's first page (see
# its docstring) -- pagination isn't wired up yet. If the oldest date it
# found is this many days short of what we asked for, treat the result as
# incomplete rather than authoritative, so a wide backfill still falls
# back to date-range probing instead of silently stopping at page one.
_DISCOVERY_COVERAGE_SLACK_DAYS = 30


def resolve_sync_start_date() -> datetime:
    """The first date to (re-)check on this sync run."""
    latest = get_latest_ingested_date()
    if latest:
        return datetime.fromisoformat(latest) + timedelta(days=1)
    return datetime.fromisoformat(ARCHIVE_START_DATE)


def _download_via_listing_discovery(
    start_date: datetime, end_date: datetime
) -> ScrapeResult | None:
    """Try the listing-page discovery path; return None to signal "fall back"."""
    from scraper.pdf_downloader import parse_display_name_to_date

    try:
        from scraper.page_scraper import discover_hansard_documents, filter_by_date_range

        documents = filter_by_date_range(discover_hansard_documents(), start_date, end_date)
    except Exception:
        logger.warning(
            "Listing-page discovery failed; falling back to date-range probing", exc_info=True
        )
        return None

    if not documents:
        return None

    oldest_found = min(parse_display_name_to_date(doc.display_name) for doc in documents)
    slack = timedelta(days=_DISCOVERY_COVERAGE_SLACK_DAYS)
    if oldest_found.date() > (start_date + slack).date():
        logger.warning(
            "Listing discovery only reached back to %s (requested from %s); "
            "treating as incomplete and falling back to date-range probing",
            oldest_found.date(),
            start_date.date(),
        )
        return None

    result = ScrapeResult(started_at=datetime.utcnow())
    for doc in documents:
        result.urls_attempted.append(doc.url)
        try:
            saved_path = download_pdf(doc.url, doc.display_name)
            if saved_path is not None:
                result.documents_downloaded.append(str(saved_path))
        except Exception as exc:
            result.failures.append(f"{doc.url}: {exc}")
    result.finished_at = datetime.utcnow()
    return result


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
    scrape_result = _download_via_listing_discovery(start_date, end_date)
    if scrape_result is None:
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
