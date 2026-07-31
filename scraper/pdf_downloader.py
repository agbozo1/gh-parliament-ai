"""Download and store Ghana Parliament Hansard PDFs.

The Parliament site publishes each sitting's brief at a predictable URL
derived from the sitting date (e.g. ``.../pb/11th%20February%2C%202025.pdf``).
This module builds that URL, downloads the PDF with retry/backoff, and logs
every scrape run.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

PARLIAMENT_BASE_URL = os.environ.get("PARLIAMENT_BASE_URL", "https://www.parliament.gh")
PDF_OUTPUT_DIR = os.environ.get("PDF_OUTPUT_DIR", "data/raw")
SCRAPE_DELAY_SECONDS = float(os.environ.get("SCRAPE_DELAY_SECONDS", "2"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))

_MONTH_NAMES = (
    "January February March April May June July August September October November December"
).split()
_DISPLAY_NAME_RE = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)\s+(" + "|".join(_MONTH_NAMES) + r"),\s+(\d{4})"
)


@dataclass
class ScrapeResult:
    """Outcome of a single scrape run, ready for logging/inspection."""

    started_at: datetime
    finished_at: datetime | None = None
    urls_attempted: list[str] = field(default_factory=list)
    documents_downloaded: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "urls_attempted": self.urls_attempted,
            "documents_downloaded": self.documents_downloaded,
            "failures": self.failures,
        }


def _day_suffix(day: int) -> str:
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def build_pdf_url(date_obj: datetime, base_url: str = PARLIAMENT_BASE_URL) -> tuple[str, str]:
    """Return (url, display_name) for a sitting date's Hansard PDF."""
    day = date_obj.day
    month = date_obj.strftime("%B")
    year = date_obj.year
    display_name = f"{day}{_day_suffix(day)} {month}, {year}"
    encoded = display_name.replace(" ", "%20").replace(",", "%2C")
    url = f"{base_url.rstrip('/')}/epanel/docs/pb/{encoded}.pdf"
    return url, display_name


def parse_display_name_to_date(display_name: str) -> datetime | None:
    """Parse a filename like '11th February, 2025' back into a datetime."""
    match = _DISPLAY_NAME_RE.search(display_name)
    if not match:
        return None
    day, month_name, year = match.groups()
    month = _MONTH_NAMES.index(month_name) + 1
    return datetime(int(year), month, int(day))


def _is_retryable(exc: BaseException) -> bool:
    """Retry connection/timeout errors and 5xx responses; not 404s or other 4xx."""
    if isinstance(exc, requests.exceptions.HTTPError):
        return exc.response is not None and exc.response.status_code >= 500
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _fetch(url: str) -> requests.Response:
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response


def download_pdf(url: str, display_name: str, output_dir: str = PDF_OUTPUT_DIR) -> Path | None:
    """Download a single PDF, retrying on transient failures.

    Returns the saved path, or None if the document doesn't exist (404s are
    not retried — a missing sitting for that date is expected, not an error).
    """
    os.makedirs(output_dir, exist_ok=True)
    dest = Path(output_dir) / f"{display_name}.pdf"

    try:
        response = _fetch(url)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.info("No sitting document at %s (404)", url)
            return None
        raise
    except requests.exceptions.RequestException:
        logger.error("Failed to download %s after retries", url, exc_info=True)
        raise

    dest.write_bytes(response.content)
    logger.info("Saved %s", dest)
    return dest


def download_range(
    start_date: datetime,
    end_date: datetime,
    output_dir: str = PDF_OUTPUT_DIR,
    base_url: str = PARLIAMENT_BASE_URL,
    delay_seconds: float = SCRAPE_DELAY_SECONDS,
) -> ScrapeResult:
    """Download every available Hansard PDF between two dates (inclusive)."""
    import time

    result = ScrapeResult(started_at=datetime.utcnow())
    current = start_date
    while current <= end_date:
        url, display_name = build_pdf_url(current, base_url=base_url)
        result.urls_attempted.append(url)
        try:
            saved_path = download_pdf(url, display_name, output_dir=output_dir)
            if saved_path is not None:
                result.documents_downloaded.append(str(saved_path))
        except requests.exceptions.RequestException as exc:
            result.failures.append(f"{url}: {exc}")
        current += timedelta(days=1)
        if delay_seconds:
            time.sleep(delay_seconds)

    result.finished_at = datetime.utcnow()
    logger.info(
        "Scrape run complete: %d attempted, %d downloaded, %d failed",
        len(result.urls_attempted),
        len(result.documents_downloaded),
        len(result.failures),
    )
    return result
