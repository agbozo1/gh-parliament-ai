"""Dynamic HTML scraping of the Parliament of Ghana Hansard listing pages.

The Hansard PDF URLs follow a predictable, date-based naming convention
(see ``pdf_downloader.build_pdf_url``), but not every calendar day has a
sitting. This module drives a headless browser against the Parliament
website's document listing (https://www.parliament.gh/docs?type=HS) to
discover which sitting dates actually have a brief published, so the
downloader doesn't have to blindly request a URL for every day in a range.

Each row's date comes from a ``showPDF('pb/<name>.pdf', '<title>')`` click
handler rather than a plain link. We only use that to extract a *date* —
the actual URL we fetch is always re-derived via ``pdf_downloader.
build_pdf_url``, a pattern already confirmed working, rather than trusting
how ``showPDF()`` itself resolves the relative path (unknown, and not
worth guessing wrong). This also naturally dedupes the listing's own
inconsistencies: the same sitting sometimes appears twice under slightly
different text (missing comma, a "(1)" correction suffix), but always
collapses to the same canonical date.

The listing is paginated via a ``P=<offset>`` query parameter in steps of
``PAGE_SIZE`` (e.g. ``?type=HS&P=50`` is page 2). ``discover_hansard_documents``
walks pages, newest-first, stopping as soon as it's covered back to
``since`` — so a daily incremental check (``since`` = yesterday) only ever
reads page 1, while a full historical backfill (``since`` = years ago)
walks back exactly as many pages as it takes and no further.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()  # must run before the os.environ.get() calls below

logger = logging.getLogger(__name__)

PARLIAMENT_BASE_URL = os.environ.get("PARLIAMENT_BASE_URL", "https://www.parliament.gh")
HANSARD_LISTING_PATH = os.environ.get("HANSARD_LISTING_PATH", "/docs?type=HS")
SCRAPE_DELAY_SECONDS = float(os.environ.get("SCRAPE_DELAY_SECONDS", "5"))

PAGE_SIZE = 50
# Safety bound on how many pages to walk back in one call, independent of
# `since` -- the real archive was ~42 pages (~2,100 docs) when last
# checked; this leaves generous headroom without risking an unbounded loop
# if pagination behavior ever changes unexpectedly.
MAX_PAGES = 80

# Row click handlers look like: showPDF('pb/24th July, 2026.pdf', 'Hansard ...')
_SHOWPDF_RE = re.compile(r"showPDF\(\s*'([^']+?\.pdf)'", re.IGNORECASE)


@dataclass
class DiscoveredDocument:
    url: str
    display_name: str


def _build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    return webdriver.Chrome(options=options)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(WebDriverException),
)
def _load_listing_page(driver: webdriver.Chrome, url: str) -> str:
    logger.info("Loading listing page: %s", url)
    driver.get(url)
    return driver.page_source


def _listing_page_url(base_url: str, listing_path: str, offset: int) -> str:
    if offset <= 0:
        return base_url.rstrip("/") + listing_path
    separator = "&" if "?" in listing_path else "?"
    return f"{base_url.rstrip('/')}{listing_path}{separator}P={offset}"


def _parse_documents(html: str, base_url: str) -> list[DiscoveredDocument]:
    """Extract canonical documents from one listing page's HTML."""
    from .pdf_downloader import build_pdf_url, parse_display_name_to_date

    documents: dict[str, DiscoveredDocument] = {}
    for raw_path in _SHOWPDF_RE.findall(html):
        raw_name = os.path.basename(raw_path)
        if raw_name.lower().endswith(".pdf"):
            raw_name = raw_name[: -len(".pdf")]

        parsed_date = parse_display_name_to_date(raw_name)
        if parsed_date is None:
            logger.warning("Could not parse a date from listing entry %r; skipping", raw_name)
            continue

        # Re-derive the canonical (url, display_name) instead of trusting
        # the site's own text — see module docstring.
        canonical_url, canonical_name = build_pdf_url(parsed_date, base_url=base_url)
        documents[canonical_name] = DiscoveredDocument(
            url=canonical_url, display_name=canonical_name
        )
    return sorted(documents.values(), key=lambda doc: doc.display_name)


def discover_hansard_documents(
    base_url: str = PARLIAMENT_BASE_URL,
    listing_path: str = HANSARD_LISTING_PATH,
    since: datetime | None = None,
    delay_seconds: float = SCRAPE_DELAY_SECONDS,
) -> list[DiscoveredDocument]:
    """Discover real sitting dates from the Hansard listing.

    With `since` omitted, fetches only the first (newest) page. With
    `since` given, walks back page by page — reusing one browser session —
    until either a page's oldest date is before `since`, a page turns up
    no documents at all (end of the real archive), or MAX_PAGES is hit.
    Listing pages are newest-first, so this naturally stops early for a
    recent `since` (a daily sync reads page 1 only) and walks further back
    only when there's actually more history to cover.

    Returns an empty list (rather than raising) if a page loads but no
    matching rows are found, since the listing page's structure is outside
    our control and may change without notice.
    """
    from .pdf_downloader import parse_display_name_to_date

    driver = _build_driver()
    all_documents: dict[str, DiscoveredDocument] = {}
    pages_read = 0
    try:
        for page_index in range(MAX_PAGES if since is not None else 1):
            offset = page_index * PAGE_SIZE
            url = _listing_page_url(base_url, listing_path, offset)
            html = _load_listing_page(driver, url)
            page_documents = _parse_documents(html, base_url)
            pages_read += 1

            if not page_documents:
                logger.info("No documents found at %s; stopping pagination", url)
                break

            for doc in page_documents:
                all_documents[doc.display_name] = doc

            if since is None:
                break

            oldest_on_page = min(
                parse_display_name_to_date(doc.display_name) for doc in page_documents
            )
            if oldest_on_page < since:
                break

            if page_index + 1 < MAX_PAGES and delay_seconds:
                time.sleep(delay_seconds)
    finally:
        driver.quit()

    result = sorted(all_documents.values(), key=lambda doc: doc.display_name)
    logger.info("Discovered %d Hansard document(s) across %d page(s)", len(result), pages_read)
    return result


def filter_by_date_range(
    documents: Iterable[DiscoveredDocument],
    start_date: datetime,
    end_date: datetime,
) -> list[DiscoveredDocument]:
    """Keep only documents whose filename parses to a date within range."""
    from .pdf_downloader import parse_display_name_to_date

    filtered = []
    for doc in documents:
        parsed = parse_display_name_to_date(doc.display_name)
        if parsed and start_date.date() <= parsed.date() <= end_date.date():
            filtered.append(doc)
    return filtered
