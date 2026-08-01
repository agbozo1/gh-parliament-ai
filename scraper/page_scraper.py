"""Dynamic HTML scraping of the Parliament of Ghana Hansard listing pages.

Not every calendar day has a sitting, and the site's PDF-naming convention
has changed over the years (recent files are named like "24th July,
2026.pdf"; older ones like "24may2005.pdf", with no ordinal/comma/spaces at
all) -- so a URL can't be reliably guessed from a date alone across the
whole archive (see ``pdf_downloader.build_pdf_url``'s docstring). This
module drives a headless browser against the Parliament website's document
listing (https://www.parliament.gh/docs?type=HS) to discover the exact,
real path for each published sitting instead.

Each row's real relative path comes from a
``showPDF('pb/<name>.pdf', '<title>')`` click handler rather than a plain
link; that path is exactly what gets fetched (confirmed against the site's
real URLs), so we use it verbatim rather than reconstructing anything. We
additionally parse a *date* out of it purely for identity/dedup: the same
sitting sometimes appears twice in the listing under slightly different
raw text (missing comma, a "(1)" correction suffix), and collapsing by
date -- rather than by the raw filename -- keeps that from producing
duplicate entries.

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
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service as FirefoxService
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()  # must run before the os.environ.get() calls below

logger = logging.getLogger(__name__)

PARLIAMENT_BASE_URL = os.environ.get("PARLIAMENT_BASE_URL", "https://www.parliament.gh")
HANSARD_LISTING_PATH = os.environ.get("HANSARD_LISTING_PATH", "/docs?type=HS")
SCRAPE_DELAY_SECONDS = float(os.environ.get("SCRAPE_DELAY_SECONDS", "5"))

# Set by the Docker image (see Dockerfile), which installs a matched
# firefox-esr + geckodriver pair instead of relying on Selenium Manager to
# auto-detect/download one -- keeps the container's browser and driver
# versions in lockstep regardless of what's on the host. Unset for local
# dev, where Selenium Manager's auto-detection is fine. We use Firefox
# rather than Chrome here because Chrome/chromedriver drop support for
# older OS versions (e.g. macOS Catalina) much sooner than Firefox does.
FIREFOX_BIN = os.environ.get("FIREFOX_BIN")
GECKODRIVER_PATH = os.environ.get("GECKODRIVER_PATH")

PAGE_SIZE = 50
# Safety bound on how many pages to walk back in one call, independent of
# `since` -- the real archive was ~42 pages (~2,100 docs) covering roughly
# 2017-2026 when last checked. HANSARD_ARCHIVE_START_DATE now reaches back
# to 2005, more than double that span, so this leaves generous headroom
# without risking an unbounded loop if pagination behavior ever changes
# unexpectedly.
MAX_PAGES = 200

# Row click handlers look like: showPDF('pb/24th July, 2026.pdf', 'Hansard ...')
_SHOWPDF_RE = re.compile(r"showPDF\(\s*'([^']+?\.pdf)'", re.IGNORECASE)


@dataclass
class DiscoveredDocument:
    url: str
    display_name: str


def _build_driver() -> webdriver.Firefox:
    options = Options()
    options.add_argument("--headless")
    if FIREFOX_BIN:
        options.binary_location = FIREFOX_BIN
    if GECKODRIVER_PATH:
        service = FirefoxService(executable_path=GECKODRIVER_PATH)
        return webdriver.Firefox(options=options, service=service)
    return webdriver.Firefox(options=options)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(WebDriverException),
)
def _load_listing_page(driver: webdriver.Firefox, url: str) -> str:
    logger.info("Loading listing page: %s", url)
    driver.get(url)
    return driver.page_source


def _listing_page_url(base_url: str, listing_path: str, offset: int) -> str:
    if offset <= 0:
        return base_url.rstrip("/") + listing_path
    separator = "&" if "?" in listing_path else "?"
    return f"{base_url.rstrip('/')}{listing_path}{separator}P={offset}"


def _parse_documents(html: str, base_url: str) -> list[DiscoveredDocument]:
    """Extract documents from one listing page's HTML."""
    from urllib.parse import quote

    from .pdf_downloader import format_display_name, parse_display_name_to_date

    documents: dict[str, DiscoveredDocument] = {}
    for raw_path in _SHOWPDF_RE.findall(html):
        raw_name = os.path.basename(raw_path)
        if raw_name.lower().endswith(".pdf"):
            raw_name = raw_name[: -len(".pdf")]

        parsed_date = parse_display_name_to_date(raw_name)
        if parsed_date is None:
            logger.warning("Could not parse a date from listing entry %r; skipping", raw_name)
            continue

        # Fetch the site's own raw path verbatim (it's the real URL -- see
        # module docstring) but key identity/dedup off the parsed date via
        # a stable display name, since the raw text isn't consistent.
        canonical_name = format_display_name(parsed_date)
        url = f"{base_url.rstrip('/')}/epanel/docs/{quote(raw_path, safe='/')}"
        documents[canonical_name] = DiscoveredDocument(url=url, display_name=canonical_name)
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
