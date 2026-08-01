"""Dynamic HTML scraping of the Parliament of Ghana Hansard listing pages.

The Hansard PDF URLs follow a predictable, date-based naming convention
(see ``pdf_downloader.build_pdf_url``), but not every calendar day has a
sitting. This module drives a headless browser against the Parliament
website's document listing to discover which sitting dates actually have a
brief published, so the downloader doesn't have to blindly request a URL
for every day in a range.
"""

from __future__ import annotations

import logging
import os
import re
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
HANSARD_LISTING_PATH = os.environ.get("HANSARD_LISTING_PATH", "/publications/business-papers")
SCRAPE_DELAY_SECONDS = float(os.environ.get("SCRAPE_DELAY_SECONDS", "5"))

_PDF_LINK_RE = re.compile(r"/epanel/docs/pb/[^\"']+\.pdf", re.IGNORECASE)


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


def discover_hansard_documents(
    base_url: str = PARLIAMENT_BASE_URL,
    listing_path: str = HANSARD_LISTING_PATH,
) -> list[DiscoveredDocument]:
    """Render the Hansard listing page and extract every PDF link found.

    Returns an empty list (rather than raising) if the page loads but no
    matching links are found, since the listing page's structure is outside
    our control and may change without notice.
    """
    url = base_url.rstrip("/") + listing_path
    driver = _build_driver()
    try:
        html = _load_listing_page(driver, url)
    finally:
        driver.quit()

    matches = sorted(set(_PDF_LINK_RE.findall(html)))
    documents = [
        DiscoveredDocument(
            url=base_url.rstrip("/") + path,
            display_name=os.path.basename(path).replace("%20", " ").replace("%2C", ","),
        )
        for path in matches
    ]
    logger.info("Discovered %d Hansard document(s) on %s", len(documents), url)
    return documents


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
