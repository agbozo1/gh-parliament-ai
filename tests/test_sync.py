from datetime import datetime, timedelta

from pipeline.sync import (
    ARCHIVE_START_DATE,
    _download_via_listing_discovery,
    resolve_sync_start_date,
    sync_hansards,
)
from scraper.page_scraper import DiscoveredDocument


def test_resolve_start_date_uses_archive_start_when_store_is_empty(mocker):
    mocker.patch("pipeline.sync.get_latest_ingested_date", return_value=None)

    start = resolve_sync_start_date()

    assert start == datetime.fromisoformat(ARCHIVE_START_DATE)


def test_resolve_start_date_resumes_day_after_latest_indexed(mocker):
    mocker.patch("pipeline.sync.get_latest_ingested_date", return_value="2025-02-11")

    start = resolve_sync_start_date()

    assert start == datetime(2025, 2, 12)


def test_sync_hansards_falls_back_to_date_range_when_discovery_unavailable(mocker):
    mocker.patch("pipeline.sync.get_latest_ingested_date", return_value=None)
    mocker.patch("pipeline.sync._download_via_listing_discovery", return_value=None)
    scrape_result = mocker.Mock(documents_downloaded=["a.pdf", "b.pdf"])
    download_mock = mocker.patch("pipeline.sync.download_range", return_value=scrape_result)
    run_ingest_mock = mocker.patch("pipeline.sync.run_ingest", return_value=5)

    result = sync_hansards()

    download_mock.assert_called_once()
    run_ingest_mock.assert_called_once()
    assert result == {"documents_downloaded": 2, "chunks_written": 5, "up_to_date": False}


def test_sync_hansards_prefers_listing_discovery_when_it_finds_documents(mocker):
    mocker.patch("pipeline.sync.get_latest_ingested_date", return_value=None)
    scrape_result = mocker.Mock(documents_downloaded=["a.pdf"])
    discovery_mock = mocker.patch(
        "pipeline.sync._download_via_listing_discovery", return_value=scrape_result
    )
    download_range_mock = mocker.patch("pipeline.sync.download_range")
    run_ingest_mock = mocker.patch("pipeline.sync.run_ingest", return_value=3)

    result = sync_hansards()

    discovery_mock.assert_called_once()
    download_range_mock.assert_not_called()
    run_ingest_mock.assert_called_once()
    assert result == {"documents_downloaded": 1, "chunks_written": 3, "up_to_date": False}


def test_sync_hansards_is_a_noop_when_already_up_to_date(mocker):
    tomorrow_indexed = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
    mocker.patch("pipeline.sync.get_latest_ingested_date", return_value=tomorrow_indexed)
    discovery_mock = mocker.patch("pipeline.sync._download_via_listing_discovery")
    download_mock = mocker.patch("pipeline.sync.download_range")
    run_ingest_mock = mocker.patch("pipeline.sync.run_ingest")

    result = sync_hansards()

    discovery_mock.assert_not_called()
    download_mock.assert_not_called()
    run_ingest_mock.assert_not_called()
    assert result["up_to_date"] is True


def test_listing_discovery_returns_none_when_discovery_raises(mocker):
    mocker.patch(
        "scraper.page_scraper.discover_hansard_documents", side_effect=RuntimeError("no firefox")
    )

    result = _download_via_listing_discovery(datetime(2025, 1, 1), datetime(2025, 1, 31))

    assert result is None


def test_listing_discovery_returns_none_when_nothing_matches_the_range(mocker):
    mocker.patch(
        "scraper.page_scraper.discover_hansard_documents",
        return_value=[
            DiscoveredDocument(url="https://x/1.pdf", display_name="11th February, 2025")
        ],
    )

    # Range doesn't include February 2025.
    result = _download_via_listing_discovery(datetime(2025, 1, 1), datetime(2025, 1, 31))

    assert result is None


def test_listing_discovery_downloads_documents_matched_in_range(mocker, tmp_path):
    mocker.patch(
        "scraper.page_scraper.discover_hansard_documents",
        return_value=[
            DiscoveredDocument(
                url="https://www.parliament.gh/epanel/docs/pb/11th%20February%2C%202025.pdf",
                display_name="11th February, 2025",
            ),
            DiscoveredDocument(
                url="https://www.parliament.gh/epanel/docs/pb/1st%20March%2C%202025.pdf",
                display_name="1st March, 2025",
            ),
        ],
    )
    download_pdf_mock = mocker.patch(
        "pipeline.sync.download_pdf", return_value=tmp_path / "saved.pdf"
    )

    result = _download_via_listing_discovery(datetime(2025, 2, 1), datetime(2025, 2, 28))

    assert download_pdf_mock.call_count == 1
    assert len(result.documents_downloaded) == 1
    assert len(result.urls_attempted) == 1
