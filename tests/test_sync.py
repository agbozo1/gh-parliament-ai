from datetime import datetime, timedelta

from pipeline.sync import ARCHIVE_START_DATE, resolve_sync_start_date, sync_hansards


def test_resolve_start_date_uses_archive_start_when_store_is_empty(mocker):
    mocker.patch("pipeline.sync.get_latest_ingested_date", return_value=None)

    start = resolve_sync_start_date()

    assert start == datetime.fromisoformat(ARCHIVE_START_DATE)


def test_resolve_start_date_resumes_day_after_latest_indexed(mocker):
    mocker.patch("pipeline.sync.get_latest_ingested_date", return_value="2025-02-11")

    start = resolve_sync_start_date()

    assert start == datetime(2025, 2, 12)


def test_sync_hansards_downloads_and_ingests_when_behind(mocker):
    mocker.patch("pipeline.sync.get_latest_ingested_date", return_value=None)
    scrape_result = mocker.Mock(documents_downloaded=["a.pdf", "b.pdf"])
    download_mock = mocker.patch("pipeline.sync.download_range", return_value=scrape_result)
    run_ingest_mock = mocker.patch("pipeline.sync.run_ingest", return_value=5)

    result = sync_hansards()

    download_mock.assert_called_once()
    run_ingest_mock.assert_called_once()
    assert result == {"documents_downloaded": 2, "chunks_written": 5, "up_to_date": False}


def test_sync_hansards_is_a_noop_when_already_up_to_date(mocker):
    tomorrow_indexed = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
    mocker.patch("pipeline.sync.get_latest_ingested_date", return_value=tomorrow_indexed)
    download_mock = mocker.patch("pipeline.sync.download_range")
    run_ingest_mock = mocker.patch("pipeline.sync.run_ingest")

    result = sync_hansards()

    download_mock.assert_not_called()
    run_ingest_mock.assert_not_called()
    assert result["up_to_date"] is True
