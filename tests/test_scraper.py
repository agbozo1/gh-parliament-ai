from datetime import datetime

import pytest
import requests

from scraper.pdf_downloader import (
    ScrapeResult,
    build_pdf_url,
    download_pdf,
    download_range,
    parse_display_name_to_date,
)


def test_build_pdf_url_formats_date_with_ordinal_suffix():
    url, display_name = build_pdf_url(datetime(2025, 2, 11), base_url="https://example.com")
    assert display_name == "11th February, 2025"
    assert url == "https://example.com/epanel/docs/pb/11th%20February%2C%202025.pdf"


def test_build_pdf_url_suffixes():
    assert build_pdf_url(datetime(2025, 1, 1))[1].startswith("1st")
    assert build_pdf_url(datetime(2025, 1, 2))[1].startswith("2nd")
    assert build_pdf_url(datetime(2025, 1, 3))[1].startswith("3rd")
    assert build_pdf_url(datetime(2025, 1, 11))[1].startswith("11th")
    assert build_pdf_url(datetime(2025, 1, 21))[1].startswith("21st")


def test_parse_display_name_round_trips_with_build_pdf_url():
    _, display_name = build_pdf_url(datetime(2025, 3, 7))
    parsed = parse_display_name_to_date(display_name)
    assert parsed == datetime(2025, 3, 7)


def test_parse_display_name_returns_none_for_garbage():
    assert parse_display_name_to_date("not-a-date.pdf") is None


class _FakeResponse:
    def __init__(self, content=b"%PDF-1.4 fake", status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(response=self)
            raise error


def test_download_pdf_saves_file(tmp_path, mocker):
    mocker.patch("scraper.pdf_downloader.requests.get", return_value=_FakeResponse())

    saved_path = download_pdf(
        "https://example.com/doc.pdf", "11th February, 2025", output_dir=str(tmp_path)
    )

    assert saved_path is not None
    assert saved_path.read_bytes() == b"%PDF-1.4 fake"
    assert saved_path.name == "11th February, 2025.pdf"


def test_download_pdf_returns_none_on_404_without_retrying(tmp_path, mocker):
    mock_get = mocker.patch(
        "scraper.pdf_downloader.requests.get", return_value=_FakeResponse(status_code=404)
    )

    result = download_pdf(
        "https://example.com/missing.pdf", "1st January, 2099", output_dir=str(tmp_path)
    )

    assert result is None
    assert mock_get.call_count == 1


def test_download_pdf_retries_on_transient_failure_then_succeeds(tmp_path, mocker):
    mocker.patch("time.sleep", return_value=None)

    mock_get = mocker.patch(
        "scraper.pdf_downloader.requests.get",
        side_effect=[
            requests.exceptions.ConnectionError("boom"),
            requests.exceptions.ConnectionError("boom again"),
            _FakeResponse(),
        ],
    )

    saved_path = download_pdf(
        "https://example.com/doc.pdf", "2nd January, 2025", output_dir=str(tmp_path)
    )

    assert saved_path is not None
    assert mock_get.call_count == 3


def test_download_pdf_raises_after_exhausting_retries(tmp_path, mocker):
    mocker.patch("time.sleep", return_value=None)
    mocker.patch(
        "scraper.pdf_downloader.requests.get",
        side_effect=requests.exceptions.ConnectionError("always fails"),
    )

    with pytest.raises(requests.exceptions.RequestException):
        download_pdf("https://example.com/doc.pdf", "3rd January, 2025", output_dir=str(tmp_path))


def test_download_range_collects_results_and_logs_run(tmp_path, mocker):
    mocker.patch(
        "scraper.pdf_downloader.download_pdf",
        side_effect=[tmp_path / "a.pdf", None, tmp_path / "c.pdf"],
    )

    result = download_range(
        datetime(2025, 1, 1),
        datetime(2025, 1, 3),
        output_dir=str(tmp_path),
        delay_seconds=0,
    )

    assert isinstance(result, ScrapeResult)
    assert len(result.urls_attempted) == 3
    assert len(result.documents_downloaded) == 2
    assert result.finished_at is not None
    assert result.as_dict()["documents_downloaded"] == result.documents_downloaded
