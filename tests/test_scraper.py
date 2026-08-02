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


def test_parse_display_name_handles_older_unpunctuated_lowercase_form():
    # The site's PDF naming convention used to be lowercase and fully
    # unpunctuated (no ordinal suffix, no comma, no spaces) before it
    # switched to the current "24th July, 2026" style.
    assert parse_display_name_to_date("24may2005") == datetime(2005, 5, 24)
    assert parse_display_name_to_date("2june2005") == datetime(2005, 6, 2)


def test_parse_display_name_handles_real_inconsistent_listing_text():
    # Actual entries pulled from the live listing: abbreviated months (with
    # or without a trailing period), periods/double-spaces as separators,
    # a leading day-of-week, and a non-standard abbreviation ("Dece").
    cases = {
        "17th October. 2024": datetime(2024, 10, 17),
        "15  Dece 2023": datetime(2023, 12, 15),
        "7 Dece 2023": datetime(2023, 12, 7),
        "28th  Nov. 2023": datetime(2023, 11, 28),
        "1 Aug, 2023": datetime(2023, 8, 1),
        "17th Feb. 2023": datetime(2023, 2, 17),
        "16th Feb 2023": datetime(2023, 2, 16),
        "9th Dec. 2022 (2)": datetime(2022, 12, 9),
        "Fri 18 Nov 2022": datetime(2022, 11, 18),
        "Tues 8 Nov 2022": datetime(2022, 11, 8),
    }
    for display_name, expected in cases.items():
        assert parse_display_name_to_date(display_name) == expected, display_name


def test_parse_display_name_returns_none_for_non_date_listing_entries():
    assert parse_display_name_to_date("OPAPlan") is None
    assert parse_display_name_to_date("OPEN_GOVERNMENT_PARTNERSHIP_SPAIN") is None


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
