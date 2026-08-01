from datetime import datetime

from scraper.page_scraper import discover_hansard_documents, filter_by_date_range

# Trimmed fixture based on real markup from https://www.parliament.gh/docs?type=HS —
# each row's date is only reachable via the showPDF(...) click handler, not a href.
SAMPLE_LISTING_HTML = """
<table class="table table-hover table-stripped doc-table">
  <tbody>
    <tr role="button" onclick="showPDF('pb/24th July, 2026.pdf','Hansard 24th July, 2026');">
      <td>Friday, 24th July, 2026</td><td>Hansard 24th July, 2026</td>
    </tr>
    <tr role="button" onclick="showPDF('pb/29th May 2026.pdf','Hansard 29th May, 2026');">
      <td>Friday, 29th May, 2026</td><td>Hansard 29th May, 2026</td>
    </tr>
    <tr role="button" onclick="showPDF('pb/16th July, 2026 (1).pdf','Hansard 16th July, 2026');">
      <td>Thursday, 16th July, 2026</td><td>Hansard 16th July, 2026</td>
    </tr>
    <tr role="button" onclick="showPDF('pb/10th June,2026.pdf','Hansard 10th June,2026');">
      <td>Wednesday, 10th June, 2026</td><td>Hansard 10th June,2026</td>
    </tr>
    <tr role="button" onclick="showPDF('pb/10th June, 2026.pdf','Hansard 10th June, 2026');">
      <td>Wednesday, 10th June, 2026</td><td>Hansard 10th June, 2026</td>
    </tr>
    <tr role="button" onclick="somethingElse('not-a-pdf-row');">
      <td>irrelevant row</td>
    </tr>
    <tr role="button" onclick="showPDF('pb/nonsense-name.pdf','Unparseable');">
      <td>unparseable date text</td>
    </tr>
  </tbody>
</table>
"""


def _mock_driver(mocker, html: str):
    mocker.patch("scraper.page_scraper._build_driver", return_value=mocker.Mock())
    mocker.patch("scraper.page_scraper._load_listing_page", return_value=html)


def test_build_driver_uses_docker_provided_firefox_and_driver_paths(mocker):
    from scraper.page_scraper import _build_driver

    mocker.patch("scraper.page_scraper.FIREFOX_BIN", "/usr/bin/firefox-esr")
    mocker.patch("scraper.page_scraper.GECKODRIVER_PATH", "/usr/local/bin/geckodriver")
    firefox_mock = mocker.patch(
        "scraper.page_scraper.webdriver.Firefox", return_value=mocker.Mock()
    )
    service_mock = mocker.patch("scraper.page_scraper.FirefoxService")

    _build_driver()

    service_mock.assert_called_once_with(executable_path="/usr/local/bin/geckodriver")
    _, kwargs = firefox_mock.call_args
    assert kwargs["options"].binary_location == "/usr/bin/firefox-esr"
    assert kwargs["service"] == service_mock.return_value


def test_build_driver_falls_back_to_selenium_manager_when_unset(mocker):
    from scraper.page_scraper import _build_driver

    mocker.patch("scraper.page_scraper.FIREFOX_BIN", None)
    mocker.patch("scraper.page_scraper.GECKODRIVER_PATH", None)
    firefox_mock = mocker.patch(
        "scraper.page_scraper.webdriver.Firefox", return_value=mocker.Mock()
    )

    _build_driver()

    _, kwargs = firefox_mock.call_args
    assert "service" not in kwargs
    assert kwargs["options"].binary_location == ""


def test_discover_parses_showpdf_click_handlers_not_href_links(mocker):
    _mock_driver(mocker, SAMPLE_LISTING_HTML)

    documents = discover_hansard_documents(base_url="https://www.parliament.gh")

    names = {doc.display_name for doc in documents}
    assert "24th July, 2026" in names
    assert "29th May, 2026" in names  # comma-less source text still parses


def test_discover_canonicalizes_urls_via_build_pdf_url(mocker):
    _mock_driver(mocker, SAMPLE_LISTING_HTML)

    documents = discover_hansard_documents(base_url="https://www.parliament.gh")

    by_name = {doc.display_name: doc for doc in documents}
    assert (
        by_name["24th July, 2026"].url
        == "https://www.parliament.gh/epanel/docs/pb/24th%20July%2C%202026.pdf"
    )


def test_discover_dedupes_inconsistent_raw_formatting_for_the_same_date(mocker):
    _mock_driver(mocker, SAMPLE_LISTING_HTML)

    documents = discover_hansard_documents(base_url="https://www.parliament.gh")

    # "10th June,2026" and "10th June, 2026" and the "(1)" correction suffix
    # on 16th July should each collapse to exactly one canonical entry.
    june_10th = [doc for doc in documents if doc.display_name == "10th June, 2026"]
    july_16th = [doc for doc in documents if doc.display_name == "16th July, 2026"]
    assert len(june_10th) == 1
    assert len(july_16th) == 1


def test_discover_skips_rows_it_cannot_parse_a_date_from(mocker):
    _mock_driver(mocker, SAMPLE_LISTING_HTML)

    documents = discover_hansard_documents(base_url="https://www.parliament.gh")

    assert all(doc.display_name != "nonsense-name" for doc in documents)


def test_discover_returns_empty_list_when_nothing_matches(mocker):
    _mock_driver(mocker, "<html><body>no documents here</body></html>")

    documents = discover_hansard_documents(base_url="https://www.parliament.gh")

    assert documents == []


def test_listing_page_url_builds_offset_query_param():
    from scraper.page_scraper import _listing_page_url

    assert (
        _listing_page_url("https://www.parliament.gh", "/docs?type=HS", 0)
        == "https://www.parliament.gh/docs?type=HS"
    )
    assert (
        _listing_page_url("https://www.parliament.gh", "/docs?type=HS", 50)
        == "https://www.parliament.gh/docs?type=HS&P=50"
    )
    assert (
        _listing_page_url("https://www.parliament.gh", "/docs?type=HS", 2050)
        == "https://www.parliament.gh/docs?type=HS&P=2050"
    )


def test_discover_paginates_when_since_given_and_stops_past_the_boundary(mocker):
    page1 = """
    <tr onclick="showPDF('pb/24th July, 2026.pdf','x');"></tr>
    <tr onclick="showPDF('pb/2nd July 2026.pdf','x');"></tr>
    """
    page2 = """
    <tr onclick="showPDF('pb/10th June, 2026.pdf','x');"></tr>
    <tr onclick="showPDF('pb/1st January, 2026.pdf','x');"></tr>
    """
    mocker.patch("scraper.page_scraper._build_driver", return_value=mocker.Mock())
    urls_seen = []

    def fake_load(driver, url):
        urls_seen.append(url)
        return page1 if len(urls_seen) == 1 else page2

    mocker.patch("scraper.page_scraper._load_listing_page", side_effect=fake_load)
    mocker.patch("scraper.page_scraper.time.sleep")

    documents = discover_hansard_documents(
        base_url="https://www.parliament.gh", since=datetime(2026, 6, 1)
    )

    # Page 1's URL has no P= param; page 2 is offset 50. Stops there because
    # page 2's oldest date (1st January, 2026) is already before `since`.
    assert urls_seen == [
        "https://www.parliament.gh/docs?type=HS",
        "https://www.parliament.gh/docs?type=HS&P=50",
    ]
    names = {doc.display_name for doc in documents}
    assert names == {
        "24th July, 2026",
        "2nd July, 2026",
        "10th June, 2026",
        "1st January, 2026",
    }


def test_discover_stops_when_a_page_returns_no_documents(mocker):
    page1 = "<tr onclick=\"showPDF('pb/24th July, 2026.pdf','x');\"></tr>"
    mocker.patch("scraper.page_scraper._build_driver", return_value=mocker.Mock())
    urls_seen = []

    def fake_load(driver, url):
        urls_seen.append(url)
        return page1 if len(urls_seen) == 1 else "<html>no documents</html>"

    mocker.patch("scraper.page_scraper._load_listing_page", side_effect=fake_load)
    mocker.patch("scraper.page_scraper.time.sleep")

    documents = discover_hansard_documents(
        base_url="https://www.parliament.gh", since=datetime(2017, 1, 1)
    )

    assert len(urls_seen) == 2
    assert {doc.display_name for doc in documents} == {"24th July, 2026"}


def test_discover_respects_max_pages_cap(mocker):
    mocker.patch("scraper.page_scraper.MAX_PAGES", 3)
    mocker.patch("scraper.page_scraper._build_driver", return_value=mocker.Mock())
    page = "<tr onclick=\"showPDF('pb/24th July, 2026.pdf','x');\"></tr>"
    load_mock = mocker.patch("scraper.page_scraper._load_listing_page", return_value=page)
    mocker.patch("scraper.page_scraper.time.sleep")

    discover_hansard_documents(base_url="https://www.parliament.gh", since=datetime(2017, 1, 1))

    assert load_mock.call_count == 3


def test_discover_without_since_reads_only_the_first_page(mocker):
    page = "<tr onclick=\"showPDF('pb/24th July, 2026.pdf','x');\"></tr>"
    mocker.patch("scraper.page_scraper._build_driver", return_value=mocker.Mock())
    load_mock = mocker.patch("scraper.page_scraper._load_listing_page", return_value=page)

    discover_hansard_documents(base_url="https://www.parliament.gh")

    assert load_mock.call_count == 1


def test_filter_by_date_range_keeps_only_dates_inside_the_window():
    from scraper.page_scraper import DiscoveredDocument

    docs = [
        DiscoveredDocument(url="https://x/a.pdf", display_name="24th July, 2026"),
        DiscoveredDocument(url="https://x/b.pdf", display_name="29th May, 2026"),
    ]

    filtered = filter_by_date_range(docs, datetime(2026, 7, 1), datetime(2026, 7, 31))

    assert [doc.display_name for doc in filtered] == ["24th July, 2026"]
