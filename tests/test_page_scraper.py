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


def test_filter_by_date_range_keeps_only_dates_inside_the_window():
    from scraper.page_scraper import DiscoveredDocument

    docs = [
        DiscoveredDocument(url="https://x/a.pdf", display_name="24th July, 2026"),
        DiscoveredDocument(url="https://x/b.pdf", display_name="29th May, 2026"),
    ]

    filtered = filter_by_date_range(docs, datetime(2026, 7, 1), datetime(2026, 7, 31))

    assert [doc.display_name for doc in filtered] == ["24th July, 2026"]
