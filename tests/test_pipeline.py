import hashlib

from langchain_core.embeddings import Embeddings

from pipeline.ingest import (
    _detect_column_gutter,
    _extract_page_text,
    build_metadata,
    clean_text,
    get_ingested_filenames,
    get_latest_ingested_date,
    get_vector_store,
    load_and_chunk_pdfs,
    run_ingest,
)


class FakePage:
    """Minimal stand-in for a pdfplumber Page, for testing column detection
    without needing a real PDF file."""

    def __init__(self, width, height, words, full_text=""):
        self.width = width
        self.height = height
        self._words = words
        self._full_text = full_text

    def extract_words(self):
        return self._words

    def extract_text(self):
        return self._full_text

    def within_bbox(self, bbox):
        x0, _top, x1, _bottom = bbox
        words_in_bbox = [w for w in self._words if w["x0"] >= x0 - 1e-6 and w["x1"] <= x1 + 1e-6]
        text = " ".join(w["text"] for w in words_in_bbox)
        return FakePage(x1 - x0, self.height, words_in_bbox, full_text=text)


def _make_words(prefix, x_start, x_end, count, top=100.0):
    step = (x_end - x_start) / count
    words = []
    for i in range(count):
        x0 = x_start + i * step
        x1 = x0 + step * 0.8
        words.append({"text": f"{prefix}{i}", "x0": x0, "x1": x1, "top": top, "bottom": top + 10})
    return words


def _make_two_column_lines(
    num_lines,
    left_range=(50, 250),
    right_range=(350, 550),
    top_start=100.0,
    line_height=15.0,
    words_per_column=20,
):
    """Words for a synthetic multi-line two-column page, dense enough per
    column that there's no bucket-sized gap within a column itself -- only
    the real gutter between columns should register as low-density."""
    words = []
    for i in range(num_lines):
        top = top_start + i * line_height
        for col, (x_start, x_end) in enumerate((left_range, right_range)):
            words += _make_words(f"L{i}C{col}W", x_start, x_end, words_per_column, top=top)
    return words


def _make_single_column_lines(
    num_lines, x_range=(50, 550), top_start=100.0, line_height=15.0, words_per_line=50
):
    """Words for a synthetic multi-line single-column page spanning the
    full width, dense enough that no bucket-sized gap appears anywhere."""
    words = []
    for i in range(num_lines):
        top = top_start + i * line_height
        words += _make_words(f"L{i}W", x_range[0], x_range[1], words_per_line, top=top)
    return words


class FakeEmbeddings(Embeddings):
    """Deterministic, dependency-free stand-in for a real embedding model."""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def test_clean_text_collapses_whitespace_and_blank_lines():
    dirty = "Hansard   report\t\ttext\n\n\n\nMore   text"
    cleaned = clean_text(dirty)
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == "Hansard report text\n\nMore text"


def test_detect_column_gutter_finds_the_gap_in_a_two_column_page():
    words = _make_two_column_lines(num_lines=12)
    page = FakePage(width=600, height=800, words=words)

    gutter_x = _detect_column_gutter(page)

    assert gutter_x is not None
    assert 250 < gutter_x < 350


def test_detect_column_gutter_returns_none_for_a_single_column_page():
    # Words spread continuously across the full width on every line -- no
    # gap in the middle.
    words = _make_single_column_lines(num_lines=12)
    page = FakePage(width=600, height=800, words=words)

    assert _detect_column_gutter(page) is None


def test_detect_column_gutter_returns_none_when_too_few_words():
    words = _make_words("word", 50, 550, 5)
    page = FakePage(width=600, height=800, words=words)

    assert _detect_column_gutter(page) is None


def test_detect_column_gutter_tolerates_a_stray_word_centered_in_the_gutter():
    # Regression test, based on a real Hansard page: a running header's
    # date ("20th November, 2025") had a bounding box that bridged the
    # gutter, landing its center almost exactly in the middle of an
    # otherwise-empty band. One such stray word must not prevent detecting
    # the real gutter from the two-column body beneath it.
    body = _make_two_column_lines(num_lines=12)
    stray = [{"text": "stray", "x0": 280.0, "x1": 320.0, "top": 50.0, "bottom": 60.0}]
    page = FakePage(width=600, height=800, words=body + stray)

    gutter_x = _detect_column_gutter(page)

    assert gutter_x is not None
    assert 250 < gutter_x < 350


def test_detect_column_gutter_requires_real_density_on_both_sides():
    # All content clustered near the left edge, with nothing on the right
    # half of the page at all -- there's a "low-density" band, but it's
    # not a real column gutter since there's no content flanking it on
    # the right. Must not be misread as two-column.
    words = _make_words("word", 50, 140, 25, top=100.0)
    page = FakePage(width=600, height=800, words=words)

    assert _detect_column_gutter(page) is None


def test_extract_page_text_splits_two_column_pages_left_then_right():
    words = _make_two_column_lines(num_lines=12)
    page = FakePage(width=600, height=800, words=words)

    text = _extract_page_text(page)

    assert text.index("L0C0W0") < text.index("L0C1W0")
    assert "L0C1W0" in text and "L11C0W19" in text


def test_extract_page_text_falls_back_to_plain_extraction_for_single_column():
    words = _make_single_column_lines(num_lines=12)
    page = FakePage(width=600, height=800, words=words, full_text="the full single-column text")

    assert _extract_page_text(page) == "the full single-column text"


def test_build_metadata_parses_date_and_session_from_filename(tmp_path):
    pdf_path = tmp_path / "11th February, 2025.pdf"
    pdf_path.touch()

    metadata = build_metadata(pdf_path, base_url="https://www.parliament.gh")

    assert metadata["pdf_filename"] == "11th February, 2025.pdf"
    assert metadata["date"] == "2025-02-11"
    assert metadata["session_id"] == "2025"
    assert metadata["source_url"].endswith("11th%20February%2C%202025.pdf")


def test_build_metadata_handles_unparseable_filename(tmp_path):
    pdf_path = tmp_path / "not-a-date.pdf"
    pdf_path.touch()

    metadata = build_metadata(pdf_path)

    assert metadata["pdf_filename"] == "not-a-date.pdf"
    assert metadata["date"] is None
    assert metadata["source_url"] is None


def test_load_and_chunk_pdfs_respects_chunk_size(tmp_path, mocker):
    pdf_path = tmp_path / "11th February, 2025.pdf"
    pdf_path.touch()

    long_text = "Ghana Parliament proceedings. " * 200  # well over 800 chars
    mocker.patch("pipeline.ingest.extract_text", return_value=long_text)

    documents = load_and_chunk_pdfs(str(tmp_path))

    assert len(documents) > 1
    for doc in documents:
        assert len(doc.page_content) <= 800
        assert doc.metadata["pdf_filename"] == "11th February, 2025.pdf"
        assert doc.metadata["date"] == "2025-02-11"


def test_load_and_chunk_pdfs_skips_empty_extraction(tmp_path, mocker):
    pdf_path = tmp_path / "8th January, 2025.pdf"
    pdf_path.touch()
    mocker.patch("pipeline.ingest.extract_text", return_value="   ")

    documents = load_and_chunk_pdfs(str(tmp_path))

    assert documents == []


def test_load_and_chunk_pdfs_missing_directory_returns_empty_list(tmp_path):
    missing_dir = tmp_path / "does-not-exist"
    assert load_and_chunk_pdfs(str(missing_dir)) == []


def test_load_and_chunk_pdfs_skips_already_ingested_filenames(tmp_path, mocker):
    (tmp_path / "11th February, 2025.pdf").touch()
    (tmp_path / "12th February, 2025.pdf").touch()
    mocker.patch("pipeline.ingest.extract_text", return_value="Some short text.")

    documents = load_and_chunk_pdfs(
        str(tmp_path), skip_filenames={"11th February, 2025.pdf"}
    )

    filenames = {doc.metadata["pdf_filename"] for doc in documents}
    assert filenames == {"12th February, 2025.pdf"}


def test_run_ingest_writes_and_reads_back_from_in_memory_chroma(tmp_path, mocker):
    pdf_path = tmp_path / "raw" / "11th February, 2025.pdf"
    pdf_path.parent.mkdir()
    pdf_path.touch()

    mocker.patch(
        "pipeline.ingest.extract_text",
        return_value="Members debated the budget statement today.",
    )
    mocker.patch("pipeline.ingest.get_embeddings", return_value=FakeEmbeddings())

    persist_dir = str(tmp_path / "chroma")
    chunks_written = run_ingest(
        input_dir=str(pdf_path.parent),
        persist_dir=persist_dir,
        collection_name="test_collection",
    )

    assert chunks_written == 1

    vector_store = get_vector_store(persist_dir=persist_dir, collection_name="test_collection")
    results = vector_store.similarity_search("budget statement", k=1)

    assert len(results) == 1
    assert "budget statement" in results[0].page_content
    assert results[0].metadata["date"] == "2025-02-11"


def test_run_ingest_is_idempotent_across_repeated_calls(tmp_path, mocker):
    """A second run over the same PDFs should add nothing new — this is
    what keeps a daily sync job from duplicating already-indexed chunks."""
    pdf_path = tmp_path / "raw" / "11th February, 2025.pdf"
    pdf_path.parent.mkdir()
    pdf_path.touch()

    mocker.patch(
        "pipeline.ingest.extract_text",
        return_value="Members debated the budget statement today.",
    )
    mocker.patch("pipeline.ingest.get_embeddings", return_value=FakeEmbeddings())

    persist_dir = str(tmp_path / "chroma")
    kwargs = dict(
        input_dir=str(pdf_path.parent), persist_dir=persist_dir, collection_name="idempotent"
    )

    first_run = run_ingest(**kwargs)
    second_run = run_ingest(**kwargs)

    assert first_run == 1
    assert second_run == 0

    vector_store = get_vector_store(persist_dir=persist_dir, collection_name="idempotent")
    assert vector_store._collection.count() == 1


def test_run_ingest_writes_in_batches_to_avoid_exceeding_chromas_max_batch_size(
    tmp_path, mocker
):
    pdf_path = tmp_path / "raw" / "11th February, 2025.pdf"
    pdf_path.parent.mkdir()
    pdf_path.touch()

    long_text = "Ghana Parliament proceedings. " * 200  # multiple 800-char chunks
    mocker.patch("pipeline.ingest.extract_text", return_value=long_text)
    mocker.patch("pipeline.ingest.ADD_BATCH_SIZE", 2)

    fake_store = mocker.Mock()
    fake_store.get.return_value = {"metadatas": []}
    mocker.patch("pipeline.ingest.get_vector_store", return_value=fake_store)

    chunks_written = run_ingest(input_dir=str(pdf_path.parent))

    all_batches = [call.args[0] for call in fake_store.add_documents.call_args_list]
    assert len(all_batches) > 1  # a single call would have exceeded the batch size
    assert all(len(batch) <= 2 for batch in all_batches)
    assert sum(len(batch) for batch in all_batches) == chunks_written


def test_get_latest_ingested_date_and_filenames(tmp_path, mocker):
    pdf_path = tmp_path / "raw" / "11th February, 2025.pdf"
    pdf_path.parent.mkdir()
    pdf_path.touch()

    mocker.patch("pipeline.ingest.extract_text", return_value="Budget debate text.")
    mocker.patch("pipeline.ingest.get_embeddings", return_value=FakeEmbeddings())

    persist_dir = str(tmp_path / "chroma")
    collection_name = "freshness"

    assert get_latest_ingested_date(persist_dir, collection_name) is None
    assert get_ingested_filenames(persist_dir, collection_name) == set()

    run_ingest(
        input_dir=str(pdf_path.parent), persist_dir=persist_dir, collection_name=collection_name
    )

    assert get_latest_ingested_date(persist_dir, collection_name) == "2025-02-11"
    assert get_ingested_filenames(persist_dir, collection_name) == {"11th February, 2025.pdf"}
