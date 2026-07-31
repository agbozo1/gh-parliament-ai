import hashlib

from langchain_core.embeddings import Embeddings

from pipeline.ingest import (
    build_metadata,
    clean_text,
    get_vector_store,
    load_and_chunk_pdfs,
    run_ingest,
)


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
