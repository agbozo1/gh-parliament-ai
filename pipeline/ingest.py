"""Ingestion pipeline: PDF -> text extraction -> cleaning -> chunking ->
embedding -> ChromaDB.

Run as a script (``python -m pipeline.ingest``) or call ``run_ingest()``
directly, e.g. from the FastAPI ``/ingest`` endpoint.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from pipeline.embeddings import get_embeddings
from scraper.pdf_downloader import build_pdf_url, parse_display_name_to_date

load_dotenv()  # must run before the os.environ.get() calls below

logger = logging.getLogger(__name__)

PDF_INPUT_DIR = os.environ.get("PDF_OUTPUT_DIR", "data/raw")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "data/chroma")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "ghana_parliament_hansard")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Chroma enforces a max batch size on add_documents() well under the
# 200k+ chunks a full historical backfill can produce -- writing in
# smaller batches avoids the whole ingest run failing outright (and
# committing nothing at all) on a large corpus.
ADD_BATCH_SIZE = 500

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


# Number of x-position buckets used to look for a column gutter -- fine
# enough to find a narrow gap, coarse enough that a handful of stray
# characters (e.g. text rendered slightly outside its column) can't hide it.
_GUTTER_BUCKETS = 50
_MIN_WORDS_TO_JUDGE_LAYOUT = 20
_MIN_GUTTER_BUCKETS = 2


def _detect_column_gutter(page) -> float | None:
    """Return the x-coordinate of a two-column gutter, or None if the page
    looks single-column.

    Hansard pages are typically two-column; pdfplumber's default
    extract_text() orders words primarily by vertical position across the
    *whole* page width, which interleaves left- and right-column lines that
    sit at similar heights into garbled text. This looks for a contiguous,
    mostly-empty vertical band roughly in the middle of the page -- the gap
    between two columns -- rather than assuming every page is split evenly
    in half, since some pages (e.g. a title page, or a full-width table)
    genuinely are single-column and would be corrupted by a blind split.
    """
    words = page.extract_words()
    if len(words) < _MIN_WORDS_TO_JUDGE_LAYOUT:
        return None

    bucket_width = page.width / _GUTTER_BUCKETS
    covered = [False] * _GUTTER_BUCKETS
    for word in words:
        start = max(0, int(word["x0"] / bucket_width))
        end = min(_GUTTER_BUCKETS - 1, int(word["x1"] / bucket_width))
        for bucket in range(start, end + 1):
            covered[bucket] = True

    # Only look for a gutter within the middle half of the page -- a gap
    # near either edge is just a margin, not a column boundary. Find the
    # longest run of empty buckets in that range; a two-column page will
    # have one, a single-column page won't.
    middle_lo, middle_hi = _GUTTER_BUCKETS // 4, _GUTTER_BUCKETS * 3 // 4
    best_run: tuple[int, int] | None = None
    run_start = None
    for bucket in range(middle_lo, middle_hi + 1):
        if not covered[bucket]:
            if run_start is None:
                run_start = bucket
            continue
        if run_start is not None:
            if best_run is None or (bucket - run_start) > (best_run[1] - best_run[0]):
                best_run = (run_start, bucket - 1)
            run_start = None
    if run_start is not None:
        run_end = middle_hi
        if best_run is None or (run_end - run_start) > (best_run[1] - best_run[0]):
            best_run = (run_start, run_end)

    if best_run is None or (best_run[1] - best_run[0] + 1) < _MIN_GUTTER_BUCKETS:
        return None
    return (best_run[0] + best_run[1] + 1) / 2 * bucket_width


def _extract_page_text(page) -> str:
    """Extract one page's text, splitting into columns if a gutter is found."""
    gutter_x = _detect_column_gutter(page)
    if gutter_x is None:
        return page.extract_text() or ""

    left = page.within_bbox((0, 0, gutter_x, page.height)).extract_text() or ""
    right = page.within_bbox((gutter_x, 0, page.width, page.height)).extract_text() or ""
    return "\n\n".join(part for part in (left, right) if part)


def extract_text(pdf_path: Path) -> str:
    """Extract raw text from every page of a PDF, splitting two-column
    pages so left/right column lines at similar heights aren't interleaved.
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages = [_extract_page_text(page) for page in pdf.pages]
    return "\n".join(page for page in pages if page)


def clean_text(text: str) -> str:
    """Strip repeated whitespace and blank-line runs left by PDF extraction."""
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def build_metadata(pdf_path: Path, base_url: str | None = None) -> dict:
    """Derive source_url, date, and session_id metadata from a filename."""
    display_name = pdf_path.stem
    base_url = base_url or os.environ.get(
        "PARLIAMENT_BASE_URL", "https://www.parliament.gh"
    )
    parsed_date = parse_display_name_to_date(display_name)

    metadata: dict = {
        "pdf_filename": pdf_path.name,
        "source_url": None,
        "date": None,
        "session_id": None,
    }
    if parsed_date is not None:
        source_url, _ = build_pdf_url(parsed_date, base_url=base_url)
        metadata["source_url"] = source_url
        metadata["date"] = parsed_date.date().isoformat()
        # Ghana's parliamentary sessions run roughly on the calendar year;
        # this is a coarse grouping, not an official session identifier.
        metadata["session_id"] = str(parsed_date.year)
    return metadata


def load_and_chunk_pdfs(
    input_dir: str = PDF_INPUT_DIR, skip_filenames: set[str] | None = None
) -> list[Document]:
    """Extract, clean, and chunk every PDF in a directory into Documents.

    skip_filenames (matched against the PDF's basename) lets callers avoid
    re-chunking files already indexed — see run_ingest, which uses this to
    stay idempotent across repeated (e.g. daily) runs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    pdf_dir = Path(input_dir)
    if not pdf_dir.exists():
        logger.warning("PDF input directory %s does not exist", pdf_dir)
        return []

    skip_filenames = skip_filenames or set()
    documents: list[Document] = []
    pdf_paths = [p for p in sorted(pdf_dir.glob("*.pdf")) if p.name not in skip_filenames]
    skipped = len(list(pdf_dir.glob("*.pdf"))) - len(pdf_paths)
    if skipped:
        logger.info("Skipping %d already-indexed PDF(s)", skipped)

    for pdf_path in pdf_paths:
        raw_text = extract_text(pdf_path)
        if not raw_text.strip():
            logger.warning("No extractable text in %s, skipping", pdf_path)
            continue
        cleaned = clean_text(raw_text)
        metadata = build_metadata(pdf_path)

        for chunk in splitter.split_text(cleaned):
            documents.append(Document(page_content=chunk, metadata=dict(metadata)))

    logger.info("Chunked %d document(s) into %d chunks", len(pdf_paths), len(documents))
    return documents


def get_vector_store(
    persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION
) -> Chroma:
    """Return a Chroma vector store.

    If CHROMA_HOST is set (e.g. running against the `chromadb` service in
    docker-compose), connect to it over HTTP. Otherwise fall back to an
    embedded, on-disk persistent store under persist_dir — the simplest
    setup for local development.
    """
    chroma_host = os.environ.get("CHROMA_HOST")
    if chroma_host:
        import chromadb

        client = chromadb.HttpClient(
            host=chroma_host,
            port=int(os.environ.get("CHROMA_PORT", "8001")),
        )
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=get_embeddings(),
        )

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )


def _existing_metadatas(vector_store: Chroma) -> list[dict]:
    return [m for m in vector_store.get(include=["metadatas"]).get("metadatas", []) if m]


def get_ingested_filenames(
    persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION
) -> set[str]:
    """PDF filenames already represented by at least one chunk in the store."""
    vector_store = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    return {m["pdf_filename"] for m in _existing_metadatas(vector_store) if m.get("pdf_filename")}


def get_latest_ingested_date(
    persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION
) -> str | None:
    """Most recent sitting date (ISO string) already indexed, or None if the store is empty."""
    vector_store = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    dates = [m["date"] for m in _existing_metadatas(vector_store) if m.get("date")]
    return max(dates) if dates else None


def run_ingest(
    input_dir: str = PDF_INPUT_DIR,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> int:
    """Chunk every not-yet-indexed PDF under input_dir and write it to ChromaDB.

    Idempotent: PDFs already represented in the collection (by filename) are
    skipped, so calling this repeatedly (e.g. from a daily sync job) only
    ever adds genuinely new documents instead of duplicating chunks.

    Returns the number of chunks written.
    """
    vector_store = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    already_ingested = {
        m["pdf_filename"] for m in _existing_metadatas(vector_store) if m.get("pdf_filename")
    }

    documents = load_and_chunk_pdfs(input_dir, skip_filenames=already_ingested)
    if not documents:
        logger.info("No new documents to ingest")
        return 0

    for start in range(0, len(documents), ADD_BATCH_SIZE):
        batch = documents[start : start + ADD_BATCH_SIZE]
        vector_store.add_documents(batch)
        logger.info(
            "Ingested chunks %d-%d of %d into collection '%s'",
            start + 1,
            start + len(batch),
            len(documents),
            collection_name,
        )
    return len(documents)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingest()
