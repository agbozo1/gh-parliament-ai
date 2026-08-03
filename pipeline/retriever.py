"""ChromaDB retrieval logic, with optional metadata filtering."""

from __future__ import annotations

from langchain_core.documents import Document

from pipeline.ingest import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, get_vector_store

# With a large corpus (hundreds of thousands of chunks spanning decades),
# a small top_k risks losing the actually-relevant chunk among other years'
# similarly-worded debates before the reranker ever sees it.
DEFAULT_TOP_K = 15


def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    date_filter: str | None = None,
    session_id: str | None = None,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> list[Document]:
    """Semantic search over ingested Hansard chunks.

    date_filter matches the chunk's ``date`` metadata field exactly
    (ISO format, e.g. "2025-02-11"). session_id matches the coarse
    ``session_id`` metadata field (see pipeline.ingest.build_metadata).
    """
    vector_store = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)

    where: dict = {}
    if date_filter and session_id:
        where = {"$and": [{"date": date_filter}, {"session_id": session_id}]}
    elif date_filter:
        where = {"date": date_filter}
    elif session_id:
        where = {"session_id": session_id}

    return vector_store.similarity_search(query, k=top_k, filter=where or None)


def list_sessions(
    year: int | None = None,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> list[str]:
    """Return the distinct session identifiers present in the vector store."""
    vector_store = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    raw = vector_store.get(include=["metadatas"])
    session_ids = {
        metadata.get("session_id")
        for metadata in raw.get("metadatas", [])
        if metadata and metadata.get("session_id")
    }
    if year is not None:
        session_ids = {s for s in session_ids if s == str(year)}
    return sorted(session_ids)


def count_documents(
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> int:
    vector_store = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    return vector_store._collection.count()
