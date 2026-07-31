"""Tools available to the Hansard research agent."""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import tool

from pipeline import retriever


@tool
def search_hansard(query: str, date_filter: Optional[str] = None) -> list[dict]:
    """Search the Ghana Parliament Hansard vector store for chunks relevant
    to a query, optionally restricted to a specific sitting date
    (ISO format, e.g. "2025-02-11").

    Returns a list of {"content": str, "metadata": dict} dicts.
    """
    docs: list[Document] = retriever.search(query, date_filter=date_filter)
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]


@tool
def list_sessions(year: Optional[int] = None) -> list[str]:
    """List parliamentary session identifiers available in the vector
    store, optionally filtered to a given year. Useful for checking what
    data is actually indexed before attempting to answer a question.
    """
    return retriever.list_sessions(year=year)


AGENT_TOOLS = [search_hansard, list_sessions]
