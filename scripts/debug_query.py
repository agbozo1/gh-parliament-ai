"""Ad-hoc retrieval debugging: run a query against the vector store and
show each candidate chunk's rerank score and a content preview, instead of
just the answer_generator's all-or-nothing refusal.

Usage: python -m scripts.debug_query "What did parliament discuss about the 2025 budget?"
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that read env vars

from agent.graph import RELEVANCE_THRESHOLD, _text_of  # noqa: E402
from agent.llm import choose_provider, get_llm  # noqa: E402
from pipeline.retriever import DEFAULT_TOP_K, search  # noqa: E402


def main(query: str) -> None:
    print(f"LLM provider for this run: {choose_provider()}\n")

    docs = search(query, top_k=DEFAULT_TOP_K)
    print(f"Retrieved {len(docs)} chunk(s) for: {query!r}\n")
    if not docs:
        print(
            "Nothing came back from similarity_search at all -- check "
            "documents_indexed on /health and whether the collection "
            "actually has content for this topic."
        )
        return

    model = get_llm()
    for doc in docs:
        prompt = (
            "On a scale of 0 to 10, how relevant is the following passage to the "
            f"question below? Respond with only a number.\n\nQuestion: {query}"
            f"\n\nPassage: {doc.page_content[:1000]}"
        )
        try:
            raw = _text_of(model.invoke(prompt))
            digits = "".join(ch for ch in raw if ch.isdigit())
            score = int(digits) if digits else 0
            error = None
        except Exception as exc:  # noqa: BLE001
            score = 0
            error = repr(exc)

        flag = "KEEP" if score >= RELEVANCE_THRESHOLD else "drop"
        date = doc.metadata.get("date")
        filename = doc.metadata.get("pdf_filename")
        print(f"[{flag} score={score}] {date} {filename}")
        if error:
            print(f"  ERROR: {error}")
        print(f"  {doc.page_content[:200]!r}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.debug_query "your question here"')
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
