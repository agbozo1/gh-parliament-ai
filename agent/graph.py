"""LangGraph agentic retrieval flow for the Hansard research assistant.

Nodes: query_classifier -> retriever -> reranker -> answer_generator ->
citation_formatter -> END, with query_classifier able to route straight to
answer_generator for queries that don't need document retrieval.

Hallucination prevention is the top priority: answer_generator refuses to
answer ("I could not find relevant information in the parliamentary
records.") whenever retrieval was needed but no chunk survived reranking.
"""

from __future__ import annotations

import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.llm import get_llm
from agent.tools import search_hansard

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "I could not find relevant information in the parliamentary records."
RELEVANCE_THRESHOLD = 3  # 0-10 scale; chunks scoring below this are dropped

DIRECT_PATH_PROMPT = (
    "You are the Ghana Parliament Hansard research assistant: a tool for "
    "exploring Ghana's parliamentary debates and records. Answer briefly. "
    "If asked who or what you are, describe yourself in those terms, not as "
    "a general-purpose AI assistant. If the message is a greeting or a "
    "question about you or how to use you, respond naturally. If it's "
    "actually a substantive question that would need real parliamentary "
    "record content to answer well, say you don't have enough information "
    "and ask the user to phrase it as a specific question about a debate, "
    "policy, or sitting date -- do not answer it from general knowledge.\n\n"
    "User: {query}"
)


class AgentState(TypedDict, total=False):
    query: str
    date_filter: Optional[str]
    needs_retrieval: bool
    retrieved: list[dict]
    reranked: list[dict]
    answer: str
    sources: list[dict]


def _text_of(response) -> str:
    return response.content if hasattr(response, "content") else str(response)


def query_classifier(state: AgentState, llm=None) -> AgentState:
    """Decide whether the query needs document retrieval or can be answered directly."""
    model = llm or get_llm()
    prompt = (
        "Classify the following user query as either RETRIEVAL (it asks about "
        "parliamentary debates, Hansard records, or anything requiring looking up "
        "documents) or DIRECT (a greeting, meta question about the assistant, or "
        "something answerable without documents). Respond with exactly one word: "
        f"RETRIEVAL or DIRECT.\n\nQuery: {state['query']}"
    )
    try:
        text = _text_of(model.invoke(prompt)).strip().upper()
        needs_retrieval = "DIRECT" not in text
    except Exception:
        logger.exception("query_classifier LLM call failed; defaulting to retrieval")
        needs_retrieval = True
    return {**state, "needs_retrieval": needs_retrieval}


def retrieve(state: AgentState) -> AgentState:
    """Pull top-k chunks (with metadata) from ChromaDB via the search_hansard tool."""
    docs = search_hansard.invoke({"query": state["query"], "date_filter": state.get("date_filter")})
    return {**state, "retrieved": docs}


def rerank(state: AgentState, llm=None) -> AgentState:
    """Score each retrieved chunk for relevance and drop anything below threshold."""
    docs = state.get("retrieved", [])
    if not docs:
        return {**state, "reranked": []}

    model = llm or get_llm()
    scored = []
    for doc in docs:
        prompt = (
            "On a scale of 0 to 10, how relevant is the following passage to the "
            f"question below? Respond with only a number.\n\nQuestion: {state['query']}"
            f"\n\nPassage: {doc['content'][:1000]}"
        )
        try:
            digits = "".join(ch for ch in _text_of(model.invoke(prompt)) if ch.isdigit())
            score = int(digits) if digits else 0
        except Exception:
            logger.exception("rerank LLM call failed for a chunk; scoring 0")
            score = 0
        scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    reranked = [doc for score, doc in scored if score >= RELEVANCE_THRESHOLD]
    return {**state, "reranked": reranked}


def generate_answer(state: AgentState, llm=None) -> AgentState:
    """Generate an answer grounded strictly in retrieved context, or refuse."""
    if not state.get("needs_retrieval", True):
        model = llm or get_llm()
        prompt = DIRECT_PATH_PROMPT.format(query=state["query"])
        text = _text_of(model.invoke(prompt))
        return {**state, "answer": text, "sources": []}

    if not state.get("reranked"):
        return {**state, "answer": NO_CONTEXT_MESSAGE, "sources": []}

    model = llm or get_llm()
    context = "\n\n---\n\n".join(doc["content"] for doc in state["reranked"])
    prompt = (
        "Answer the user's question using ONLY the parliamentary record excerpts "
        "below. If the excerpts do not contain enough information to answer, "
        f'respond with exactly "{NO_CONTEXT_MESSAGE}" and nothing else. Do not use '
        f"outside knowledge.\n\nExcerpts:\n{context}\n\nQuestion: {state['query']}"
    )
    text = _text_of(model.invoke(prompt))
    return {**state, "answer": text}


def format_citations(state: AgentState) -> AgentState:
    """Attach source references (document name, date, URL) to the answer."""
    if not state.get("reranked") or state.get("answer") == NO_CONTEXT_MESSAGE:
        return {**state, "sources": []}

    sources = []
    seen = set()
    for doc in state["reranked"]:
        metadata = doc.get("metadata", {})
        key = metadata.get("pdf_filename")
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "document": metadata.get("pdf_filename"),
                "date": metadata.get("date"),
                "url": metadata.get("source_url"),
            }
        )
    return {**state, "sources": sources}


def _route_after_classification(state: AgentState) -> str:
    return "retriever" if state.get("needs_retrieval", True) else "answer_generator"


def build_graph(llm=None):
    """Build and compile the LangGraph agent.

    Pass `llm` to inject a stub/mock chat model (e.g. in tests); when
    omitted, each node resolves a live model via agent.llm.get_llm().
    """
    graph = StateGraph(AgentState)

    graph.add_node("query_classifier", lambda state: query_classifier(state, llm=llm))
    graph.add_node("retriever", retrieve)
    graph.add_node("reranker", lambda state: rerank(state, llm=llm))
    graph.add_node("answer_generator", lambda state: generate_answer(state, llm=llm))
    graph.add_node("citation_formatter", format_citations)

    graph.set_entry_point("query_classifier")
    graph.add_conditional_edges(
        "query_classifier",
        _route_after_classification,
        {"retriever": "retriever", "answer_generator": "answer_generator"},
    )
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "answer_generator")
    graph.add_edge("answer_generator", "citation_formatter")
    graph.add_edge("citation_formatter", END)

    return graph.compile()


def run_agent(query: str, date_filter: Optional[str] = None, llm=None) -> dict:
    """Run the agent end-to-end and return {"answer": str, "sources": list[dict]}."""
    app = build_graph(llm=llm)
    result = app.invoke({"query": query, "date_filter": date_filter})
    return {
        "answer": result.get("answer", NO_CONTEXT_MESSAGE),
        "sources": result.get("sources", []),
    }
