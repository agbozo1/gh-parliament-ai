from types import SimpleNamespace

from langchain_core.documents import Document

from agent.graph import NO_CONTEXT_MESSAGE, build_graph, run_agent


class FakeLLM:
    """Routes canned responses based on which prompt template was used."""

    def __init__(self, classification="RETRIEVAL", scores=None, answer="This is the answer."):
        self.classification = classification
        self.scores = scores or []
        self._score_iter = iter(self.scores)
        self.answer = answer
        self.calls = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        if "Classify the following user query" in prompt:
            return SimpleNamespace(content=self.classification)
        if "how relevant is the following passage" in prompt:
            score = next(self._score_iter, 0)
            return SimpleNamespace(content=str(score))
        if "Answer the user's question" in prompt:
            return SimpleNamespace(content=self.answer)
        # DIRECT-path answer (query passed straight to the LLM)
        return SimpleNamespace(content=self.answer)


def _fake_docs():
    return [
        Document(
            page_content="The Minister of Finance presented the 2025 budget statement.",
            metadata={
                "pdf_filename": "11th February, 2025.pdf",
                "date": "2025-02-11",
                "source_url": "https://www.parliament.gh/epanel/docs/pb/11th%20February%2C%202025.pdf",
                "session_id": "2025",
            },
        ),
        Document(
            page_content="Members debated road infrastructure funding.",
            metadata={
                "pdf_filename": "12th February, 2025.pdf",
                "date": "2025-02-12",
                "source_url": "https://www.parliament.gh/epanel/docs/pb/12th%20February%2C%202025.pdf",
                "session_id": "2025",
            },
        ),
    ]


def test_retrieval_path_answers_with_citations(mocker):
    mocker.patch("agent.tools.retriever.search", return_value=_fake_docs())
    llm = FakeLLM(
        classification="RETRIEVAL",
        scores=[9, 8],
        answer="The budget statement was presented in February 2025.",
    )

    graph = build_graph(llm=llm)
    result = graph.invoke({"query": "What did the budget statement cover?", "date_filter": None})

    assert result["needs_retrieval"] is True
    assert result["answer"] == "The budget statement was presented in February 2025."
    assert len(result["sources"]) == 2
    assert result["sources"][0]["document"] == "11th February, 2025.pdf"
    assert result["sources"][0]["date"] == "2025-02-11"


def test_retrieval_path_prompt_discourages_over_cautious_refusal(mocker):
    # Regression test: real reranked chunks are often short, disjointed
    # Hansard fragments rather than a tidy narrative, and the answer prompt
    # was refusing even when the reranker had already confirmed relevance.
    # Assert the prompt actually sent to the model tells it to summarize
    # partial/fragmentary context instead of demanding a complete answer
    # before it'll say anything.
    mocker.patch("agent.tools.retriever.search", return_value=_fake_docs())
    llm = FakeLLM(classification="RETRIEVAL", scores=[9, 8], answer="Answer.")

    graph = build_graph(llm=llm)
    graph.invoke({"query": "What did the budget statement cover?", "date_filter": None})

    answer_prompt = next(p for p in llm.calls if "Answer the user's question" in p)
    assert "partial or fragmentary" in answer_prompt
    assert "do not withhold an answer" in answer_prompt.lower()


def test_direct_path_skips_retrieval_entirely(mocker):
    search_mock = mocker.patch("agent.tools.retriever.search", return_value=_fake_docs())
    llm = FakeLLM(
        classification="DIRECT",
        answer="Hello! I can help you explore Ghana's Hansard records.",
    )

    graph = build_graph(llm=llm)
    result = graph.invoke({"query": "Hi, what can you do?", "date_filter": None})

    assert result["needs_retrieval"] is False
    assert result["answer"] == "Hello! I can help you explore Ghana's Hansard records."
    assert result["sources"] == []
    search_mock.assert_not_called()


def test_direct_path_grounds_the_llm_with_assistant_identity(mocker):
    mocker.patch("agent.tools.retriever.search", return_value=_fake_docs())
    llm = FakeLLM(classification="DIRECT", answer="Hi there!")

    graph = build_graph(llm=llm)
    graph.invoke({"query": "what conversations do people have in general?", "date_filter": None})

    direct_prompt = llm.calls[-1]
    assert "Ghana Parliament Hansard research assistant" in direct_prompt
    assert "general-purpose AI assistant" in direct_prompt


def test_refuses_when_no_chunks_survive_reranking(mocker):
    mocker.patch("agent.tools.retriever.search", return_value=_fake_docs())
    # Both chunks score below the relevance threshold.
    llm = FakeLLM(classification="RETRIEVAL", scores=[1, 0])

    graph = build_graph(llm=llm)
    result = graph.invoke({"query": "What is the capital of France?", "date_filter": None})

    assert result["reranked"] == []
    assert result["answer"] == NO_CONTEXT_MESSAGE
    assert result["sources"] == []


def test_refuses_when_retrieval_returns_nothing(mocker):
    mocker.patch("agent.tools.retriever.search", return_value=[])
    llm = FakeLLM(classification="RETRIEVAL")

    graph = build_graph(llm=llm)
    result = graph.invoke({"query": "Anything about the 1990s?", "date_filter": None})

    assert result["retrieved"] == []
    assert result["answer"] == NO_CONTEXT_MESSAGE
    assert result["sources"] == []


def test_run_agent_resolves_the_llm_once_and_reuses_it_across_all_nodes(mocker):
    # query_classifier, reranker, and answer_generator each fall back to
    # get_llm() when no llm is threaded through -- run_agent must resolve
    # it exactly once per request so a single query can't inconsistently
    # mix providers (get_llm() picks randomly among configured keys).
    mocker.patch("agent.tools.retriever.search", return_value=_fake_docs())
    fake_llm = FakeLLM(classification="RETRIEVAL", scores=[9, 8], answer="Answer.")
    get_llm_mock = mocker.patch("agent.graph.get_llm", return_value=fake_llm)

    result = run_agent("What did the budget statement cover?")

    get_llm_mock.assert_called_once()
    assert result["answer"] == "Answer."


def test_citation_formatter_dedupes_by_document(mocker):
    dup_docs = [_fake_docs()[0], _fake_docs()[0]]
    mocker.patch("agent.tools.retriever.search", return_value=dup_docs)
    llm = FakeLLM(
        classification="RETRIEVAL",
        scores=[9, 9],
        answer="Answer grounded in one document.",
    )

    graph = build_graph(llm=llm)
    result = graph.invoke({"query": "Tell me about the budget", "date_filter": None})

    assert len(result["sources"]) == 1
