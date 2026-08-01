# 🏛️ Ghana Parliament Hansard RAG

A production-shaped Retrieval-Augmented Generation system over the Ghana
Parliament's Hansard (parliamentary debate) records. It scrapes published
sitting briefs, indexes them into a vector store, and answers natural-language
questions about parliamentary proceedings through an agentic LangGraph
pipeline that refuses to answer when it can't ground a response in the
actual record — built as a portfolio piece for AI/Forward-Deployed
Engineering work, and equally usable as a real civic-transparency tool.

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        A[page_scraper.py<br/>discover sitting dates] --> B[pdf_downloader.py<br/>fetch + store PDFs]
        B --> C[data/raw/*.pdf]
        C --> D[ingest.py<br/>extract + clean + chunk + embed]
        D --> E[(ChromaDB<br/>ghana_parliament_hansard)]
    end

    subgraph Query
        F[FastAPI /query] --> G[LangGraph agent]
        G --> H{query_classifier}
        H -- retrieval needed --> I[retriever]
        H -- answerable directly --> M[answer_generator]
        I --> E
        I --> J[reranker]
        J --> M
        M --> K[citation_formatter]
        K --> F
    end
```

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Orchestration | LangChain + LangGraph |
| Vector store | ChromaDB (embedded, or server via docker-compose) |
| Embeddings | OpenAI `text-embedding-3-small` or local `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | OpenAI, Anthropic (Claude), Mistral, or DeepSeek — whichever key is configured |
| API | FastAPI (async) |
| Frontend | Static HTML/CSS/JS (`web/`), served by FastAPI locally or standalone on Netlify |
| Scraping | `requests` (date-predictable URLs) + Selenium (listing-page discovery) |
| Containerization | Docker / docker-compose |
| CI | GitHub Actions (lint, test, docker build) |

## Quickstart

```bash
git clone https://github.com/agbozo1/gh-parliament-ai.git
cd gh-parliament-ai
cp .env.example .env
# edit .env: set at least one LLM provider key (OPENAI_API_KEY, ANTHROPIC_API_KEY,
# MISTRAL_API_KEY, or DEEPSEEK_API_KEY). EMBEDDING_PROVIDER=huggingface needs no key.

docker-compose up --build
```

Open **http://localhost:8000** for the query UI (or **http://localhost:8000/docs** for the raw Swagger API).

Trigger a scrape + ingest run, then ask a question:

```bash
curl -X POST http://localhost:8000/ingest -d '{"days_back": 60}' -H 'Content-Type: application/json'

curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What did Parliament discuss about the 2025 budget statement?"}'
```

Example response:

```json
{
  "answer": "On 11th February 2025, the Minister of Finance presented the 2025 budget statement to Parliament, covering revenue projections and planned expenditure...",
  "sources": [
    {
      "document": "11th February, 2025.pdf",
      "date": "2025-02-11",
      "url": "https://www.parliament.gh/epanel/docs/pb/11th%20February%2C%202025.pdf"
    }
  ],
  "session_id": "b3e1..."
}
```

If nothing relevant is indexed, the agent replies:
`"I could not find relevant information in the parliamentary records."`
instead of guessing.

### Local (no Docker)

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Without `CHROMA_HOST` set, ChromaDB runs embedded and persists to `data/chroma/`.
FastAPI serves the frontend at `/` from the same process — no separate dev server needed.

### Frontend (`web/`)

A minimal, dependency-free HTML/CSS/JS query UI — no build step, no framework.
FastAPI mounts it at `/` automatically whenever the `web/` directory is present,
so `uvicorn api.main:app` alone gets you a working UI locally.

To host it separately (e.g. **Netlify**, decoupled from the API — see
[Known limitations](#known-limitations--next-steps)):

1. Point Netlify at this repo with `netlify.toml` already set to
   `publish = "web"` — no build command needed.
2. Deploy the API itself elsewhere (Render/Fly.io/Railway/your own Docker host).
3. Edit `API_BASE` at the top of `web/app.js` to that API's URL before deploying
   (it defaults to same-origin, which only works when FastAPI serves both).

### Tests

```bash
pytest tests/ -v
```

No API keys are required — the LLM and vector store are mocked/stubbed in tests.

## Design decisions

**Why LangGraph over a single `RetrievalQA` chain.** A plain chain always
retrieves, always answers, and has no way to refuse. The graph adds a
routing step (skip retrieval for chit-chat), a reranking step (drop
low-relevance chunks before they reach the answer prompt), and a hard
refusal path when nothing relevant survives reranking — the single
highest-priority requirement for a system answering questions about
government records, where a confident hallucination is worse than "I
don't know."

**Why ChromaDB.** Local-first and file-persisted for zero-ops development,
but the same code talks to a server (`CHROMA_HOST`/`CHROMA_PORT`) for a
docker-compose or hosted deployment — no code change between environments,
and native metadata filtering (`date`, `session_id`) supports scoped
queries without a second index.

**Chunking strategy.** `RecursiveCharacterTextSplitter` at
`chunk_size=800, chunk_overlap=100`. Hansard text is dense, formal prose;
800 characters keeps a chunk to roughly one exchange or statement without
fragmenting mid-sentence as often as a smaller size would, and the overlap
prevents cutting a quoted figure or name across a chunk boundary.

**Multi-provider LLM selection.** Rather than hard-coding one vendor, the
app reads whichever of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
`MISTRAL_API_KEY` / `DEEPSEEK_API_KEY` is present and builds the matching
LangChain chat model; with several keys set it picks one at random per
process (or pin one via `LLM_PROVIDER`). This keeps the app deployable
wherever a key happens to be available, including free-tier-only
environments.

## Known limitations & next steps

- **Netlify hosting**: Netlify's serverless functions aren't a good fit for
  a stateful FastAPI app with a persistent vector store and a LangGraph
  agent loop — Netlify works well for the docs/marketing surface, but the
  API itself is best run on a container host (Render, Fly.io, Railway, or
  the included Dockerfile on any VM) with the `chromadb` service pointed at
  a persistent volume.
- The scraper's `page_scraper.py` assumes a listing page exists at a
  predictable path; if the Parliament site's structure changes, the
  date-based `pdf_downloader` fallback still works for known date ranges.
- Reranking is LLM-scored rather than a dedicated cross-encoder — cheaper
  to run, but adds one LLM call per candidate chunk. A local
  `sentence-transformers` cross-encoder would cut latency and cost at the
  expense of another model dependency.
- `/ingest` runs in a background thread with in-memory job tracking; a
  restart loses in-flight job status. A real deployment would move this to
  a task queue (Celery/RQ) with persistent job state.
- No authentication on the API — add it before exposing this beyond local/demo use.

## Credits

Built on publicly available records from the
[Parliament of Ghana](https://www.parliament.gh) to promote transparency
and civic engagement.

by: Ebenezer Agbozo, PhD.
