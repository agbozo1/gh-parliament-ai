# Current State (V1) — Architecture Assessment

This document captures the state of the codebase on `main` before the V2
upgrade, so the rationale for every V2 decision is traceable back to a
concrete gap. V1 is preserved untouched on `main`; all V2 work happens on a
separate branch.

## 1. Current Architecture

```
[Streamlit page 1]         [Streamlit page 2]              [Streamlit page 3]
Download Briefs      →     Train Model                →    Query Briefs
     |                          |                                |
     v                          v                                v
requests.get(url)         pdfplumber.extract_text     FAISS.load_local(...)
save to proceedings/      RecursiveCharacterTextSplitter    + OllamaEmbeddings
                           (chunk_size=512, overlap=50)     + OllamaLLM (llama3.1)
                           OllamaEmbeddings (all-minilm)    RetrievalQA.from_llm
                           FAISS.from_documents(...)             |
                           vector_db.save_local(              persist_queries()
                             "parliament_faiss_db_allminilm")  -> persisted_queries.csv
```

**Scraper (`pages/1 - Download Briefs.py`)**
- Not Selenium — there is no dynamic page rendering at all. The Parliament
  Hansard PDF URL is fully predictable from a date
  (`https://www.parliament.gh/epanel/docs/pb/{day}{suffix} {month}, {year}.pdf`),
  so the "scraper" is a plain `requests.get` in a loop over a date range.
- No retry logic — a single failed request is silently skipped (only a 200
  status code is saved).
- No logging of scrape runs (no timestamps, no record of which URLs were
  hit vs. missed).
- Base URL, output folder, and request delay are all hardcoded.

**Storage**
- PDFs land in `proceedings/` (flat directory, filename = human-readable
  date, e.g. `11th February, 2025.pdf`). No metadata sidecar — the filename
  is the only structured data captured (source date), and it's re-parsed
  from string content, not stored as structured fields.

**Chunking / Indexing (`pages/2 - Train Model.py`)**
- `pdfplumber` extracts raw text per PDF, no cleaning (headers/footers,
  page-break artifacts, whitespace runs are not stripped).
- `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)`.
- Only metadata stored per chunk: `{"source": filename}`. No source URL,
  no parsed date field, no session/sitting identifier.
- Embeddings: `OllamaEmbeddings(model="all-minilm:latest")` — requires a
  local Ollama daemon running the model; no cloud embedding option, no
  provider abstraction.
- Vector store: FAISS, saved to a hardcoded local folder
  (`parliament_faiss_db_allminilm`) via `save_local`/`load_local`. No
  persistent server, no metadata filtering support at query time (FAISS's
  default LangChain wrapper does simple similarity search only).

**Retrieval / Query (`pages/3 - Query Briefs.py`, `app.py`)**
- `RetrievalQA.from_llm` — a single-shot LangChain chain, not an agent.
  No query classification, no reranking, no citation formatting, no
  refusal behavior when retrieval returns nothing relevant (the chain will
  still ask the LLM to answer from whatever it retrieved, however
  irrelevant).
- LLM: `OllamaLLM(model="llama3.1:latest")` — local only, no API-based
  provider option, not configurable via environment variable.
- Query interface: Streamlit UI only. No REST API, no programmatic access.
- "Persistence" of queries is a CSV file (`persisted_queries.csv`) written
  with pandas — not a database, not queryable, grows unbounded, and is
  duplicated almost verbatim between `app.py` and page 3.

**Dependencies (`requirements.txt`)**
- Lists `langchain`, `langchain_ollama`, `pandas`, `streamlit`, `torch`,
  `transformers` — but not `pdfplumber` or `requests`, both of which are
  imported directly. The file is incomplete relative to actual imports.
- `torch==1.8.1` is from 2021 and predates most current sentence-transformer
  releases; `transformers==4.32.0.dev0` is a dev/pre-release pin. Both are
  almost certainly unpinned-in-practice (whatever gets resolved at install
  time), which makes the environment non-reproducible.

## 2. What Is Working Well and Should Be Preserved

- **The date → URL construction logic** (`build_parliament_pdf_url` /
  `get_day_suffix`) is correct and directly reusable — the Parliament site's
  naming convention is genuinely this predictable, so V2 keeps this as the
  core of `pdf_downloader.py` rather than replacing it with Selenium
  scraping for scraping's sake.
- **The overall pipeline shape** (download → extract → chunk → embed →
  index → retrieve → answer) is the right shape; it just needs each stage
  hardened and decoupled from Streamlit.
- **RecursiveCharacterTextSplitter** as the chunking strategy is a
  reasonable choice and is kept in V2 (with revised size/overlap per the
  V2 spec: 800/100 instead of 512/50, to keep more contiguous parliamentary
  context per chunk).
- The local, no-API-key embedding option (sentence-transformers /
  all-MiniLM) is a good fallback path and is preserved in V2 as the
  `huggingface` embedding provider.

## 3. What Is Missing or Weak vs. the Target Stack

- No retry/backoff on network calls (PDF downloads silently drop on any
  transient failure).
- No structured logging anywhere.
- No environment-variable configuration — every path, model name, and URL
  is a literal in source.
- No metadata beyond `source` filename — can't filter by date range or
  session at query time.
- No dedicated vector database — FAISS local files aren't a service, can't
  be queried concurrently, and don't support rich metadata filtering.
- No agentic retrieval flow — a single LangChain `RetrievalQA` chain with
  no branching, no reranking, no grounding check, no citation attachment.
- No hallucination guardrail — the chain will answer even with irrelevant
  or empty context.
- No REST API — only a Streamlit UI, so nothing else can integrate with it.
- No tests of any kind.
- No CI/CD.
- No containerization (Dockerfile/docker-compose).
- No cloud LLM/embedding provider support — hard dependency on a locally
  running Ollama daemon, which doesn't fit a hosted deployment target.
- Query history persistence via CSV is not concurrency-safe and not
  queryable.

## 4. Technical Debt / Hardcoded Values to Clean Up

- Hardcoded FAISS folder name `parliament_faiss_db_allminilm` (typo-prone,
  repeated in three files).
- Hardcoded model names `all-minilm:latest` / `llama3.1:latest` in three
  places (`app.py`, page 2, page 3) instead of one config point.
- Hardcoded `proceedings` folder name repeated across files.
- `persist_queries` function duplicated verbatim in `app.py` and
  `pages/3 - Query Briefs.py`.
- `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` is a workaround for an OpenMP
  library conflict rather than a real fix — a sign the dependency set
  (torch/transformers pins) needs to be resolved properly instead.
- `requirements.txt` missing direct dependencies (`pdfplumber`, `requests`).
- No `.gitignore` entries visible for the checked-in `proceedings/*.pdf`
  binaries or the FAISS index files — large binary artifacts are committed
  directly to the repo.
