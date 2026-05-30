# Supabase Support Bot

A RAG demo: a FastAPI service that answers questions about Supabase using its
public documentation as the only source of truth, plus an embeddable chat
widget. Built as a sales demo to show how a docs-trained support bot looks for
a real B2B SaaS product.

- **LLM + embeddings:** Google Gemini (Flash + embedding-001) via `google-genai`
- **Vector store:** Chroma, bring-your-own-embeddings, committed to the repo
- **Runtime:** FastAPI, designed to fit on Render's free tier (512MB RAM)

The bot answers ONLY from retrieved Supabase docs context and says so plainly
when the docs don't cover a question — no hallucinated APIs.

## Architecture

```
app/config.py      Settings loaded from .env via pydantic-settings
app/llm.py         Gemini client wrapper: embed_text() and generate()
app/vectorstore.py Chroma query by embedding
app/rag.py         Retrieve top-k, build grounded prompt, generate answer
app/main.py        FastAPI: /health, /chat, serves static/widget.html at /

ingest/scrape.py   Sitemap + page fetch (article selector)
ingest/chunk.py    ~800 char chunks with ~100 char overlap
ingest/ingest.py   scrape -> chunk -> embed -> store in Chroma

static/widget.html Embeddable chat widget (vanilla JS, no build step)
data/chroma/       Persisted vector store (committed)
tests/             pytest: chunker, /health, /chat (Gemini mocked)
```

## Local setup

Requires **Python 3.11**.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
GEMINI_API_KEY=...your free-tier Gemini key...
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001
TOP_K=5
DOCS_SITEMAP=https://supabase.com/docs/sitemap.xml
MAX_PAGES=200
```

## Ingest (runs locally, commit the result)

The vector store is built locally and committed to the repo. The deployed
server only **reads** from it — it never scrapes at request time.

```bash
# pilot: 5 pages
python -m ingest.ingest --limit 5 --reset

# full ingest from scratch (wipes the collection first)
python -m ingest.ingest --reset

# idempotent top-up (skips URLs whose chunks are already in Chroma)
python -m ingest.ingest
# or equivalently:
make resume
```

`data/chroma/` will be created/updated. Commit it.

### Quota note

Gemini's free tier caps `gemini-embedding-001` at **1000 embed_content
requests per day**. A full 200-page ingest produces roughly that many chunks,
so the ingest will hit the daily limit. When it does, the script raises a
`google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED` and exits.

The ingest is **idempotent**: each chunk is stored with id `f"{url}#{j}"`, and
the script skips any URL whose `url#0` is already present in the collection.
So after the quota resets (~24h), simply re-run `make resume` (or
`python -m ingest.ingest`) to top up the remaining pages without re-embedding
the ones already indexed.

## Run the server locally

```bash
uvicorn app.main:app --reload
```

- `http://localhost:8000/`         → chat widget
- `GET /health`                    → `{"status":"ok"}`
- `POST /chat` `{"message": str}`  → `{"answer": str, "sources": [url, ...]}`

## Tests

```bash
pytest -q
```

The Gemini call inside `/chat` is mocked, so tests do not consume API quota.

## Deploy to Render (free tier)

1. Push the repo to GitHub. Make sure `data/chroma/` is committed.
2. On Render, create a new **Web Service** from the repo.
3. Render auto-detects the `Dockerfile`. No build command override needed.
4. Set environment variables in Render:
   - `GEMINI_API_KEY`
   - `GEMINI_CHAT_MODEL=gemini-2.5-flash`
   - `GEMINI_EMBED_MODEL=gemini-embedding-001`
   - `TOP_K=5` (optional)
5. Render injects `PORT`; the Dockerfile binds uvicorn to `0.0.0.0:${PORT}`.
6. Render's free tier health check works because `/health` accepts both GET and HEAD.

## Constraints (by design)

- Zero cost — free tiers only, no paid APIs or paid infra.
- Gemini Flash models only (Pro is paywalled).
- Chroma in bring-your-own-embeddings mode — never pulls a local embedding
  model at runtime, so the deployed runtime stays lean.
