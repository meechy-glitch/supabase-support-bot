# Supabase Support Bot — Project Context

## What this is
A RAG demo: a FastAPI service that answers questions over Supabase's public
documentation, with an embeddable chat widget. It's a sales demo to show B2B
SaaS founders I can build a docs-trained support bot for their product.

## Hard constraints (do not violate)
- ZERO cost. Free tiers only. No paid APIs, no paid infra.
- Must run inside Render's free tier (512MB RAM). Keep the runtime lean.
- LLM + embeddings = Google Gemini via the `google-genai` SDK (`from google import genai`).
  NEVER use `google.generativeai` (that import is deprecated and will fail).
- Use Gemini Flash models only (Pro is paywalled). Model strings live in .env.
- Chroma is used in bring-your-own-embeddings mode: we pass precomputed embeddings
  and query by embedding. Do NOT use Chroma's default embedding function (it would
  pull a local model and bloat the runtime).
- Ingestion runs LOCALLY and is committed to the repo (data/chroma/). The deployed
  server only READS the vector store. Never scrape at request time.

## Code conventions
- Python 3.11 runtime. Use `str | None` union syntax. NEVER `Optional[str]`,
  never `from typing import Optional`.
- Fully type-hinted, clean, production-quality. Small focused modules.
- Use pydantic-settings for config loaded from environment variables.
- Async FastAPI endpoints.

## Architecture
- app/config.py     -> settings from env (Gemini key, model names, TOP_K, etc.)
- app/llm.py        -> Gemini client wrapper: embed_text() and generate()
- app/vectorstore.py-> Chroma client wrapper: query by embedding
- app/rag.py        -> retrieve top-k chunks, build grounded prompt, generate answer
- app/main.py       -> FastAPI app: /health (GET+HEAD), /chat (POST), serves widget
- ingest/scrape.py  -> fetch sitemap, fetch pages, extract main text
- ingest/chunk.py   -> chunk text with overlap (hand-rolled, no langchain)
- ingest/ingest.py  -> scrape -> chunk -> embed -> store in Chroma
- static/widget.html-> embeddable chat widget demo page
- tests/            -> pytest

## Behavior rules for the bot
- Answer ONLY from retrieved Supabase docs context. If the context doesn't cover it,
  say so plainly — do not invent. Ground every answer in the retrieved chunks.

## Commands
- Ingest (local, once):  python -m ingest.ingest
- Run server:            uvicorn app.main:app --reload
- Run tests:             pytest -q

## Verify-before-coding rule
The Gemini SDK method signatures and current model strings change. Before writing
any code that calls Gemini, confirm the correct `google-genai` usage and current
Flash + embedding model names with a tiny smoke test, then proceed. Do not assume.