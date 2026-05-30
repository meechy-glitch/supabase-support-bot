# Supabase Support Bot

**An AI support assistant that answers questions from Supabase's documentation in seconds — and links to the exact docs page behind every answer.**

🔗 **Live demo:** https://supabase-support-bot.onrender.com

> _First load may take ~30s — the demo runs on a free instance that sleeps when idle._

<!-- Add a screenshot or GIF of the widget answering a question here.
     Drag an image into the GitHub README editor, or commit it to /assets and link it:
     ![Demo](assets/demo.gif) -->

## What it does

Developers integrating a tool burn a lot of time hunting through documentation for answers that are already written down. This assistant reads a product's docs, finds the most relevant sections for any question, and answers in plain language — grounded entirely in the documentation, with citations. If the answer isn't in the docs, it says so rather than inventing one.

It's built here as a demo over **Supabase's public documentation** (~119 pages, ~970 indexed sections). The same system can be trained on any product's docs.

## How it works

A retrieval-augmented generation (RAG) pipeline:

1. **Ingest** — scrapes the documentation from its sitemap, splits it into overlapping chunks, embeds each chunk, and stores them in a vector database. Runs once, offline; the vector store ships with the app.
2. **Retrieve** — embeds the user's question and pulls the most semantically relevant chunks.
3. **Generate** — passes those chunks to an LLM with strict instructions to answer *only* from the provided context, then returns the answer plus the source URLs it used.

## Stack

| Layer | Choice |
|-------|--------|
| API | FastAPI (async) — `/chat` and `/health`, CORS-enabled |
| Vector store | Chroma (bring-your-own-embeddings) |
| Embeddings + generation | Google Gemini |
| Frontend | Embeddable vanilla-JS chat widget |
| Tests | pytest |
| Deploy | Docker on Render |

## Run locally

```bash
git clone https://github.com/meechy-glitch/supabase-support-bot.git
cd supabase-support-bot
python3.11 -m venv venv && source venv/bin/activate
make install
```

Add a free [Google AI Studio](https://aistudio.google.com) key to `.env`:

```bash
GEMINI_API_KEY=your_key_here
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001
TOP_K=5
```

Then build the index and run it:

```bash
make ingest    # scrape + embed the docs into the vector store (one-time)
make serve     # start the API + widget at http://localhost:8000
make test      # run the test suite
```

## A note on the free tier

The embedding model's free tier caps at 1,000 requests/day, so the full ingest is split across runs. `make resume` skips already-indexed pages and embeds only what's left, so you can finish the index over two days without re-spending quota. The deployed app only *reads* the vector store — it never re-scrapes — so day-to-day running cost is effectively zero.

---

### Want one trained on your product's docs?

This is a working template for any documentation-heavy product. If you'd like a support assistant trained on your own docs, reach out — [LinkedIn](https://www.linkedin.com/in/mitchell-ebosele-148656292).
