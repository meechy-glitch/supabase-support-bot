"""Local-only ingest: scrape Supabase docs, chunk, embed, store in Chroma.

Run with:  python -m ingest.ingest
Optional:  python -m ingest.ingest --limit 5   # for a quick pilot
"""
from __future__ import annotations

import argparse
import time

import chromadb

from app.config import settings
from app.llm import embed_text
from ingest.chunk import chunk_text
from ingest.scrape import fetch_doc_urls, fetch_page_text, _session

EMBED_DELAY_SECONDS = 0.15  # respect free-tier rate limits


def _get_collection() -> chromadb.api.models.Collection.Collection:
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def run(limit: int | None = None, reset: bool = False) -> None:
    collection = _get_collection()
    if reset:
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        client.delete_collection(settings.chroma_collection)
        collection = _get_collection()

    session = _session()
    urls = fetch_doc_urls(session)
    if limit is not None:
        urls = urls[:limit]

    existing_ids = set(collection.get(include=[])["ids"])
    print(f"[ingest] will process {len(urls)} url(s); {len(existing_ids)} chunks already indexed")
    total_chunks = 0
    for i, url in enumerate(urls, start=1):
        if f"{url}#0" in existing_ids:
            print(f"[{i}/{len(urls)}] skip (already indexed): {url}")
            continue
        page = fetch_page_text(url, session)
        if page is None:
            print(f"[{i}/{len(urls)}] skip (empty): {url}")
            continue
        title, text = page
        chunks = chunk_text(text)
        if not chunks:
            print(f"[{i}/{len(urls)}] skip (no chunks): {url}")
            continue

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []
        for j, ch in enumerate(chunks):
            ids.append(f"{url}#{j}")
            documents.append(ch)
            embeddings.append(embed_text(ch))
            metadatas.append({"source": url, "title": title, "chunk_index": j})
            time.sleep(EMBED_DELAY_SECONDS)

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"[{i}/{len(urls)}] {title[:60]!r:62} chunks={len(chunks)} (total={total_chunks})")

    print(f"[ingest] done. total chunks indexed: {total_chunks}")
    print(f"[ingest] collection count: {collection.count()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="cap number of pages")
    p.add_argument("--reset", action="store_true", help="wipe collection before ingesting")
    args = p.parse_args()
    run(limit=args.limit, reset=args.reset)


if __name__ == "__main__":
    main()
