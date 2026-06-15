"""Local-only ingest: scrape Supabase docs, chunk, embed, store in Chroma.

Run with:  python -m ingest.ingest
Optional:  python -m ingest.ingest --limit 5   # for a quick pilot
"""
from __future__ import annotations

import argparse
import sqlite3
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


def _vector_index_synced() -> bool:
    """True when the on-disk HNSW snapshot covers every record in sqlite.

    Chroma tracks a `max_seq_id` high-water mark per segment. The METADATA
    (sqlite) segment commits on every write; the VECTOR (HNSW) segment only
    advances once its snapshot is flushed to the .bin files. If the vector
    high-water mark trails the metadata one, the newest chunks live only in the
    write-ahead log and are absent from the durable, deployable index.
    """
    db = settings.chroma_path / "chroma.sqlite3"
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute(
            "SELECT s.scope, m.seq_id "
            "FROM max_seq_id m JOIN segments s ON s.id = m.segment_id"
        ).fetchall()
    finally:
        con.close()
    by_scope = {scope: seq for scope, seq in rows}
    vector = by_scope.get("VECTOR")
    metadata = by_scope.get("METADATA")
    if vector is None or metadata is None:
        return False
    return vector >= metadata


def flush_vector_index(collection: chromadb.api.models.Collection.Collection) -> None:
    """Force the local HNSW snapshot to persist every stored vector to disk.

    The local-persisted vector segment only writes its .bin snapshot after
    `hnsw:sync_threshold` (default 1000) un-synced records accumulate. A resume
    that adds fewer than that leaves the new vectors in the write-ahead log
    only: present in sqlite and in the in-memory index (rebuilt from the WAL on
    load) but missing from the on-disk snapshot, so a freshly started/deployed
    server cannot retrieve them durably.

    Re-upserting the already-stored records (embeddings are read back from the
    store, so this costs zero embedding-API quota) pushes the un-synced count
    past the threshold, which triggers a full snapshot of the in-memory index
    to disk. This both repairs an index left stale by an earlier short resume
    and guarantees the tail of the current run is persisted before we exit.
    """
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    ids = data["ids"]
    if not ids:
        return
    collection.upsert(
        ids=ids,
        embeddings=[list(e) for e in data["embeddings"]],
        documents=data["documents"],
        metadatas=data["metadatas"],
    )


def verify_retrieval(
    collection: chromadb.api.models.Collection.Collection,
    probe: str = "How do I create a storage bucket?",
) -> bool:
    """Embed a probe about a newly-indexed topic and confirm the persisted
    index returns a relevant chunk. Prints a PASS/FAIL line and returns the
    result so callers can set an exit status.
    """
    synced = _vector_index_synced()
    hits = collection.query(
        query_embeddings=[embed_text(probe)],
        n_results=settings.top_k,
        include=["documents", "metadatas"],
    )
    sources = [m.get("source", "") for m in hits["metadatas"][0]]
    relevant = any("storage" in s for s in sources)
    ok = synced and relevant
    print(f"[verify] vector index synced to disk: {synced}")
    print(f"[verify] probe {probe!r} -> top sources:")
    for src in sources:
        print(f"           {src}")
    print(f"[verify] {'PASS' if ok else 'FAIL'}: new chunks searchable from persisted index")
    return ok


def run(limit: int | None = None, reset: bool = False, repair: bool = False) -> None:
    collection = _get_collection()
    if repair:
        print("[repair] forcing HNSW snapshot to persist all stored vectors")
        flush_vector_index(collection)
        print(f"[repair] done. collection count: {collection.count()}")
        verify_retrieval(collection)
        return
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

    if total_chunks:
        # Persist the tail of this run: a batch smaller than the sync threshold
        # would otherwise stay in the WAL and never reach the on-disk snapshot.
        flush_vector_index(collection)
    print(f"[ingest] done. total chunks indexed: {total_chunks}")
    print(f"[ingest] collection count: {collection.count()}")
    if total_chunks:
        verify_retrieval(collection)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="cap number of pages")
    p.add_argument("--reset", action="store_true", help="wipe collection before ingesting")
    p.add_argument(
        "--repair",
        action="store_true",
        help="re-persist the HNSW index from stored embeddings (no scrape, no embedding cost)",
    )
    args = p.parse_args()
    run(limit=args.limit, reset=args.reset, repair=args.repair)


if __name__ == "__main__":
    main()
