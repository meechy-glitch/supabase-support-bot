"""Local-only ingest: scrape Supabase docs, chunk, embed, store in Chroma.

Run with:  python -m ingest.ingest
Optional:  python -m ingest.ingest --limit 5   # for a quick pilot
"""
from __future__ import annotations

import argparse
import sqlite3
import time

import chromadb
from google.genai import errors as genai_errors

from app.config import settings
from app.llm import embed_text
from ingest.chunk import chunk_text
from ingest.scrape import crawl_doc_urls, fetch_page_text, _session

EMBED_DELAY_SECONDS = 0.15  # respect free-tier rate limits
EST_CHUNKS_PER_PAGE = 8  # rough projection for dry-run cost estimates


def _is_quota_error(exc: Exception) -> bool:
    """True for a Gemini free-tier quota/rate exhaustion (HTTP 429)."""
    return isinstance(exc, genai_errors.APIError) and getattr(exc, "code", None) == 429


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
    probe: str = "how do I create a storage bucket",
) -> bool:
    """Embed a probe about a newly-indexed topic and confirm the persisted index
    returns the storage-bucket page in its top 3 results. Prints the top sources
    and a PASS/FAIL line; PASS requires the index be synced to disk AND a source
    matching /storage/buckets or containing 'creating-buckets'.
    """
    synced = _vector_index_synced()
    hits = collection.query(
        query_embeddings=[embed_text(probe)],
        n_results=3,
        include=["metadatas"],
    )
    sources = [m.get("source", "") for m in hits["metadatas"][0]]
    relevant = any("/storage/buckets" in s or "creating-buckets" in s for s in sources)
    ok = synced and relevant
    print(f"[verify] vector index synced to disk: {synced}")
    print(f"[verify] probe {probe!r} -> top 3 sources:")
    for src in sources:
        print(f"           {src}")
    print(f"[verify] {'PASS' if ok else 'FAIL'}: storage-bucket page retrievable from persisted index")
    return ok


def run(
    limit: int | None = None,
    reset: bool = False,
    repair: bool = False,
    dry_run: bool = False,
) -> None:
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
    print(f"[crawl] discovering /docs URLs (depth={settings.depth}, max_pages={settings.max_pages})")
    urls = crawl_doc_urls(session)
    if limit is not None:
        urls = urls[:limit]

    existing_ids = set(collection.get(include=[])["ids"])
    new_urls = [u for u in urls if f"{u}#0" not in existing_ids]
    already_indexed = len(urls) - len(new_urls)
    # Prioritize the known gap: embed /storage/ pages first so they land inside
    # the day-1 quota window even if a 429 cuts the run short. Stable sort keeps
    # BFS order within each group.
    new_urls.sort(key=lambda u: 0 if "/storage/" in u else 1)
    print(f"[crawl] discovered {len(urls)} URL(s); {already_indexed} already indexed, {len(new_urls)} new")

    if dry_run:
        est_chunks = len(new_urls) * EST_CHUNKS_PER_PAGE
        print("[dry-run] no embedding will be performed.")
        print(f"[dry-run] new URLs not yet in index: {len(new_urls)}")
        print(f"[dry-run] projected new chunks (~{EST_CHUNKS_PER_PAGE}/page): {est_chunks}")
        print(f"[dry-run] projected Gemini embedding calls: {est_chunks}")
        print("[dry-run] projected cost: $0.00 (Gemini free tier; cost is rate-limited quota, not dollars)")
        print("[dry-run] sample of new URLs:")
        for url in new_urls[:15]:
            print(f"           {url}")
        if len(new_urls) > 15:
            print(f"           ... and {len(new_urls) - 15} more")
        return

    total_chunks = 0
    newly_embedded = 0
    stopped_early = False
    stop_reason = ""
    for i, url in enumerate(new_urls, start=1):
        page = fetch_page_text(url, session)
        if page is None:
            print(f"[{i}/{len(new_urls)}] skip (empty): {url}")
            continue
        title, text = page
        chunks = chunk_text(text)
        if not chunks:
            print(f"[{i}/{len(new_urls)}] skip (no chunks): {url}")
            continue

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []
        try:
            for j, ch in enumerate(chunks):
                ids.append(f"{url}#{j}")
                documents.append(ch)
                embeddings.append(embed_text(ch))
                metadatas.append({"source": url, "title": title, "chunk_index": j})
                time.sleep(EMBED_DELAY_SECONDS)
        except genai_errors.APIError as exc:
            # Reaches here only after embed_text() has exhausted its retries, so
            # this is a sustained failure (daily quota at 429, or a prolonged
            # outage at 5xx). Drop this partial page (never upserted, so url#0
            # stays absent and `make resume` re-embeds it cleanly) and stop
            # before burning further failed calls.
            stop_reason = "quota" if _is_quota_error(exc) else f"service error {getattr(exc, 'code', '?')}"
            print(f"[{i}/{len(new_urls)}] Gemini {stop_reason} on {url} — stopping cleanly")
            stopped_early = True
            break

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        newly_embedded += 1
        print(f"[{i}/{len(new_urls)}] {title[:60]!r:62} chunks={len(chunks)} (total={total_chunks})")

    if total_chunks:
        # Persist the tail of this run: a batch smaller than the sync threshold
        # would otherwise stay in the WAL and never reach the on-disk snapshot.
        flush_vector_index(collection)
    remaining = len(new_urls) - newly_embedded
    print(f"[ingest] stopped early ({stop_reason})." if stopped_early else "[ingest] done.")
    print(f"[ingest]   total URLs discovered: {len(urls)}")
    print(f"[ingest]   already indexed:       {already_indexed}")
    print(f"[ingest]   newly embedded pages:  {newly_embedded} ({total_chunks} chunks)")
    print(f"[ingest]   remaining new pages:   {remaining}")
    print(f"[ingest]   final collection count: {collection.count()}")
    if stopped_early:
        when = ("tomorrow after the Gemini quota resets" if stop_reason == "quota"
                else "once Gemini recovers (a 5xx is transient — likely shortly)")
        print(f"[ingest] {remaining} page(s) still unembedded. Run `make resume` {when} "
              "— skip-already-indexed will pick up where this left off.")
    # verify_retrieval embeds a probe; skip it when we stopped on exhausted quota
    # (there's no embed budget left) — run it on the next successful resume.
    if total_chunks and not (stopped_early and stop_reason == "quota"):
        verify_retrieval(collection)
    elif stopped_early and stop_reason == "quota":
        print("[ingest] skipping verify_retrieval: embed quota exhausted. "
              "It will run automatically on the next `make resume`.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="cap number of pages")
    p.add_argument("--reset", action="store_true", help="wipe collection before ingesting")
    p.add_argument(
        "--repair",
        action="store_true",
        help="re-persist the HNSW index from stored embeddings (no scrape, no embedding cost)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="crawl and report new URLs + projected embedding cost without embedding",
    )
    args = p.parse_args()
    run(limit=args.limit, reset=args.reset, repair=args.repair, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
