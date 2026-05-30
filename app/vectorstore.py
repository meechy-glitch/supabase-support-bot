from dataclasses import dataclass

import chromadb

from app.config import settings


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    title: str
    distance: float


_client: chromadb.api.ClientAPI | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(settings.chroma_path))
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def query_by_embedding(embedding: list[float], top_k: int | None = None) -> list[RetrievedChunk]:
    k = top_k or settings.top_k
    collection = _get_collection()
    res = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        RetrievedChunk(
            text=doc,
            source=meta.get("source", ""),
            title=meta.get("title", ""),
            distance=float(dist),
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]
