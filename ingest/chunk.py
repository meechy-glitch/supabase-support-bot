CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows, preferring paragraph boundaries within each window."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            split_at = window.rfind("\n\n")
            if split_at == -1 or split_at < chunk_size // 2:
                split_at = window.rfind("\n")
            if split_at == -1 or split_at < chunk_size // 2:
                split_at = window.rfind(". ")
                if split_at != -1:
                    split_at += 1
            if split_at != -1 and split_at >= chunk_size // 2:
                end = start + split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
