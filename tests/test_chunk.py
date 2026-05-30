from ingest.chunk import chunk_text


def test_empty_string_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


def test_short_text_returns_single_chunk():
    short = "Just a few words about Supabase auth."
    assert chunk_text(short) == [short]


def test_long_text_is_split_with_overlap():
    paragraph = "This is one sentence about Supabase Row Level Security policies. "
    text = (paragraph * 50).strip()  # ~3300 chars
    chunks = chunk_text(text, chunk_size=800, overlap=100)

    assert len(chunks) >= 4
    for c in chunks:
        assert len(c) <= 800
        assert c.strip() == c

    # overlap: each non-first chunk should share the start with the tail of the previous one
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-80:]
        # tail words should appear at the head of the next chunk
        assert any(w in chunks[i][:200] for w in prev_tail.split() if len(w) > 4)


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)
