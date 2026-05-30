from dataclasses import dataclass

from app.llm import embed_text, generate
from app.vectorstore import RetrievedChunk, query_by_embedding

SYSTEM_INSTRUCTION = (
    "You are a support assistant for Supabase. "
    "Answer the user's question using ONLY the Supabase documentation excerpts provided as CONTEXT. "
    "If the answer is not in the context, reply plainly that the docs you have don't cover it; "
    "do not invent APIs, flags, or behavior. "
    "Be concise, prefer short paragraphs and code blocks when helpful, and reference concrete steps."
)


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[str]


def _build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] SOURCE: {c.source}\nTITLE: {c.title}\n---\n{c.text}")
    context = "\n\n".join(blocks) if blocks else "(no documentation excerpts were retrieved)"
    return (
        f"CONTEXT (Supabase docs excerpts):\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using ONLY the context above. If the context does not contain the answer, say so."
    )


def answer_question(question: str) -> Answer:
    embedding = embed_text(question)
    chunks = query_by_embedding(embedding)
    prompt = _build_user_prompt(question, chunks)
    text = generate(SYSTEM_INSTRUCTION, prompt)
    sources: list[str] = []
    for c in chunks:
        if c.source and c.source not in sources:
            sources.append(c.source)
    return Answer(answer=text.strip(), sources=sources)
