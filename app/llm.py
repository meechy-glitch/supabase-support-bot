import time

from groq import APIConnectionError, APIStatusError, Groq
from sentence_transformers import SentenceTransformer

from app.config import settings

# all-mpnet-base-v2 produces 768-dim embeddings, matching the vectors the Chroma
# index was built with (gemini-embedding-001), so the existing store stays valid
# and no re-ingestion is needed. Loaded once at import so the ~420MB model is not
# rebuilt on every request. Weights come from the local HuggingFace cache (baked
# into the image at build time; see Dockerfile).
_EMBED_MODEL_NAME = "all-mpnet-base-v2"
_embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
_groq = Groq(api_key=settings.groq_api_key)

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


class UpstreamUnavailableError(Exception):
    """Raised when the Groq chat model stays unreachable after all retries."""


def embed_text(text: str) -> list[float]:
    """Embed a single string locally with all-mpnet-base-v2 (768 dims)."""
    return _embed_model.encode(text).tolist()


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def generate(system_instruction: str, user_prompt: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = _groq.chat.completions.create(
                model=settings.groq_chat_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_BACKOFF_SECONDS[attempt])
    raise UpstreamUnavailableError(
        "Groq chat completion failed after all retries"
    ) from last_exc
