import time

from google import genai
from groq import APIConnectionError, APIStatusError, Groq

from app.config import settings

_gemini = genai.Client(api_key=settings.gemini_api_key)
_groq = Groq(api_key=settings.groq_api_key)

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


class UpstreamUnavailableError(Exception):
    """Raised when the Groq chat model stays unreachable after all retries."""


def embed_text(text: str) -> list[float]:
    resp = _gemini.models.embed_content(
        model=settings.gemini_embed_model,
        contents=text,
    )
    return list(resp.embeddings[0].values)


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
