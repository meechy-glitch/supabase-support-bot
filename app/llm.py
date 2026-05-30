from google import genai
from google.genai import types

from app.config import settings

_client = genai.Client(api_key=settings.gemini_api_key)


def embed_text(text: str) -> list[float]:
    resp = _client.models.embed_content(
        model=settings.gemini_embed_model,
        contents=text,
    )
    return list(resp.embeddings[0].values)


def generate(system_instruction: str, user_prompt: str) -> str:
    resp = _client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    return resp.text or ""
