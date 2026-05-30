from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"

    top_k: int = 5
    docs_sitemap: str = "https://supabase.com/docs/sitemap.xml"
    max_pages: int = 200

    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "supabase_docs"


settings = Settings()
