from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str
    groq_chat_model: str = "qwen/qwen3.6-27b"

    top_k: int = 5
    docs_sitemap: str = "https://supabase.com/docs/sitemap.xml"
    # Global ceiling on pages visited by the recursive crawler.
    max_pages: int = 600
    # Recursive crawl depth: 0 = sitemap pages, 1 = their children, 2 = grandchildren.
    depth: int = 2

    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "supabase_docs_mpnet"


settings = Settings()
