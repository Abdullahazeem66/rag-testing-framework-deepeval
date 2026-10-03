from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)


class Settings(BaseSettings):
    """RAG app settings. Every field can be overridden by an env var (or .env) of the same name."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    chat_model: str = "gpt-6-luna"
    rewrite_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"

    docs_dir: Path = ROOT / "data" / "docs"
    index_dir: Path = ROOT / "data" / "index"
    collection_name: str = "orbitly_kb"

    chunk_max_chars: int = 1200
    top_k: int = 4
    # How many prior turns (user + assistant messages) are sent to the generator.
    history_window: int = 8


settings = Settings()
