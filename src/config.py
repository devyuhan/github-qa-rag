"""Settings and environment variable loading."""

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Settings:
    github_token: str = field(default_factory=lambda: os.environ["GITHUB_TOKEN"])
    openai_api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ["ANTHROPIC_API_KEY"]
    )
    pinecone_api_key: str = field(
        default_factory=lambda: os.environ["PINECONE_API_KEY"]
    )
    pinecone_index_name: str = field(
        default_factory=lambda: os.environ.get("PINECONE_INDEX_NAME", "github-qa")
    )

    # Defaults
    embed_model_name: str = "text-embedding-3-small"
    embed_dimensions: int = 1536
    llm_model_name: str = "claude-sonnet-4-20250514"
    similarity_top_k: int = 5
    default_extensions: tuple[str, ...] = (
        ".py", ".js", ".ts", ".md", ".rst", ".txt", ".yaml", ".json",
    )
