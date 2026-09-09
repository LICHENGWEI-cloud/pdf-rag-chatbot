# src/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Text Splitting
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Paths
    VECTOR_DB_DIR: str = "./data/chroma_db"

    # Model Provider Selection ("openai" or "ollama")
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "openai")

    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    # Ollama Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_LLM_MODEL: str = "llama3.2"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"