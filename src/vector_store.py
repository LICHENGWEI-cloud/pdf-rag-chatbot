# src/vector_store.py
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.config import Config

class VectorStoreManager:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.embeddings = self._get_embedding_model()

    def _get_embedding_model(self):
        if self.config.MODEL_PROVIDER == "ollama":
            return OllamaEmbeddings(
                model=self.config.OLLAMA_EMBED_MODEL,
                base_url=self.config.OLLAMA_BASE_URL
            )
        return OpenAIEmbeddings(
            model=self.config.OPENAI_EMBED_MODEL,
            api_key=self.config.OPENAI_API_KEY
        )

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Dynamically set collection_name based on provider to prevent dimension mismatch."""
        collection_name = f"pdf_rag_{self.config.MODEL_PROVIDER}"
        
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.config.VECTOR_DB_DIR,
            collection_name=collection_name
        )