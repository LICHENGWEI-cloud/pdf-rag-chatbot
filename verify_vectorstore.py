# verify_vectorstore.py
from langchain_core.documents import Document
from src.vector_store import VectorStoreManager
from src.config import Config

# Mock config or provide API Key
cfg = Config(MODEL_PROVIDER="openai", OPENAI_API_KEY="sk-fake-key-for-structure-test")
try:
    vs_manager = VectorStoreManager(cfg)
    print("VectorStoreManager initialized with model provider:", cfg.MODEL_PROVIDER)
except Exception as e:
    print("Initialization error:", e)