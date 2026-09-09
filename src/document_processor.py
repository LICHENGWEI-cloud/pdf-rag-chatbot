# src/document_processor.py
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config

class DocumentProcessor:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """Loads a PDF and returns split text chunks."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if not documents:
            raise ValueError("The uploaded PDF is empty or could not be read.")
        return self.text_splitter.split_documents(documents)