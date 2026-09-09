# src/rag_engine.py
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config

class RAGEngine:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.llm = self._get_llm()

    def _get_llm(self):
        if self.config.MODEL_PROVIDER == "ollama":
            return ChatOllama(
                model=self.config.OLLAMA_LLM_MODEL,
                base_url=self.config.OLLAMA_BASE_URL,
                temperature=0
            )
        return ChatOpenAI(
            model=self.config.OPENAI_LLM_MODEL,
            api_key=self.config.OPENAI_API_KEY,
            temperature=0
        )

    def build_chain(self, retriever):
        """Creates a retrieval QA chain."""
        system_prompt = (
            "You are a helpful assistant answering questions based on provided document context.\n"
            "Use only the provided context below to answer the user's question. "
            "If you do not know the answer based on the context, state that you don't know.\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)