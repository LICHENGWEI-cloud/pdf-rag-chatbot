# 📄 Modular RAG PDF Chatbot

A production-ready Retrieval-Augmented Generation (RAG) PDF assistant built with **Python**, **LangChain**, **ChromaDB**, **Streamlit**, and support for both **OpenAI API** and local **Ollama** models.

---

## 🗺️ Project Architecture & Roadmap

<Timeline>
  <TimelineEvent title="Phase 1: Environment & Core Setup" time="Completed">
    - Established modular repository layout.
    - Configured runtime parameter management via `src/config.py`.
    - Added provider selection for OpenAI and Ollama.
  </TimelineEvent>

  <TimelineEvent title="Phase 2: Ingestion & Vector Storage Pipeline" time="Completed">
    - Built PDF document loading and text chunking in `src/document_processor.py`.
    - Integrated ChromaDB and embeddings in `src/vector_store.py`.
    - Dynamic collection naming to prevent embedding dimension mismatch issues.
  </TimelineEvent>

  <TimelineEvent title="Phase 3: RAG Engine & Streamlit Interface" time="Completed">
    - Implemented LangChain LCEL retrieval chains in `src/rag_engine.py`.
    - Created Streamlit UI in `app.py` with multi-turn chat support.
  </TimelineEvent>

  <TimelineEvent title="Phase 4: Advanced Features" time="Planned">
    - Display source document page citations in chat responses.
    - Multi-PDF document collection querying.
    - Automated unit tests with `pytest`.
  </TimelineEvent>
</Timeline>

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone [https://github.com/YOUR_USERNAME/rag-pdf-chatbot.git](https://github.com/YOUR_USERNAME/rag-pdf-chatbot.git)
cd rag-pdf-chatbot
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### 2. Configure Environment Variables