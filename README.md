# 📄 Modular RAG PDF Assistant

A production-ready Retrieval-Augmented Generation (RAG) PDF chatbot built with **Python**, **LangChain**, **ChromaDB**, **Streamlit**, and support for both **OpenAI API** and local **Ollama** models.

---

## 🗺️ Learning & Development Roadmap

```
     ┌─────────────┐
     │  Upload PDF │
     └──────┬──────┘
            ▼
┌───────────────────────┐
│  Document Processor   │ ──► Load PDF & split into overlapping chunks
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│     Vector Store      │ ──► Compute embeddings & index in ChromaDB
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│      RAG Engine       │ ──► Retrieve relevant contexts & prompt LLM
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│     Streamlit UI      │ ──► Multi-turn conversational interface
└───────────────────────┘
```

### Milestone Progress

- [x] **Phase 1: Project Architecture Setup**
  - [x] Create virtual environment and manage dependencies in `requirements.txt`.
  - [x] Establish modular folder layout separating UI, config, vector store, document processor, and core RAG engine.
  - [x] Build centralized configuration (`src/config.py`) supporting dynamic provider toggling (OpenAI vs. Ollama).

- [x] **Phase 2: Ingestion & Vector Pipeline**
  - [x] Implement PDF text extraction and recursive character chunking in `src/document_processor.py`.
  - [x] Build ChromaDB vector storage manager in `src/vector_store.py`.
  - [x] Add dynamic provider-based collection naming (`pdf_rag_openai` vs `pdf_rag_ollama`) to eliminate embedding dimension mismatch errors (768 vs 1536).

- [x] **Phase 3: RAG Chain & Conversational UI**
  - [x] Construct LangChain Expression Language (LCEL) chains in `src/rag_engine.py`.
  - [x] Implement multi-turn Streamlit chat UI in `app.py` with session state persistence.
  - [x] Add sidebar configuration for API keys and model selection.

- [ ] **Phase 4: Production Enhancements (Upcoming)**
  - [ ] Add source attribution preview (page numbers and text snippets) in chat responses.
  - [ ] Support multi-document uploading and vector store resetting directly from the UI.
  - [ ] Add automated unit tests using `pytest`.

---

## 🛠️ Tech Stack

* **UI Framework:** Streamlit
* **Orchestration:** LangChain (`langchain-community`, `langchain-core`)
* **Vector Store:** ChromaDB
* **Document Processing:** PyPDF (`pypdf`)
* **Models Supported:**
  * **OpenAI:** `gpt-4o-mini` (LLM) + `text-embedding-3-small` (Embeddings)
  * **Ollama (Local):** `llama3.2` (LLM) + `nomic-embed-text` (Embeddings)

---

## 📁 Repository Structure

```text
rag-pdf-chatbot/
│
├── .gitignore               # Excludes virtualenv, API secrets, & vector DB storage
├── .env.example             # Example environment configuration
├── README.md                # Project documentation and roadmap
├── requirements.txt         # Project dependencies
├── app.py                   # Streamlit web application entrypoint
│
├── src/                     # Core application modules
│   ├── __init__.py
│   ├── config.py            # Central settings & configuration class
│   ├── document_processor.py# PDF loading & text splitting
│   ├── vector_store.py      # Embeddings & ChromaDB management
│   └── rag_engine.py        # LangChain LCEL pipeline assembly
│
└── data/                    # Local storage directory
    ├── chroma_db/           # Persistent vector database storage (git-ignored)
    └── uploads/             # Temporary staging folder for uploaded files
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* **Python 3.10+**
* *(Optional)* **Ollama** installed locally if running without an OpenAI API Key ([Download Ollama](https://ollama.com)).

### 2. Installation
Clone the repository and set up your virtual environment:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-pdf-chatbot.git
cd rag-pdf-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Add your credentials inside `.env`:
```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

*If using Ollama locally, pull the required models first:*
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Run the Application
Launch the Streamlit web server:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Verification & Testing

Verify that individual components work independently before running the full UI:

```bash
# 1. Test package imports
python -c "import streamlit, langchain, chromadb, pypdf; print('Imports Successful!')"

# 2. Test Configuration Loading
python -c "from src.config import Config; cfg = Config(); print(f'Provider: {cfg.MODEL_PROVIDER}')"

# 3. Test Vector Store Manager Setup
python -c "from src.vector_store import VectorStoreManager; from src.config import Config; vs = VectorStoreManager(Config(OPENAI_API_KEY='dummy')); print('VectorStore Ready')"
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.