# app.py
import os
import tempfile
import streamlit as st

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine

# ------------------------------------------------------------------------------
# 1. Page Configuration & Sidebar Setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="PDF RAG Assistant with Sources", layout="wide")
st.title("📄 PDF Assistant (RAG Pipeline with Citations)")

with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox("Select Provider", ["openai", "ollama"], index=0)
    
    api_key = ""
    if provider == "openai":
        api_key = Config.OPENAI_API_KEY
        if not api_key:
            st.warning("Please enter your OpenAI API key to continue.")

    st.markdown("---")
    st.info("Upload a PDF to build the knowledge base and perform grounded QA.")

# Instantiate dynamic configuration based on UI selection
config = Config(
    MODEL_PROVIDER=provider,
    OPENAI_API_KEY=api_key if provider == "openai" else ""
)

# ------------------------------------------------------------------------------
# 2. Session State Initialization
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ------------------------------------------------------------------------------
# 3. PDF Ingestion & Pipeline Assembly
# ------------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file and (provider == "ollama" or api_key):
    if st.button("Process Document"):
        with st.spinner("Processing PDF, generating embeddings, and indexing..."):
            try:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Step A: Parse and Chunk Document
                processor = DocumentProcessor(config)
                chunks = processor.process_pdf(tmp_path)
                st.write(f"Created **{len(chunks)}** text chunks.")

                # Step B: Index Chunks into Vector Store
                vs_manager = VectorStoreManager(config)
                vector_store = vs_manager.create_vector_store(chunks)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                # Step C: Construct RAG Chain
                rag_engine = RAGEngine(config)
                st.session_state.rag_chain = rag_engine.build_chain(retriever)

                st.success("PDF processed successfully! You can now ask questions with source grounding.")
                os.remove(tmp_path)  # Cleanup temp file
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# ------------------------------------------------------------------------------
# 4. Render Conversation History with Sources
# ------------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Render source citations for previous assistant messages
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Referenced Context & Page Numbers"):
                for idx, doc in enumerate(message["sources"], start=1):
                    # PyPDF stores 0-indexed page numbers in metadata['page']
                    page_num = doc.metadata.get("page", 0) + 1 
                    st.markdown(f"**Reference [{idx}] — Page {page_num}**")
                    st.info(f'"{doc.page_content.strip()}"')

# ------------------------------------------------------------------------------
# 5. User Input & Inference Execution
# ------------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.rag_chain:
        st.error("Please upload and process a PDF document first.")
    else:
        # A. Display User Question
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # B. Generate Assistant Answer & Extract Citations
        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                response = st.session_state.rag_chain.invoke({"input": prompt})
                
                answer = response["answer"]
                source_documents = response.get("context", [])  # Retrieved chunks

                # Render LLM Answer
                st.markdown(answer)

                # Render Citation Collapsible Box
                if source_documents:
                    with st.expander("🔍 View Referenced Context & Page Numbers"):
                        for idx, doc in enumerate(source_documents, start=1):
                            page_num = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**Reference [{idx}] — Page {page_num}**")
                            st.info(f'"{doc.page_content.strip()}"')

                # C. Save Output and Sources to Session State
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_documents
                })