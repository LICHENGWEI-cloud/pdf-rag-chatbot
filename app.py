# app.py
import os
import tempfile
import streamlit as st

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine

st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📄 PDF Assistant (RAG Pipeline)")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox("Select Provider", ["openai", "ollama"], index=0)
    
    api_key = ""
    if provider == "openai":
        api_key = Config.OPENAI_API_KEY
        if not api_key:
            st.warning("Please enter your OpenAI API key to continue.")

    st.markdown("---")
    st.info("Upload a PDF to build the knowledge base.")

# Instantiate Config dynamically
config = Config(
    MODEL_PROVIDER=provider,
    OPENAI_API_KEY=api_key if provider == "openai" else ""
)

# Manage Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# PDF Upload Section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file and (provider == "ollama" or api_key):
    if st.button("Process Document"):
        with st.spinner("Processing PDF and generating embeddings..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Process PDF -> Chunks
                processor = DocumentProcessor(config)
                chunks = processor.process_pdf(tmp_path)
                st.write(f"Created {len(chunks)} text chunks.")

                # Vector Store -> Retriever
                vs_manager = VectorStoreManager(config)
                vector_store = vs_manager.create_vector_store(chunks)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                # Build RAG Chain
                rag_engine = RAGEngine(config)
                st.session_state.rag_chain = rag_engine.build_chain(retriever)

                st.success("PDF indexed successfully! You can now ask questions below.")
                os.remove(tmp_path)  # Cleanup temp file
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.rag_chain:
        st.error("Please upload and process a PDF document first.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant message
        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                response = st.session_state.rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})