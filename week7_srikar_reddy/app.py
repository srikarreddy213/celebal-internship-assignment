import streamlit as st
import os
import tempfile
import glob
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any

# ==========================================
# CONSTANTS & DEFAULT CONFIGURATION
# ==========================================
# Try to get API key from Streamlit secrets first, then fall back to environment variables
def get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

DEFAULT_API_KEY = get_secret("GOOGLE_API_KEY", get_secret("GEMINI_API_KEY", ""))

# Page configurations
st.set_page_config(
    page_title="RAG Intellect - Document QA Workspace",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None

# ==========================================
# PREMIUM CSS STYLING
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Font configurations */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }

    /* Core header layout */
    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 0.5rem;
    }
    .main-title {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Glassmorphic elements */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        transition: all 0.3s ease;
        margin-bottom: 1.25rem;
    }
    
    /* Support light theme specifically */
    [data-theme="light"] .glass-card {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.03);
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.12);
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Chat bubble enhancements */
    .chat-source-container {
        margin-top: 10px;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        padding-top: 8px;
    }

    /* KPI metric highlights */
    .metric-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.04) 0%, rgba(124, 58, 237, 0.04) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366f1;
        margin: 0;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Custom status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .status-inactive {
        background-color: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CUSTOM LOCAL LLM CLASS (T5-Base Fallback)
# ==========================================
class FlanT5LLM(LLM):
    tokenizer: Any = None
    model: Any = None
    max_new_tokens: int = 256

    @property
    def _llm_type(self) -> str:
        return "flan-t5"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==========================================
# RESOURCE CACHING
# ==========================================
@st.cache_resource(show_spinner="⏳ Initializing Embedding Engine (SentenceTransformers)...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="⏳ Loading Local LLM (google/flan-t5-base) - This may take a moment...")
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return FlanT5LLM(tokenizer=tokenizer, model=model)

@st.cache_resource(show_spinner="⚙️ Analyzing PDF and building FAISS Vector Index...")
def get_vector_store(pdf_path, chunk_size, chunk_overlap):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)

    # Initialize Embeddings
    embeddings = load_embedding_model()

    # Build Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, chunks, len(pages)

# ==========================================
# SIDEBAR / CONFIGURATION PANEL
# ==========================================
st.sidebar.markdown(
    "<h3 style='margin-bottom:0;'>⚙️ System Controls</h3>",
    unsafe_allow_html=True
)

# Backend Selection
llm_backend = st.sidebar.selectbox(
    "LLM Engine",
    ["Google Gemini (Pre-configured)", "Google Gemini (Custom Key)", "Local Model (flan-t5-base)"],
    index=0,
    help="Select the AI Backend to generate answers."
)

# API Key determination
google_api_key = ""
api_status_html = ""

if llm_backend == "Google Gemini (Pre-configured)":
    google_api_key = DEFAULT_API_KEY
    if google_api_key.strip():
        api_status_html = "<span class='status-badge status-active'>● Gemini Key Loaded</span>"
    else:
        api_status_html = "<span class='status-badge status-inactive'>● Key Missing</span>"
elif llm_backend == "Google Gemini (Custom Key)":
    google_api_key = st.sidebar.text_input(
        "Enter API Key",
        type="password",
        placeholder="AIzaSy...",
        help="Input your Gemini API key from Google AI Studio"
    )
    if google_api_key.strip():
        api_status_html = "<span class='status-badge status-active'>● Custom Key Loaded</span>"
    else:
        api_status_html = "<span class='status-badge status-inactive'>● Key Missing</span>"
else:
    api_status_html = "<span class='status-badge status-active'>● CPU Mode Active</span>"

# Display Connection Badge in Sidebar
st.sidebar.markdown(api_status_html, unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("<h3>📁 Ingestion Control</h3>", unsafe_allow_html=True)
doc_source = st.sidebar.radio(
    "Document Source",
    ["Use default 'knowledge.pdf'", "Upload custom PDF"],
    index=0
)

uploaded_file = None
if doc_source == "Upload custom PDF":
    uploaded_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])

# Settings expander
with st.sidebar.expander("🛠️ Splitting Parameters"):
    chunk_size = st.slider("Chunk Size", 200, 1500, 500, step=50, help="Max length of each extracted text block.")
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 100, step=10, help="Characters to duplicate between neighbor blocks.")

# Clear Chat Session State
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.toast("Chat history cleared!")

# ==========================================
# INGESTION PIPELINE EXECUTION
# ==========================================
pdf_path = None
if doc_source == "Use default 'knowledge.pdf'":
    if os.path.exists("knowledge.pdf"):
        pdf_path = "knowledge.pdf"
    else:
        # Search directory for any pdf
        local_pdfs = glob.glob("*.pdf")
        if local_pdfs:
            pdf_path = local_pdfs[0]
        else:
            st.sidebar.error("❌ Default PDF not found. Please upload a PDF using the upload menu.")
else:
    if uploaded_file is not None:
        # Save to temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = temp_path
        st.session_state.ingested_file = uploaded_file.name

# ==========================================
# MAIN APP INTERFACE
# ==========================================
st.markdown(
    "<div class='header-container'><div class='main-title'>📄 RAG Intellect Workspace</div></div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='sub-title'>Perform deep intelligent document analysis using Retrieval-Augmented Generation.</div>",
    unsafe_allow_html=True
)

if pdf_path:
    try:
        # 1. Build database or retrieve from cache
        vector_store, chunks, total_pages = get_vector_store(pdf_path, chunk_size, chunk_overlap)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # 2. Build tabs
        tab1, tab2, tab3 = st.tabs([
            "💬 Chat Workspace", 
            "📄 Document Chunks Explorer", 
            "📊 System Info & Analytics"
        ])

        # Setup LLM backend
        if "Gemini" in llm_backend:
            if not google_api_key.strip():
                st.warning("⚠️ Google API Key is missing. Provide an API key in the sidebar.")
                st.stop()
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                # set system environment key as well
                os.environ["GOOGLE_API_KEY"] = google_api_key
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=google_api_key,
                    temperature=0.3
                )
            except ImportError:
                st.error("❌ `langchain-google-genai` is not available. Please install it.")
                st.stop()
        else:
            llm = load_local_llm()

        # Build prompt templates
        RAG_PROMPT = (
            "You are a helpful and precise assistant. Use ONLY the context below to answer the question.\n"
            "If the answer is not in the context, say: 'I don't have enough information in this document.'\n\n"
            "Context:\n"
            "------------------------------------------------------------\n"
            "{context}\n"
            "------------------------------------------------------------\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        prompt_tmpl = PromptTemplate.from_template(RAG_PROMPT)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL chain assembly
        rag_chain = (
            RunnableLambda(lambda q: {
                "context": format_docs(retriever.invoke(q)),
                "question": q
            })
            | prompt_tmpl
            | llm
            | StrOutputParser()
        )

        # ------------------------------------------
        # TAB 1: CHAT WORKSPACE
        # ------------------------------------------
        with tab1:
            st.markdown("### 💬 Chat with your Document")
            st.caption("Ask specific questions and the agent will answer using context retrieved from the PDF.")

            # Display Chat History
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("sources"):
                        with st.expander("🔍 View Retrieved Sources"):
                            for idx, src in enumerate(message["sources"], 1):
                                st.markdown(
                                    f"**Source {idx} | Page {src['page']}**\n"
                                    f"```text\n{src['content']}\n```"
                                )

            # Chat Input
            if prompt := st.chat_input("Ask something about the document..."):
                # Render user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Render assistant message
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Analyzing document database..."):
                        start_time = time.time()
                        # Retrieve contexts
                        retrieved_docs = retriever.invoke(prompt)
                        # Invoke RAG chain
                        answer = rag_chain.invoke(prompt)
                        elapsed_time = time.time() - start_time
                        
                        # Prepare sources
                        sources_list = []
                        for doc in retrieved_docs:
                            sources_list.append({
                                "page": doc.metadata.get("page", 0) + 1,
                                "content": doc.page_content
                            })

                        # Stream / Display response
                        message_placeholder.markdown(answer)
                        
                        # Show sources in expander
                        if sources_list:
                            with st.expander("🔍 View Retrieved Sources"):
                                for idx, src in enumerate(sources_list, 1):
                                    st.markdown(
                                        f"**Source {idx} | Page {src['page']}**\n"
                                        f"```text\n{src['content']}\n```"
                                    )
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_list,
                    "time_taken": elapsed_time
                })
                # Rerun to update chat flow
                st.rerun()

        # ------------------------------------------
        # TAB 2: DOCUMENT CHUNKS EXPLORER
        # ------------------------------------------
        with tab2:
            st.markdown("### 📄 Vector Database Explorer")
            st.markdown("This panel lets you directly query the underlying vector database to inspect which text chunks are returned and see their relevance.")

            query_col, score_col = st.columns([3, 1])
            with query_col:
                db_query = st.text_input("Enter search phrase to test retrieval:", placeholder="Type a concept or keyword...")
            with score_col:
                k_val = st.slider("Number of Chunks (K)", 1, 10, 4)

            if db_query:
                # Perform similarity search with score
                results = vector_store.similarity_search_with_score(db_query, k=k_val)
                st.markdown(f"Found **{len(results)}** matches for query: *\"{db_query}\"*")
                
                for idx, (doc, score) in enumerate(results, 1):
                    # FAISS score is L2 distance, smaller is better (closer)
                    # Normalize L2 distance to percentage similarity metric just for visual convenience
                    similarity = max(0.0, min(100.0, 100.0 - (score * 50.0)))
                    
                    with st.container():
                        st.markdown(
                            f"<div class='glass-card'>"
                            f"<h4>Chunk {idx} | Page {doc.metadata.get('page', 0) + 1}</h4>"
                            f"<p><b>Vector Relevance Score:</b> {similarity:.2f}% (L2 Distance: {score:.4f})</p>"
                            f"<hr style='margin: 8px 0; border: none; border-top: 1px solid rgba(148, 163, 184, 0.15);'>"
                            f"<p style='font-size:0.95rem; font-style:italic;'>\"{doc.page_content}\"</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
            else:
                # Display generic statistics and first 3 chunks
                st.info("💡 Enter a search query above to inspect exact matches. Below are the first few database chunks:")
                for idx in range(min(3, len(chunks))):
                    c = chunks[idx]
                    st.markdown(
                        f"<div class='glass-card'>"
                        f"<h4>Chunk {idx + 1} (Page {c.metadata.get('page', 0) + 1})</h4>"
                        f"<p style='font-size:0.95rem;'>{c.page_content}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # ------------------------------------------
        # TAB 3: SYSTEM INFO & ANALYTICS
        # ------------------------------------------
        with tab3:
            st.markdown("### 📊 System Information & Diagnostics")
            
            # Key statistics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-val'>{os.path.basename(pdf_path)}</div><div class='metric-lbl'>File Name</div></div>",
                    unsafe_allow_html=True
                )
            with kpi2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-val'>{total_pages}</div><div class='metric-lbl'>Total Pages</div></div>",
                    unsafe_allow_html=True
                )
            with kpi3:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-val'>{len(chunks)}</div><div class='metric-lbl'>Text Chunks</div></div>",
                    unsafe_allow_html=True
                )
            with kpi4:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-val'>{llm_backend.split('(')[0]}</div><div class='metric-lbl'>Active Model</div></div>",
                    unsafe_allow_html=True
                )

            # Execution logic flow diagram
            st.markdown("### 🔄 Ingestion & Retrieval Workflow")
            st.markdown("""
            ```mermaid
            graph TD
                A[Ingested PDF: pdf_path] -->|PyPDFLoader| B[Pages Loaded]
                B -->|RecursiveCharacterTextSplitter| C[Text Chunks]
                C -->|sentence-transformers/all-MiniLM-L6-v2| D[Embeddings Creation]
                D -->|FAISS indexing| E[FAISS Vector Store]
                
                F[User Query] -->|Embedding Search| E
                E -->|Similarity Retrieval| G[Top-K Relevance Context]
                G -->|Construct Prompt Template| H[Formatted Prompt]
                H -->|Active LLM Backend| I[Final Text Response]
            ```
            """)

    except Exception as e:
        st.error(f"❌ Core processing error: {e}")
        st.exception(e)
else:
    st.info("💡 To begin, upload a custom PDF document or place a 'knowledge.pdf' file in the root directory.")
