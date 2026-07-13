import streamlit as st
import os
import tempfile
import glob
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
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(page_title="RAG Document QA System", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for modern premium styling
# NOTICE: We have doubled the curly braces {{ }} and used the correct parameter 'unsafe_allow_html'
st.markdown("""
<style>
.main-header {{
    font-size: 2.5rem; 
    font-weight: 700; 
    color: #1E3A8A; 
    margin-bottom: 0.5rem; 
}}
.sub-header {{
    font-size: 1.2rem; 
    color: #4B5563; 
    margin-bottom: 2rem; 
}}
.status-card {{
    background-color: #F3F4F6; 
    border-radius: 8px; 
    padding: 1rem; 
    border-left: 4px solid #3B82F6; 
    margin-bottom: 1.5rem; 
}}
.answer-card {{
    background-color: #EFF6FF; 
    border-radius: 8px; 
    padding: 1.5rem; 
    border-left: 5px solid #2563EB; 
    margin-top: 1.5rem; 
    margin-bottom: 1.5rem; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
}}
.source-card {{
    background-color: #FAFAFA; 
    border-radius: 6px; 
    padding: 1rem; 
    border: 1px solid #E5E7EB; 
    margin-bottom: 0.75rem; 
}}
</style>
""", unsafe_allow_html=True)
# ==========================================
# CUSTOM LOCAL LLM CLASS (T5-Base)
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
@st.cache_resource(show_spinner="Loading Embedding Model (sentence-transformers)...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="Loading Local LLM (google/flan-t5-base)...")
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return FlanT5LLM(tokenizer=tokenizer, model=model)

# ==========================================
# APP LAYOUT
# ==========================================
st.markdown("<div class='main-header'>📄 Document Question Answering System</div>", unsafe_allowed_html=True)
st.markdown("<div class='sub-header'>Powered by Retrieval-Augmented Generation (RAG)</div>", unsafe_allowed_html=True)

# Sidebar configurations
st.sidebar.header("🛠️ Configuration")

# LLM Backend selection
llm_backend = st.sidebar.selectbox(
    "Choose LLM Backend",
    ["Local (flan-t5-base)", "Google Gemini (Requires API Key)"],
    index=0
)

# Gemini API Key Input
google_api_key = ""
if llm_backend == "Google Gemini (Requires API Key)":
    google_api_key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        placeholder="AIzaSy...",
        help="Input your Gemini API key from Google AI Studio"
    )

st.sidebar.markdown("---")
st.sidebar.header("📁 Document Ingestion")

# PDF selection mode
doc_source = st.sidebar.radio(
    "Select PDF Document Source",
    ["Use default 'knowledge.pdf'", "Upload custom PDF"]
)

uploaded_file = None
if doc_source == "Upload custom PDF":
    uploaded_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])

# Advanced chunk configurations
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Advanced Splitter Settings"):
    chunk_size = st.slider("Chunk Size (characters)", 200, 1000, 500, step=50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 300, 100, step=10)

# ==========================================
# PIPELINE EXECUTION
# ==========================================
# Step 1: Locate and resolve PDF path
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
            st.error("❌ Default PDF not found. Please upload a PDF using the sidebar option.")
else:
    if uploaded_file is not None:
        # Save uploaded PDF to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = temp_path

# Step 2: Initialize vector store and models if PDF is resolved
if pdf_path:
    try:
        # Load PDF & split
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)

        # Show status
        st.markdown(
            f"<div class='status-card'>✅ <b>Document Loaded:</b> {os.path.basename(pdf_path)} "
            f"({len(pages)} pages split into {len(chunks)} overlapping chunks)</div>",
            unsafe_allowed_html=True
        )

        # Load models
        embeddings = load_embedding_model()

        # Build Vector Store
        @st.cache_resource(show_spinner="Building Vector Store...")
        def get_vector_store(_chunks, _embeddings):
            return FAISS.from_documents(_chunks, _embeddings)

        vector_store = get_vector_store(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Load LLM
        if llm_backend == "Google Gemini (Requires API Key)":
            if not google_api_key.strip():
                st.warning("⚠️ Please enter a Google API Key in the sidebar to proceed with Gemini.")
                st.stop()
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=google_api_key,
                    temperature=0.3
                )
            except ImportError:
                st.error("❌ langchain-google-genai not installed. Please install it in the requirements.txt.")
                st.stop()
        else:
            llm = load_local_llm()

        # Build LCEL Chain
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
        prompt = PromptTemplate.from_template(RAG_PROMPT)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnableLambda(lambda q: {
                "context": format_docs(retriever.invoke(q)),
                "question": q
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        # ==========================================
        # INTERACTIVE Q&A PANEL
        # ==========================================
        st.subheader("❓ Ask any question about the document:")
        user_query = st.text_input(
            "Enter your query here...",
            placeholder="What is this document about? / Summarize key points..."
        )

        if user_query:
            with st.spinner("Retrieving context and generating answer..."):
                # 1. Retrieve chunks for rendering
                retrieved_docs = retriever.invoke(user_query)

                # 2. Invoke RAG chain
                answer = rag_chain.invoke(user_query)

                # Render generated answer
                st.markdown("<div class='answer-card'>", unsafe_allowed_html=True)
                st.markdown("<h3>💡 Generated Answer:</h3>", unsafe_allowed_html=True)
                st.markdown(f"<p>{answer}</p>", unsafe_allowed_html=True)
                st.markdown("</div>", unsafe_allowed_html=True)

                # Render retrieved sources in expander
                with st.expander("📚 View Retrieved Source Context Chunks"):
                    for idx, doc in enumerate(retrieved_docs, 1):
                        pg = doc.metadata.get("page", 0)
                        st.markdown(f"<div class='source-card'>", unsafe_allowed_html=True)
                        st.markdown(f"<b>Chunk {idx} | Page {pg + 1}</b>", unsafe_allowed_html=True)
                        st.markdown(f"<p style='font-size: 0.95rem;'>{doc.page_content}</p>", unsafe_allowed_html=True)
                        st.markdown(f"</div>", unsafe_allowed_html=True)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("💡 Please upload a custom PDF document or place a 'knowledge.pdf' file in the directory to begin.")
