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

# Custom CSS with escaped curly braces
st.markdown("""
<style>
.main-header {{font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }}
.sub-header {{font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem; }}
.status-card {{background-color: #F3F4F6; border-radius: 8px; padding: 1rem; border-left: 4px solid #3B82F6; margin-bottom: 1.5rem; }}
.answer-card {{background-color: #EFF6FF; border-radius: 8px; padding: 1.5rem; border-left: 5px solid #2563EB; margin-top: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.source-card {{background-color: #FAFAFA; border-radius: 6px; padding: 1rem; border: 1px solid #E5E7EB; margin-bottom: 0.75rem; }}
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
    def _llm_type(self) -> str: return "flan-t5"
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==========================================
# RESOURCE CACHING
# ==========================================
@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})

@st.cache_resource(show_spinner="Loading Local LLM...")
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return FlanT5LLM(tokenizer=tokenizer, model=model)

# ==========================================
# APP LAYOUT
# ==========================================
st.markdown("<div class='main-header'>📄 Document Question Answering System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Powered by Retrieval-Augmented Generation (RAG)</div>", unsafe_allow_html=True)

st.sidebar.header("🛠️ Configuration")
llm_backend = st.sidebar.selectbox("Choose LLM Backend", ["Local (flan-t5-base)", "Google Gemini (Requires API Key)"], index=0)
google_api_key = ""
if llm_backend == "Google Gemini (Requires API Key)":
    google_api_key = st.sidebar.text_input("Google API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📁 Document Ingestion")
doc_source = st.sidebar.radio("Select PDF Document Source", ["Use default 'knowledge.pdf'", "Upload custom PDF"])
uploaded_file = None
if doc_source == "Upload custom PDF":
    uploaded_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])

with st.sidebar.expander("⚙️ Advanced Splitter Settings"):
    chunk_size = st.slider("Chunk Size (characters)", 200, 1000, 500, step=50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 300, 100, step=10)

# ==========================================
# PIPELINE EXECUTION
# ==========================================
pdf_path = None
if doc_source == "Use default 'knowledge.pdf'":
    if os.path.exists("knowledge.pdf"): pdf_path = "knowledge.pdf"
    else: st.error("❌ Default PDF not found.")
elif uploaded_file is not None:
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
    pdf_path = temp_path

if pdf_path:
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(pages)
        st.markdown(f"<div class='status-card'>✅ <b>Document Loaded:</b> {os.path.basename(pdf_path)}</div>", unsafe_allow_html=True)
        
        embeddings = load_embedding_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        if llm_backend == "Google Gemini (Requires API Key)":
            if not google_api_key.strip():
                st.warning("⚠️ Enter Google API Key.")
                st.stop()
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
        else:
            llm = load_local_llm()

        RAG_PROMPT = "Use the context to answer: {context}\nQuestion: {question}"
        prompt = PromptTemplate.from_template(RAG_PROMPT)
        rag_chain = (RunnableLambda(lambda q: {"context": "\n".join([d.page_content for d in retriever.invoke(q)]), "question": q}) | prompt | llm | StrOutputParser())

        st.subheader("❓ Ask any question:")
        user_query = st.text_input("Enter your query here...")
        if user_query:
            answer = rag_chain.invoke(user_query)
            st.markdown(f"<div class='answer-card'><h3>💡 Generated Answer:</h3><p>{answer}</p></div>", unsafe_allow_html=True)
            with st.expander("📚 View Retrieved Source Context"):
                for doc in retriever.invoke(user_query):
                    st.markdown(f"<div class='source-card'><p>{doc.page_content}</p></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error: {e}")
