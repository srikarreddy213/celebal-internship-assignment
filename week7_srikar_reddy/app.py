import streamlit as st
import os
import tempfile
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
st.set_page_config(page_title="RAG Document QA System", page_icon="📄", layout="wide")

st.markdown("""
<style>
.main-header {{font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }}
.sub-header {{font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem; }}
.status-card {{background-color: #F3F4F6; border-radius: 8px; padding: 1rem; border-left: 4px solid #3B82F6; margin-bottom: 1.5rem; }}
.answer-card {{background-color: #EFF6FF; border-radius: 8px; padding: 1rem; border-left: 5px solid #2563EB; margin-top: 1rem; }}
.source-card {{background-color: #FAFAFA; border-radius: 6px; padding: 1rem; border: 1px solid #E5E7EB; margin-bottom: 0.5rem; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CUSTOM LOCAL LLM CLASS
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
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return FlanT5LLM(tokenizer=tokenizer, model=model)

# ==========================================
# APP UI & LOGIC
# ==========================================
st.markdown("<div class='main-header'>📄 Document QA System</div>", unsafe_allow_html=True)

llm_backend = st.sidebar.selectbox("Choose LLM Backend", ["Local (flan-t5-base)", "Google Gemini"])
google_api_key = st.sidebar.text_input("Google API Key", type="password")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(loader.load())
    vector_store = FAISS.from_documents(chunks, load_embedding_model())
    retriever = vector_store.as_retriever()

    if llm_backend == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)
    else:
        llm = load_local_llm()

    user_query = st.text_input("Ask a question:")
    if user_query:
        context = "\n".join([d.page_content for d in retriever.invoke(user_query)])
        answer = llm.invoke(f"Context: {context}\nQuestion: {user_query}")
        st.markdown(f"<div class='answer-card'><b>Answer:</b> {answer}</div>", unsafe_allow_html=True)
