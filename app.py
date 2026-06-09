"""
RAG Document Q&A Application
Powered by Groq API
"""

import os
import streamlit as st
import tempfile
from rag_module import RAGApplication
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration
GROK_API_KEY = os.getenv("GROK_API_KEY")  # Grok API key

# Page Setup
st.set_page_config(
    page_title="Document Q&A",
    page_icon="🔍",
    layout="wide"
)

# Modern CSS Styling
st.markdown("""
<style>
    /* Main container */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Chat container */
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .ai-msg {
        background: white;
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Status cards */
    .status-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .status-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 10px;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialize session state"""
    if "rag" not in st.session_state:
        st.session_state.rag = RAGApplication(GROK_API_KEY)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False


def render_header():
    """Render page header"""
    st.markdown("""
    <div class="header-container">
        <p class="header-title">🔍 Document Q&A</p>
        <p class="header-subtitle">Upload documents and ask questions powered by AI</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with controls"""
    with st.sidebar:
        st.markdown("### 📄 Upload Documents")
        
        # File upload
        files = st.file_uploader(
            "Choose PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if files:
            if st.button("🚀 Process Files", use_container_width=True):
                process_files(files)
        
        st.markdown("---")
        
        # Text input
        st.markdown("### ✏️ Or Paste Text")
        text = st.text_area("Enter text content", height=150, label_visibility="collapsed")
        
        if text:
            if st.button("📝 Process Text", use_container_width=True):
                process_text(text)
        
        st.markdown("---")
        
        # Status
        if st.session_state.docs_loaded:
            st.success("✅ Documents ready!")
        else:
            st.info("📤 Upload documents to start")
        
        # Clear button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.rag.clear_memory()
            st.rerun()


def process_files(files):
    """Process uploaded files"""
    with st.spinner("Processing documents..."):
        total = 0
        for file in files:
            ext = file.name.split(".")[-1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            try:
                if st.session_state.docs_loaded:
                    chunks = st.session_state.rag.add_documents(tmp_path, ext)
                else:
                    chunks = st.session_state.rag.process_uploaded_file(tmp_path, ext)
                    st.session_state.docs_loaded = True
                total += chunks
            finally:
                os.unlink(tmp_path)
        
        st.success(f"✅ Processed {len(files)} file(s) ({total} chunks)")
        st.rerun()


def process_text(text):
    """Process text input"""
    with st.spinner("Processing text..."):
        chunks = st.session_state.rag.process_text(text)
        st.session_state.docs_loaded = True
        st.success(f"✅ Processed text ({chunks} chunks)")
        st.rerun()


def render_chat():
    """Render chat interface"""
    # Display messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if st.session_state.docs_loaded:
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Get AI response
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag.ask_question(question)
                    answer = response["answer"]
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <p style="font-size: 3rem;">📄</p>
            <p style="font-size: 1.2rem;">Upload documents from the sidebar to start asking questions</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    init_session()
    render_header()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
