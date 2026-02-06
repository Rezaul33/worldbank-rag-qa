"""
Debug Streamlit RAG initialization issues.
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append('..')

def debug_rag_init():
    """Debug RAG initialization step by step."""
    st.title("🔍 RAG Initialization Debug")
    
    st.write("### Step 1: Import Test")
    try:
        from retriever.rag_retriever import RAGRetriever
        st.success("✅ RAGRetriever import successful")
    except Exception as e:
        st.error(f"❌ RAGRetriever import failed: {e}")
        return
    
    st.write("### Step 2: Ollama Connection Test")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            st.success("✅ Ollama API accessible from Streamlit")
            models = response.json().get("models", [])
            st.write(f"Models: {[m['name'] for m in models]}")
        else:
            st.error(f"❌ Ollama API error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Ollama connection error: {e}")
        return
    
    st.write("### Step 3: RAG System Initialization")
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    if st.button("🚀 Initialize RAG System", type="primary"):
        with st.spinner("Initializing RAG system..."):
            try:
                st.session_state.retriever = RAGRetriever(ollama_model="llama2:latest")
                st.success("✅ RAG system initialized successfully!")
                st.session_state.rag_ready = True
            except Exception as e:
                st.error(f"❌ RAG initialization failed: {e}")
                st.code(str(e))
    
    st.write("### Step 4: Test Query")
    if st.session_state.get('rag_ready', False):
        if st.button("🧪 Test Query", type="secondary"):
            with st.spinner("Testing query..."):
                try:
                    result = st.session_state.retriever.answer_query("hello", top_k=3)
                    st.success("✅ Query successful!")
                    st.write(f"Answer: {result.get('answer', 'No answer')}")
                except Exception as e:
                    st.error(f"❌ Query failed: {e}")
    else:
        st.warning("⚠️ Please initialize RAG system first")

if __name__ == "__main__":
    debug_rag_init()
