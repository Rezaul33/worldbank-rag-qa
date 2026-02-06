"""
Minimal Streamlit test to isolate the issue.
"""

import streamlit as st
import sys
sys.path.append('..')

from retriever.rag_retriever import RAGRetriever

def minimal_streamlit_test():
    """Minimal Streamlit test."""
    st.title("🧪 Minimal Streamlit Test")
    
    # Initialize RAG system
    if 'rag' not in st.session_state:
        with st.spinner("Initializing RAG..."):
            st.session_state.rag = RAGRetriever(ollama_model="llama2:latest")
    
    # Test query
    query = st.text_input("Enter query:", key="query")
    
    if st.button("Test Query", type="primary"):
        if query and st.session_state.get('rag'):
            with st.spinner("Processing..."):
                try:
                    result = st.session_state.rag.answer_query(query, top_k=3)
                    
                    st.success("✅ Query processed!")
                    st.write(f"**Query:** {query}")
                    st.write(f"**Answer:** {result.get('answer', 'No answer')}")
                    st.write(f"**Sources:** {len(result.get('sources', []))}")
                    
                    # Show similarity scores
                    sources = result.get('sources', [])
                    if sources:
                        for i, source in enumerate(sources):
                            st.write(f"Source {i+1}: {source.get('similarity', 0):.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    minimal_streamlit_test()
