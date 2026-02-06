"""
Working Streamlit app for World Bank RAG system.
Clean, simple, and functional.
"""

import streamlit as st
import sys
import time

# Add parent directory to path
sys.path.append('..')

from retriever.rag_retriever import RAGRetriever

# Page configuration
st.set_page_config(
    page_title="World Bank RAG System",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.answer-container {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #e1e5e9;
    margin: 1rem 0;
}
.source-item {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 0.25rem;
    margin: 0.5rem 0;
    border-left: 3px solid #28a745;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

def main():
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">🏦 World Bank RAG System</div>', unsafe_allow_html=True)
    
    # Initialize RAG system
    if st.session_state.rag is None:
        with st.spinner("🔄 Initializing RAG system..."):
            try:
                st.session_state.rag = RAGRetriever(ollama_model="llama2:latest")
                st.success("✅ RAG system initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {e}")
                return
    
    # Query input
    query = st.text_input(
        "Ask a question about World Bank development reports:",
        placeholder="e.g., What are the main challenges in global development?",
        key="query"
    )
    
    # Search button
    if st.button("🔍 Search", type="primary") and query and query.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "timestamp": time.time()
        })
        
        # Process query
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                result = st.session_state.rag.answer_query(query, top_k=5)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "timestamp": time.time(),
                    "retrieval_time": result.get("retrieval_time", 0)
                })
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return
    
    # Display chat history
    if st.session_state.messages:
        st.markdown("### 💬 Chat History")
        
        for i, message in enumerate(reversed(st.session_state.messages)):
            if message["role"] == "user":
                st.markdown(f"**🧑 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {message['content'][:300]}...")
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    st.markdown("**📚 Sources:**")
                    for j, source in enumerate(message["sources"], 1):
                        st.markdown(f"{j}. {source['filename']} - Page {source['page']} (Similarity: {source['similarity']:.4f})")
                
                # Show metrics
                if "retrieval_time" in message:
                    st.markdown(f"**⏱️ Time:** {message['retrieval_time']:.2f}s")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
