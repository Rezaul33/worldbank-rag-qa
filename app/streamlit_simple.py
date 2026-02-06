"""
Simplified Streamlit app for World Bank RAG system.
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
    layout="wide",
    initial_sidebar_state="expanded"
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
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag():
    """Initialize RAG system."""
    if st.session_state.rag is None:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag = RAGRetriever(ollama_model="llama2:latest")
        return True
    return st.session_state.rag is not None

def main():
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">🏦 World Bank RAG System</div>', unsafe_allow_html=True)
    
    # Initialize RAG system
    if not initialize_rag():
        st.error("❌ Failed to initialize RAG system")
        return
    
    # Sidebar - Settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model info
        st.markdown("**🤖 Model:** llama2:latest")
        
        # System status
        st.markdown("**✅ Status:** Ready")
        
        # Chat management
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Ask a Question")
        
        query = st.text_input(
            "Enter your question about World Bank development reports:",
            placeholder="e.g., What are the main challenges in global development?",
            key="query_input"
        )
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query and query.strip():
                handle_query(query.strip())
    
    with col2:
        st.markdown("### 📊 Quick Test")
        
        if st.button("🧪 Test Sample Query", use_container_width=True):
            handle_query("What are the main challenges in global development?")

def handle_query(query):
    """Handle user query."""
    if not initialize_rag():
        st.error("❌ RAG system not initialized")
        return
    
    # Add to chat history
    st.session_state.chat_history.append({
        "question": query,
        "timestamp": time.time(),
        "type": "user"
    })
    
    # Process query
    with st.spinner("🔍 Searching documents and generating answer..."):
        try:
            result = st.session_state.rag.answer_query(query, top_k=5)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return
    
    # Add result to chat history
    st.session_state.chat_history.append({
        "question": query,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "retrieval_time": result.get("retrieval_time", 0),
        "timestamp": time.time(),
        "type": "assistant"
    })
    
    # Display results
    display_results(query, result)

def display_results(query, result):
    """Display query results."""
    # Answer section
    st.markdown("### 📝 Answer")
    answer = result.get("answer", "")
    if answer:
        st.markdown(f'<div class="answer-container">{answer}</div>', unsafe_allow_html=True)
    else:
        st.error("❌ No answer generated")
    
    # Sources section
    sources = result.get("sources", [])
    if sources:
        st.markdown("### 📚 Sources")
        
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source['filename']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📄 Document:** {source['filename']}")
                    st.markdown(f"**📖 Page:** {source['page']}")
                    st.markdown(f"**📊 Similarity:** {source['similarity']:.4f}")
                
                with col2:
                    if st.button("📋 Copy", key=f"copy_{i}", use_container_width=True):
                        st.write("Copied to clipboard")
        
        # Performance metrics
        st.markdown("### ⚡ Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Retrieval Time", f"{result.get('retrieval_time', 0):.2f}s")
        with col2:
            st.metric("Documents Found", len(sources))
        with col3:
            avg_similarity = sum(s['similarity'] for s in sources) / len(sources) if sources else 0
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        
        for i, chat_item in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Q{i}: {chat_item['question'][:50]}..."):
                st.write(f"**Question:** {chat_item['question']}")
                if chat_item['type'] == 'assistant':
                    st.write(f"**Answer:** {chat_item['answer'][:200]}...")
                    st.write(f"**Time:** {chat_item.get('retrieval_time', 0):.2f}s")

if __name__ == "__main__":
    main()
