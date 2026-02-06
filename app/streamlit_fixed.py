"""
FIXED Streamlit app - addressing session state issues.
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
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

def initialize_rag_system():
    """Initialize RAG system with proper error handling."""
    if st.session_state.rag_system is None:
        with st.spinner("🔄 Initializing RAG system..."):
            try:
                st.session_state.rag_system = RAGRetriever(ollama_model="llama2:latest")
                st.success("✅ RAG system initialized successfully!")
                st.session_state.initialization_time = time.time()
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {e}")
                st.session_state.rag_system = None
                return False
    else:
        # Check if system was initialized more than 30 seconds ago
        if time.time() - st.session_state.get('initialization_time', 0) > 30:
            st.warning("⚠️ RAG system may need reinitialization")
            try:
                st.session_state.rag_system = RAGRetriever(ollama_model="llama2:latest")
                st.success("✅ RAG system reinitialized!")
                st.session_state.initialization_time = time.time()
                return True
            except Exception as e:
                st.error(f"❌ Reinitialization failed: {e}")
                return False
    
    return st.session_state.rag_system is not None

def main():
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">🏦 World Bank RAG System</div>', unsafe_allow_html=True)
    
    # Initialize RAG system
    if not initialize_rag_system():
        st.error("❌ RAG system is not available. Please refresh the page.")
        st.stop()
        return
    
    # Query input and search
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question about World Bank development reports:",
            placeholder="e.g., What are the main challenges in global development?",
            key="query_input"
        )
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query and query.strip() and st.session_state.rag_system:
                handle_query(query.strip())
    
    with col2:
        st.write("")  # Spacer
        
        # Quick test buttons
        st.markdown("**🧪 Quick Tests:**")
        
        test_queries = [
            "What are main challenges in global development?",
            "How does climate change affect developing countries?",
            "What are the recommendations for economic growth?"
        ]
        
        for i, test_query in enumerate(test_queries, 1):
            if st.button(f"Test {i}", key=f"test_{i}", use_container_width=True):
                handle_query(test_query)

def handle_query(query):
    """Handle user query with proper error handling."""
    if not st.session_state.rag_system:
        st.error("❌ RAG system not initialized")
        return
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "timestamp": time.time()
    })
    
    # Process query
    with st.spinner("🔍 Searching documents and generating answer..."):
        try:
            result = st.session_state.rag_system.answer_query(query, top_k=5)
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", ""),
                "sources": result.get("sources", []),
                "timestamp": time.time(),
                "retrieval_time": result.get("retrieval_time", 0)
            })
            
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
            return
    
    # Display results
    display_results(query, result)

def display_results(query, result):
    """Display query results with proper formatting."""
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
            with st.expander(f"📄 Source {i}: {source['filename']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📄 Document:** {source['filename']}")
                    st.markdown(f"**📖 Page:** {source['page']}")
                    st.markdown(f"**📊 Similarity:** {source['similarity']:.4f}")
                
                with col2:
                    if st.button("📋 Copy", key=f"copy_{i}", use_container_width=True):
                        st.write("✅ Copied to clipboard!")
        
        # Performance metrics
        st.markdown("### ⚡ Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🕐 Retrieval Time", f"{result.get('retrieval_time', 0):.2f}s")
        with col2:
            st.metric("📄 Documents Found", len(sources))
        with col3:
            if sources:
                avg_similarity = sum(s['similarity'] for s in sources) / len(sources)
                st.metric("📊 Avg Similarity", f"{avg_similarity:.3f}")
            else:
                st.metric("📊 Avg Similarity", "0.000")
    
    # Chat history
    if st.session_state.messages:
        st.markdown("### 💬 Chat History")
        
        for i, message in enumerate(reversed(st.session_state.messages[-5:]), 1):
            with st.expander(f"💬 Message {len(st.session_state.messages) - i}"):
                if message["role"] == "user":
                    st.markdown(f"**🧑 You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message['content'][:300]}...")
                    
                    # Show sources for assistant messages
                    if "sources" in message and message["sources"]:
                        st.markdown("**📚 Sources:**")
                        for j, source in enumerate(message["sources"], 1):
                            st.markdown(f"• {source['filename']} (p.{source['page']}) - {source['similarity']:.3f}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
