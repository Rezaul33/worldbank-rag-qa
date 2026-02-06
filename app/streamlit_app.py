"""
Streamlit web interface for World Bank RAG system.
Provides interactive Q&A interface with document retrieval and AI-powered answers.
"""

import streamlit as st
import sys
import os
import time
import json
from typing import List, Dict, Any
import pandas as pd

# Add parent directory to path for imports
sys.path.append('..')

from retriever.rag_retriever import RAGRetriever
from generator.answer_generator import AnswerGenerator
from evaluation.metrics import RAGEvaluator

# Page configuration
st.set_page_config(
    page_title="World Bank RAG System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
    }
    .source-item {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #28a745;
    }
    .answer-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = RAGEvaluator()

def initialize_rag_system():
    """Initialize RAG system with selected model."""
    if st.session_state.rag_system is None:
        with st.spinner("Initializing RAG system..."):
            try:
                st.session_state.rag_system = RAGRetriever(
                    ollama_model=st.session_state.get('selected_model', 'llama2:latest')
                )
                st.session_state.answer_generator = AnswerGenerator()
                st.session_state.rag_ready = True
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {e}")
                st.session_state.rag_ready = False
                return False
    return st.session_state.get('rag_ready', False)

def check_ollama_status():
    """Check Ollama connection status."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, [model["name"] for model in models]
        return False, []
    except:
        return False, []

def display_metrics():
    """Display system metrics in sidebar."""
    if st.session_state.rag_system:
        # Get collection stats
        stats = st.session_state.rag_system.vector_store.get_collection_stats()
        
        st.sidebar.markdown("### 📊 System Metrics")
        st.sidebar.markdown(f"**Documents in DB:** {stats.get('total_documents', 0):,}")
        st.sidebar.markdown(f"**Collection:** {stats.get('collection_name', 'N/A')}")
        
        # Model info
        st.sidebar.markdown("### 🤖 Model Info")
        st.sidebar.markdown(f"**Model:** {st.session_state.get('selected_model', 'llama2:latest')}")
        
        # Chat history
        st.sidebar.markdown("### 💬 Chat History")
        st.sidebar.markdown(f"**Queries:** {len(st.session_state.chat_history)}")

def display_chat_interface():
    """Display main chat interface."""
    st.markdown('<div class="main-header">🏦 World Bank RAG System</div>', unsafe_allow_html=True)
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question about World Bank development reports:",
            placeholder="e.g., What are the main challenges in global development?",
            key="query_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Align button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query and query.strip():
                handle_query(query.strip())

def handle_query(query: str):
    """Handle user query and display results."""
    # Initialize RAG system if needed
    if not initialize_rag_system():
        st.error("❌ Failed to initialize RAG system. Please check Ollama connection.")
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
            result = st.session_state.rag_system.answer_query(query, top_k=5)
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
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
    display_query_result(query, result)

def display_query_result(query: str, result: Dict[str, Any]):
    """Display query results with formatting."""
    # Answer section
    st.markdown("### 📝 Answer")
    
    answer = result.get("answer", "")
    if answer:
        st.markdown(f'<div class="answer-container">{answer}</div>', unsafe_allow_html=True)
        
        # Quality assessment
        if 'answer_generator' in st.session_state and st.session_state.answer_generator:
            quality = st.session_state.answer_generator.assess_answer_quality(
                answer, query, result.get("sources", [])
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Length", f"{quality['length_score']:.2f}")
            with col2:
                st.metric("Relevance", f"{quality['relevance_score']:.2f}")
            with col3:
                st.metric("Citations", f"{quality['citation_score']:.2f}")
            with col4:
                st.metric("Overall", f"{quality['overall_score']:.2f}")
    else:
        st.error("❌ No answer generated. Please try again.")
    
    # Sources section
    sources = result.get("sources", [])
    if sources:
        st.markdown("### 📚 Sources")
        
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source['filename']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**📄 Document:** {source['filename']}")
                    st.markdown(f"**📖 Page:** {source['page']}")
                    st.markdown(f"**🔢 Chunk:** {source.get('chunk_index', 'N/A')}")
                    st.markdown(f"**📊 Similarity:** {source['similarity']:.4f}")
                
                with col2:
                    if st.button("📋 Copy", key=f"copy_source_{i}"):
                        st.write(f"Copied source {i} to clipboard")
                        # In a real app, you'd use clipboard functionality here
        
        # Performance metrics
        st.markdown("### ⚡ Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retrieval Time", f"{result.get('retrieval_time', 0):.2f}s")
        with col2:
            st.metric("Documents Found", len(sources))
    
    # Error handling
    if result.get("error"):
        st.error(f"❌ Error: {result['error']}")

def display_sidebar():
    """Display sidebar with settings and info."""
    st.sidebar.markdown("## ⚙️ Settings")
    
    # Model selection
    ollama_status, available_models = check_ollama_status()
    
    if ollama_status:
        selected_model = st.sidebar.selectbox(
            "🤖 Ollama Model",
            options=available_models,
            index=0,
            key="selected_model"
        )
        
        # Refresh models button
        if st.sidebar.button("🔄 Refresh Models"):
            st.rerun()
    else:
        st.sidebar.error("❌ Ollama not connected")
        st.sidebar.markdown("**Troubleshooting:**")
        st.sidebar.markdown("1. Make sure Ollama is running: `ollama serve`")
        st.sidebar.markdown("2. Check if port 11434 is accessible")
        return
    
    # Retrieval settings
    st.sidebar.markdown("### 🔍 Retrieval Settings")
    top_k = st.sidebar.slider("Number of documents to retrieve", 1, 10, 5, key="top_k")
    similarity_threshold = st.sidebar.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.05, key="similarity_threshold")
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        include_sources = st.checkbox("Include sources in answer", True, key="include_sources")
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.1, 0.05, key="temperature")
        max_tokens = st.slider("Max response tokens", 100, 2000, 1000, 100, key="max_tokens")
    
    # Chat management
    st.sidebar.markdown("### 💬 Chat Management")
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.sidebar.button("💾 Export Chat"):
        export_chat_history()

def export_chat_history():
    """Export chat history to JSON."""
    if st.session_state.chat_history:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"chat_history_{timestamp}.json"
        
        # Create download button
        st.download_button(
            label="📥 Download Chat History",
            data=json.dumps(st.session_state.chat_history, indent=2),
            file_name=filename,
            mime="application/json"
        )
    else:
        st.sidebar.warning("No chat history to export")

def display_analytics():
    """Display analytics dashboard."""
    if not st.session_state.chat_history:
        st.info("📊 No chat history available for analytics.")
        return
    
    st.markdown("## 📈 Analytics Dashboard")
    
    # Prepare data
    df = pd.DataFrame(st.session_state.chat_history)
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", len(df))
    with col2:
        if 'retrieval_time' in df.columns:
            avg_time = df['retrieval_time'].mean()
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
    with col3:
        successful_queries = len(df[df['type'] == 'assistant'])
        st.metric("Successful Responses", successful_queries)
    with col4:
        if successful_queries > 0:
            success_rate = successful_queries / len(df) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Response Time Distribution")
        if 'retrieval_time' in df.columns:
            st.bar_chart(df['retrieval_time'])
    
    with col2:
        st.markdown("### Query Timeline")
        if 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'], unit='s')
            hourly_counts = df.groupby(df['time'].dt.hour).size()
            st.line_chart(hourly_counts)

def main():
    """Main application function."""
    # Initialize RAG system
    initialize_rag_system()
    
    # Sidebar
    display_sidebar()
    display_metrics()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Analytics", "ℹ️ About"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_analytics()
    
    with tab3:
        st.markdown("## ℹ️ About World Bank RAG System")
        st.markdown("""
        ### 🎯 Purpose
        This system provides intelligent question-answering capabilities for World Bank Development Reports using:
        
        - **📚 Document Corpus**: World Bank Development Reports (2020-2025)
        - **🔍 Vector Search**: Chroma database with semantic similarity
        - **🤖 AI Generation**: Ollama integration for context-aware answers
        - **📊 Quality Metrics**: Automated evaluation and scoring
        
        ### 🚀 Features
        - **Real-time Search**: Retrieve relevant documents instantly
        - **Context-Aware Answers**: AI responses with source citations
        - **Performance Analytics**: Track system performance over time
        - **Interactive Interface**: Modern, user-friendly web interface
        
        ### 📈 System Stats
        - **Documents**: 3,010 chunks from 6 reports
        - **Embeddings**: 384-dimensional vectors
        - **Models**: Multiple Ollama models supported
        - **Response Time**: Typically 10-60 seconds
        
        ### 🔧 Technologies
        - **Backend**: Python, ChromaDB, SentenceTransformers
        - **Frontend**: Streamlit
        - **AI**: Ollama with Llama2/Llama3 models
        - **Vector DB**: Chroma with persistent storage
        """)

if __name__ == "__main__":
    main()
