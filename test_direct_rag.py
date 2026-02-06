"""
Direct RAG test bypassing any caching.
"""

import sys
sys.path.append('..')

from retriever.rag_retriever import RAGRetriever

def test_direct_rag():
    """Test RAG directly."""
    print("🧪 Direct RAG Test")
    print("=" * 40)
    
    # Initialize fresh RAG system
    print("Initializing RAG system...")
    retriever = RAGRetriever(ollama_model="llama2:latest")
    
    # Test query
    query = "What are main challenges in global development?"
    print(f"Testing query: '{query}'")
    
    try:
        result = retriever.answer_query(query, top_k=3)
        
        print(f"\nQuery result:")
        print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
        print(f"Sources found: {len(result.get('sources', []))}")
        print(f"Retrieval time: {result.get('retrieval_time', 0):.2f}s")
        
        if result.get('sources'):
            for i, source in enumerate(result['sources'][:2]):
                print(f"  Source {i+1}: {source['filename']} - Similarity {source.get('similarity', 0):.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_direct_rag()
