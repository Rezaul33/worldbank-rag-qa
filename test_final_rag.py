"""
Final RAG test - exact same as Streamlit but standalone.
"""

import sys
sys.path.append('..')

from retriever.rag_retriever import RAGRetriever

def main():
    """Final RAG test."""
    print("🚀 Final RAG Test")
    print("=" * 40)
    
    # Initialize RAG system (same as Streamlit)
    print("Initializing RAG system...")
    retriever = RAGRetriever(ollama_model="llama2:latest")
    
    # Test query
    query = "What are main challenges in global development?"
    print(f"Testing query: '{query}'")
    
    try:
        # Call answer_query directly (same as Streamlit)
        result = retriever.answer_query(query, top_k=5)
        
        print(f"\n✅ SUCCESS!")
        print(f"Query: {query}")
        print(f"Answer: {result.get('answer', '')[:200]}...")
        print(f"Sources: {len(result.get('sources', []))}")
        
        if result.get('sources'):
            for i, source in enumerate(result['sources'][:3]):
                print(f"  Source {i+1}: {source['filename']} - Similarity {source.get('similarity', 0):.4f}")
        
        print(f"Retrieval time: {result.get('retrieval_time', 0):.2f}s")
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
        else:
            print("🎉 RAG system working perfectly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
