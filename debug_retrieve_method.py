"""
Debug the retrieve_documents method specifically.
"""

import sys
sys.path.append('..')

from retriever.rag_retriever import RAGRetriever

def debug_retrieve_method():
    """Debug retrieve_documents method."""
    print("🔍 Debugging retrieve_documents Method")
    print("=" * 50)
    
    # Initialize RAG system
    print("Initializing RAG system...")
    retriever = RAGRetriever(ollama_model="llama2:latest")
    
    # Test query
    query = "What are main challenges in global development?"
    print(f"Testing query: '{query}'")
    
    try:
        # Test retrieve_documents directly
        print("\n1. Testing retrieve_documents method:")
        docs = retriever.retrieve_documents(query, top_k=5)
        
        print(f"   Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"   Doc {i+1}: Similarity {doc.get('similarity', 0):.4f}")
            print(f"           Content: {doc.get('document', '')[:100]}...")
        
        # Test vector store query directly
        print("\n2. Testing vector store query directly:")
        store_results = retriever.vector_store.query(
            query_text=query,
            embedding_model=retriever.embedding_generator.model,
            n_results=5
        )
        
        if "error" not in store_results:
            print(f"   Store returned {len(store_results['results'])} documents")
            for i, result in enumerate(store_results['results']):
                print(f"   Result {i+1}: Similarity {result['similarity']:.4f}")
        else:
            print(f"   Store error: {store_results['error']}")
        
        # Compare results
        print(f"\n3. Comparison:")
        print(f"   retrieve_documents returned: {len(docs)}")
        print(f"   vector_store.query returned: {len(store_results.get('results', []))}")
        
        if len(docs) != len(store_results.get('results', [])):
            print("   ❌ MISMATCH: Different number of results!")
        else:
            print("   ✅ Results match")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_retrieve_method()
