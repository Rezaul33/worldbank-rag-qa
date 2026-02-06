"""
Debug document filtering logic in RAG retriever.
"""

import sys
sys.path.append('..')

from retriever.rag_retriever import RAGRetriever

def debug_filtering():
    """Debug filtering logic in RAG retriever."""
    print("🔍 Debugging Document Filtering")
    print("=" * 50)
    
    # Initialize RAG system
    retriever = RAGRetriever(ollama_model="llama2:latest")
    
    # Test query
    query = "What are main challenges in global development?"
    print(f"Testing query: '{query}'")
    
    try:
        # Step 1: Test document retrieval
        print("\n1. Testing document retrieval:")
        retrieved_docs = retriever.retrieve_documents(query, top_k=5)
        
        print(f"   Retrieved {len(retrieved_docs)} documents")
        for i, doc in enumerate(retrieved_docs):
            print(f"   Doc {i+1}: Similarity {doc.get('similarity', 0):.4f}")
            print(f"           Content: {doc.get('document', '')[:100]}...")
        
        # Step 2: Test context generation
        print("\n2. Testing context generation:")
        context = retriever.generate_context(retrieved_docs)
        print(f"   Context length: {len(context)} chars")
        print(f"   Context preview: {context[:200]}...")
        
        # Step 3: Check if context indicates no documents
        if "No relevant documents found" in context:
            print("   ❌ CONTEXT SAYS NO RELEVANT DOCS!")
        else:
            print("   ✅ Context contains documents")
        
        # Step 4: Test full RAG pipeline
        print("\n3. Testing full RAG pipeline:")
        result = retriever.answer_query(query, top_k=5)
        
        print(f"   Final result: {result.get('error', 'Success')}")
        print(f"   Sources returned: {len(result.get('sources', []))}")
        print(f"   Answer length: {len(result.get('answer', ''))}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_filtering()
