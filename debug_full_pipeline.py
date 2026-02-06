"""
Debug full RAG pipeline to find filtering issue.
"""

import sys
sys.path.append('..')

from vector_store.chroma_db import ChromaVectorStore
from embeddings.embedding_generator import EmbeddingGenerator

def debug_full_pipeline():
    """Debug complete RAG pipeline step by step."""
    print("🔍 Debugging Full RAG Pipeline")
    print("=" * 50)
    
    # Initialize components
    vector_store = ChromaVectorStore()
    generator = EmbeddingGenerator()
    
    # Test query
    query = "What are main challenges in global development?"
    print(f"Testing query: '{query}'")
    
    try:
        # Step 1: Get query embedding
        query_embedding = generator.model.encode(query)
        print(f"\n1. Query embedding generated: {len(query_embedding)} dimensions")
        
        # Step 2: Raw Chroma query
        print("\n2. Raw Chroma query:")
        raw_results = vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"   Raw results count: {len(raw_results['ids'][0])}")
        print(f"   Raw distances: {raw_results['distances'][0]}")
        
        # Step 3: Manual similarity calculation
        print("\n3. Manual similarity calculation:")
        for i, distance in enumerate(raw_results['distances'][0]):
            similarity = 1 - (distance / 2)
            print(f"   Distance {i+1}: {distance:.4f} -> Similarity: {similarity:.4f}")
        
        # Step 4: Check vector store query method
        print("\n4. Vector store query method:")
        store_results = vector_store.query(
            query_text=query,
            embedding_model=generator.model,
            n_results=5
        )
        
        if "error" not in store_results:
            print(f"   Store results count: {len(store_results['results'])}")
            for i, result in enumerate(store_results['results']):
                print(f"   Result {i+1}: Distance {result['distance']:.4f} -> Similarity {result['similarity']:.4f}")
        else:
            print(f"   Store error: {store_results['error']}")
        
        # Step 5: Check if any filtering happens
        print("\n5. Checking for filtering:")
        positive_similarities = [r for r in store_results.get('results', []) if r.get('similarity', 0) > 0]
        print(f"   Results with positive similarity: {len(positive_similarities)}")
        
        if len(positive_similarities) == 0:
            print("   ❌ ALL SIMILARITIES ARE NEGATIVE OR ZERO!")
            print("   This explains why RAG says 'no relevant documents'")
        else:
            print("   ✅ Found positive similarities:")
            for r in positive_similarities[:3]:
                print(f"      Similarity: {r['similarity']:.4f}")
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_full_pipeline()
