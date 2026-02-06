"""
Debug Chroma database content and search functionality.
"""

import sys
sys.path.append('..')

from vector_store.chroma_db import ChromaVectorStore
from embeddings.embedding_generator import EmbeddingGenerator

def debug_chroma():
    """Debug Chroma database and search."""
    print("🔍 Debugging Chroma Database")
    print("=" * 50)
    
    # Initialize components
    print("1. Initializing components...")
    vector_store = ChromaVectorStore()
    generator = EmbeddingGenerator()
    
    # Get collection stats
    print("\n2. Collection Statistics:")
    stats = vector_store.get_collection_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test database content
    print("\n3. Testing database content...")
    try:
        # Get a few sample documents
        sample_results = vector_store.collection.get(limit=5)
        print(f"   Sample documents found: {len(sample_results['ids'])}")
        
        if sample_results['documents']:
            print("   Sample document preview:")
            for i, (doc_id, doc, metadata) in enumerate(zip(
                sample_results['ids'][:3], 
                sample_results['documents'][:3], 
                sample_results['metadatas'][:3]
            )):
                print(f"     Doc {i+1}: {doc[:100]}...")
                print(f"     Metadata: {metadata}")
                print()
    except Exception as e:
        print(f"   Error accessing database: {e}")
    
    # Test search with different queries
    test_queries = [
        "development",
        "global", 
        "challenges",
        "climate",
        "economic",
        "world bank"
    ]
    
    print("\n4. Testing search queries:")
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            results = vector_store.query(query, generator.model, n_results=3)
            if "error" not in results:
                print(f"   Results: {len(results['results'])} documents")
                for i, result in enumerate(results['results'][:2]):
                    print(f"     {i+1}. Similarity: {result['similarity']:.4f}")
                    print(f"        Doc: {result['document'][:80]}...")
            else:
                print(f"   Error: {results['error']}")
        except Exception as e:
            print(f"   Search error: {e}")
    
    print("\n5. Testing embedding generation:")
    test_text = "global development challenges"
    try:
        embedding = generator.generate_embedding(test_text)
        print(f"   Generated embedding for: '{test_text}'")
        print(f"   Embedding shape: {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"   Embedding error: {e}")

if __name__ == "__main__":
    debug_chroma()
