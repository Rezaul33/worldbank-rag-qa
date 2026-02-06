"""
Debug Chroma distance calculation to understand the metric being used.
"""

import sys
sys.path.append('..')

from vector_store.chroma_db import ChromaVectorStore
from embeddings.embedding_generator import EmbeddingGenerator

def debug_chroma_distance():
    """Debug Chroma distance calculation."""
    print("🔍 Debugging Chroma Distance Calculation")
    print("=" * 50)
    
    # Initialize components
    vector_store = ChromaVectorStore()
    generator = EmbeddingGenerator()
    
    # Test with known embeddings
    test_query = "development"
    print(f"Testing query: '{test_query}'")
    
    try:
        # Get raw Chroma results
        query_embedding = generator.model.encode(test_query)
        print(f"Query embedding shape: {len(query_embedding)}")
        print(f"Query embedding sample: {query_embedding[:5]}")
        
        # Query Chroma directly
        results = vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"\nRaw Chroma results:")
        print(f"IDs: {results['ids'][0]}")
        print(f"Distances: {results['distances'][0]}")
        
        # Analyze distance values
        distances = results['distances'][0]
        print(f"\nDistance analysis:")
        print(f"Min distance: {min(distances):.4f}")
        print(f"Max distance: {max(distances):.4f}")
        print(f"Mean distance: {sum(distances)/len(distances):.4f}")
        
        # Test different similarity conversions
        print(f"\nSimilarity conversion tests:")
        for i, dist in enumerate(distances):
            old_sim = 1 - dist  # Original wrong method
            new_sim = 1 - (dist / 2)  # New method
            cosine_sim = 1 - dist  # Cosine distance conversion
            euclidean_sim = 1 / (1 + dist)  # Euclidean distance conversion
            
            print(f"  Result {i+1}:")
            print(f"    Distance: {dist:.4f}")
            print(f"    Old similarity: {old_sim:.4f}")
            print(f"    New similarity: {new_sim:.4f}")
            print(f"    Cosine similarity: {cosine_sim:.4f}")
            print(f"    Euclidean similarity: {euclidean_sim:.4f}")
        
        # Check Chroma metadata
        print(f"\nCollection metadata:")
        print(f"Collection name: {vector_store.collection.name}")
        print(f"Collection metadata: {vector_store.collection.metadata}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_chroma_distance()
