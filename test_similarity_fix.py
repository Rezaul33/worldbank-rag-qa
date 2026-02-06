"""
Test the similarity fix directly.
"""

import sys
sys.path.append('..')

from vector_store.chroma_db import ChromaVectorStore
from embeddings.embedding_generator import EmbeddingGenerator

def test_similarity_fix():
    """Test if similarity fix is working."""
    print("🧪 Testing Similarity Fix")
    print("=" * 40)
    
    # Initialize components
    vector_store = ChromaVectorStore()
    generator = EmbeddingGenerator()
    
    # Test query
    query = "development"
    print(f"Testing query: '{query}'")
    
    try:
        # Use the vector store query method (same as Streamlit)
        results = vector_store.query(
            query_text=query,
            embedding_model=generator.model,
            n_results=3
        )
        
        print(f"\nResults from vector_store.query():")
        if "error" not in results:
            for i, result in enumerate(results["results"]):
                print(f"  Result {i+1}:")
                print(f"    Distance: {result['distance']:.4f}")
                print(f"    Similarity: {result['similarity']:.4f}")
                print(f"    Document: {result['document'][:80]}...")
                print()
        else:
            print(f"Error: {results['error']}")
            
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    test_similarity_fix()
