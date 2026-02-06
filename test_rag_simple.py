"""
Simple RAG test using existing Chroma database without re-loading models.
"""

import requests
import json
from vector_store.chroma_db import ChromaVectorStore
from embeddings.embedding_generator import EmbeddingGenerator

def test_simple_rag():
    """Test RAG with minimal model loading."""
    
    # Initialize vector store (this should work)
    vector_store = ChromaVectorStore()
    
    # Try to load embedding generator with offline mode
    try:
        generator = EmbeddingGenerator()
        print("Embedding generator loaded successfully")
    except Exception as e:
        print(f"Error loading embedding generator: {e}")
        print("This is likely a network issue with HuggingFace.")
        return
    
    # Test Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            print(f"Ollama connected. Available models: {model_names}")
        else:
            print("Ollama not responding correctly")
            return
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        return
    
    # Test a simple query
    test_query = "What are the main challenges in global development?"
    print(f"\nTesting query: {test_query}")
    
    try:
        # Query Chroma
        results = vector_store.query(test_query, generator.model, n_results=3)
        
        if "error" not in results:
            print(f"Found {len(results['results'])} relevant documents")
            for i, result in enumerate(results["results"], 1):
                print(f"\nResult {i}:")
                print(f"Similarity: {result['similarity']:.4f}")
                print(f"Document: {result['metadata']['filename']}")
                print(f"Page: {result['metadata']['page_number']}")
                print(f"Text preview: {result['document'][:100]}...")
        else:
            print(f"Query error: {results['error']}")
            
    except Exception as e:
        print(f"Error during query: {e}")

if __name__ == "__main__":
    test_simple_rag()
